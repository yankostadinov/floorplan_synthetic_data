import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

RGBA_COLORS = {
    "WHITE": (255, 255, 255, 255),
    "BLACK": (0, 0, 0, 255),
    "BLUE": (232, 242, 255, 255),
    "YELLOW": (255, 244, 230, 255),
    "GREEN": (235, 255, 235, 255),
    "PINK": (235, 245, 245, 255),
    "GREY": (245, 245, 245, 255),
}


Cell = Tuple[int, int]  # (x, y)


@dataclass
class Room:
    id: str
    type: str
    cells: Set[Cell]


@dataclass
class Door:
    rooms: Tuple[str, str]
    style: str  # "arc" | "sliding"
    orientation: str  # "vertical" or "horizontal"
    p0: Cell  # (x, y) start (grid-edge coordinates)
    p1: Cell  # (x, y) end   (grid-edge coordinates)


@dataclass
class Furniture:
    type: str  # "cupboard" | "wardrobe"
    room_id: str
    wall_orientation: str  # "vertical" | "horizontal"
    wall_p0: Cell
    wall_p1: Cell


def build_occupancy(rooms: Dict[str, Room]) -> Dict[Cell, str]:
    occupancy: Dict[Cell, str] = {}
    # sort keys and cells to make sure order is consistent between seeds
    for room_id in sorted(rooms.keys()):
        room = rooms[room_id]
        for cell in sorted(room.cells):
            if cell in occupancy:
                raise ValueError(
                    f"Overlap detected at cell {cell} between {occupancy[cell]} and {
                        room_id
                    }"
                )
            occupancy[cell] = room_id
    return occupancy


def neighbors4(x: int, y: int) -> List[Cell]:
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def shared_room_boundary_edges(
    occupancy: Dict[Cell, str],
    room_a: str,
    room_b: str,
) -> List[Tuple[str, Cell, Cell]]:
    """
    Return list of boundary edges separating room a and b.

    Edge format:
      ("vertical", (x, y0), (x, y1)) for vertical edge at x between y0..y1 (length 1 cell)
      ("horizontal", (x0, y), (x1, y)) for horizontal edge at y between x0..x1 (length 1 cell)
    """
    edges = []
    # sort keys to make sure order is consistent between seeds
    for x, y in sorted(occupancy.keys()):
        if occupancy[(x, y)] != room_a:
            continue
        # right neighbor => vertical edge at x+1
        neighbor_right = (x + 1, y)
        if neighbor_right in occupancy and occupancy[neighbor_right] == room_b:
            edges.append(("vertical", (x + 1, y), (x + 1, y + 1)))
        # left neighbor => vertical edge at x
        neighbor_left = (x - 1, y)
        if neighbor_left in occupancy and occupancy[neighbor_left] == room_b:
            edges.append(("vertical", (x, y), (x, y + 1)))
        # down neighbor => horizontal edge at y+1
        neighbor_down = (x, y + 1)
        if neighbor_down in occupancy and occupancy[neighbor_down] == room_b:
            edges.append(("horizontal", (x, y + 1), (x + 1, y + 1)))
        # up neighbor => horizontal edge at y
        neighbor_up = (x, y - 1)
        if neighbor_up in occupancy and occupancy[neighbor_up] == room_b:
            edges.append(("horizontal", (x, y), (x + 1, y)))
    return sorted(edges)


def exterior_boundary_edges(
    occupancy: Dict[Cell, str],
    room_id: str,
    grid_width: int,
    grid_height: int,
) -> List[Tuple[str, Cell, Cell]]:
    """
    Return boundary edges where room borders the outside of the plan (grid perimeter).
    Edge format:
      ("vertical", (x, y0), (x, y1)) for vertical edge at x between y0..y1 (length 1 cell)
      ("horizontal", (x0, y), (x1, y)) for horizontal edge at y between x0..x1 (length 1 cell)
    """
    edges: List[Tuple[str, Cell, Cell]] = []
    for (x, y), rid in occupancy.items():
        if rid != room_id:
            continue

        # left perimeter => vertical edge at x
        if x == 0:
            edges.append(("vertical", (x, y), (x, y + 1)))
        # right perimeter => vertical edge at x+1
        if x == grid_width - 1:
            edges.append(("vertical", (x + 1, y), (x + 1, y + 1)))

        # top perimeter => horizontal edge at y
        if y == 0:
            edges.append(("horizontal", (x, y), (x + 1, y)))
        # bottom perimeter => horizontal edge at y+1
        if y == grid_height - 1:
            edges.append(("horizontal", (x, y + 1), (x + 1, y + 1)))

    return edges


def wall_adjacency_from_occupancy(
    occupancy: Dict[Cell, str],
    grid_width: int,
    grid_height: int,
) -> Dict[str, Set[str]]:
    """
    Adjacency = rooms that share at least one wall segment (4-neighborhood).
    Not based on doors.
    """
    adjacency: Dict[str, Set[str]] = {}

    def add(a: str, b: str) -> None:
        if a == b:
            return
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    # sort keys to make sure order is consistent between seeds
    for x, y in sorted(occupancy.keys()):
        a = occupancy[(x, y)]

        # right neighbor (avoid double counting by only looking right/down)
        if x + 1 < grid_width:
            b = occupancy.get((x + 1, y))
            if b is not None and b != a:
                add(a, b)

        # down neighbor
        if y + 1 < grid_height:
            b = occupancy.get((x, y + 1))
            if b is not None and b != a:
                add(a, b)

    return adjacency


class FloorplanGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def sample_program(self) -> Dict:
        """
        Returns:
        rooms: list of (id, type)
        door_edges: list of (idA, idB)  # required doors
        meta knobs
        """
        num_bedrooms = self.rng.choice([1, 2, 2, 3])
        num_bathrooms = self.rng.choice(
            [max(1, num_bedrooms - 1), num_bedrooms, num_bedrooms]
        )

        hallway_label_visible = self.rng.choice([True, False])
        want_island = self.rng.random() < 0.55

        open_side = self.rng.choice(["left", "right", "top", "bottom"])

        # hallway doesn't span full length. The leftover "cap" at one end
        # is owned by either open-space or private rooms.
        hallway_cap_owner = self.rng.choice(["open", "private"])
        hallway_cap_anchor = self.rng.choice(
            ["start", "end"]
        )  # start=end depends on axis

        # entrance door placed on hallway or living exterior wall
        entrance_room = self.rng.choice(["H", "L"])

        # open-space split: which side becomes kitchen (vs living) depends on delimiter orientation.
        # "kitchen_first" as "kitchen is on the 'lower index' side of the split"
        # (top/left depending on delimiter).
        kitchen_first = self.rng.random() < 0.5

        rooms = [("L", "living"), ("K", "kitchen"), ("H", "hallway")]
        for i in range(num_bedrooms):
            rooms.append((f"B{i + 1}", "bedroom"))
        for i in range(num_bathrooms):
            rooms.append((f"BA{i + 1}", "bathroom"))

        # add WIC to some bedrooms
        wics = []
        for i in range(num_bedrooms):
            if self.rng.random() < 0.45:
                wic_id = f"WIC{i + 1}"
                wics.append((wic_id, "WIC", f"B{i + 1}"))
                rooms.append((wic_id, "WIC"))

        # door constraints:
        # - kitchen and living are open-space: NO door between them
        # - hallway connects to living with a door
        # - hallway connects to each bedroom and bathroom with a door (geometry will ensure adjacency), unless bathroom is connected to a bedroom
        door_edges = [("L", "H")]
        for i in range(num_bedrooms):
            door_edges.append(("H", f"B{i + 1}"))
        for i in range(num_bathrooms):
            door_edges.append(("H", f"BA{i + 1}"))
        for wic_id, _, bedroom_id in wics:
            door_edges.append((bedroom_id, wic_id))

        return {
            "rooms": rooms,
            "door_edges": door_edges,
            "wics": wics,  # (wic_id, type, bedroom_id)
            "knobs": {
                "hallway_label_visible": hallway_label_visible,
                "kitchen_island": want_island,
                "open_side": open_side,
                "hallway_cap_owner": hallway_cap_owner,
                "hallway_cap_anchor": hallway_cap_anchor,
                "kitchen_first": kitchen_first,
                "entrance_room": entrance_room,
            },
        }

    def realize_geometry(
        self,
        program: Dict,
        grid_width: int = 80,
        grid_height: int = 60,
    ) -> Tuple[Dict[str, Room], Dict]:
        rooms_def = program["rooms"]
        room_types = {rid: rtype for rid, rtype in rooms_def}

        open_side = program["knobs"]["open_side"]
        # "open" | "private"
        cap_owner = program["knobs"]["hallway_cap_owner"]
        # "start" | "end"
        cap_anchor = program["knobs"]["hallway_cap_anchor"]
        kitchen_first = program["knobs"]["kitchen_first"]

        vertical_layout = open_side in ("left", "right")  # hallway is vertical strip
        # sizes
        if vertical_layout:
            hallway_thickness = max(5, int(grid_width * 0.10))
            open_thickness = int(grid_width * self.rng.uniform(0.48, 0.60))
            if grid_width - open_thickness - hallway_thickness < 12:
                open_thickness = max(12, grid_width - hallway_thickness - 12)
        else:
            hallway_thickness = max(5, int(grid_height * 0.12))
            open_thickness = int(grid_height * self.rng.uniform(0.45, 0.58))
            if grid_height - open_thickness - hallway_thickness < 10:
                open_thickness = max(10, grid_height - hallway_thickness - 10)

        # hallway span along its long axis
        if vertical_layout:
            hallway_len = int(grid_height * self.rng.uniform(0.60, 0.88))
            hallway_len = max(hallway_len, max(18, grid_height // 2))
            cap_len = grid_height - hallway_len
            if cap_len < 6:
                cap_len = 0
                hallway_len = grid_height

            if cap_len == 0:
                hallway_y0, hallway_y1 = 0, grid_height
                cap_range = None
            else:
                if cap_anchor == "start":
                    # cap at TOP
                    cap_range = ("top", 0, cap_len)
                    hallway_y0, hallway_y1 = cap_len, grid_height
                else:
                    # cap at BOTTOM
                    cap_range = ("bottom", grid_height - cap_len, grid_height)
                    hallway_y0, hallway_y1 = 0, grid_height - cap_len

            # compute x ranges based on open_side
            if open_side == "left":
                open_x0, open_x1 = 0, open_thickness
                hallway_x0, hallway_x1 = open_x1, open_x1 + hallway_thickness
            else:  # "right"
                open_x0, open_x1 = grid_width - open_thickness, grid_width
                hallway_x0, hallway_x1 = open_x0 - hallway_thickness, open_x0

            # assign open + hallway cells (private computed later by complement)
            open_cells: Set[Cell] = set()
            hallway_cells: Set[Cell] = set()

            for x in range(grid_width):
                for y in range(grid_height):
                    in_cap = False
                    if cap_range is not None:
                        _, c0, c1 = cap_range
                        in_cap = c0 <= y < c1

                    if in_cap:
                        # no hallway in cap; entire cap owned by either open or private.
                        if cap_owner == "open":
                            open_cells.add((x, y))
                        # else: private (skip; will be part of complement)
                    else:
                        # hallway span: normal split open/hall/private
                        if open_x0 <= x < open_x1:
                            open_cells.add((x, y))
                        elif hallway_x0 <= x < hallway_x1:
                            hallway_cells.add((x, y))
                        else:
                            # private (skip; complement)
                            pass

        else:
            # horizontal hallway
            hallway_len = int(grid_width * self.rng.uniform(0.60, 0.88))
            hallway_len = max(hallway_len, max(22, grid_width // 2))
            cap_len = grid_width - hallway_len
            if cap_len < 8:
                cap_len = 0
                hallway_len = grid_width

            if cap_len == 0:
                hallway_x0, hallway_x1 = 0, grid_width
                cap_range = None
            else:
                if cap_anchor == "start":
                    # cap at LEFT
                    cap_range = ("left", 0, cap_len)
                    hallway_x0, hallway_x1 = cap_len, grid_width
                else:
                    # cap at RIGHT
                    cap_range = ("right", grid_width - cap_len, grid_width)
                    hallway_x0, hallway_x1 = 0, grid_width - cap_len

            if open_side == "top":
                open_y0, open_y1 = 0, open_thickness
                hallway_y0, hallway_y1 = open_y1, open_y1 + hallway_thickness
            else:  # "bottom"
                open_y0, open_y1 = grid_height - open_thickness, grid_height
                hallway_y0, hallway_y1 = open_y0 - hallway_thickness, open_y0

            open_cells: Set[Cell] = set()
            hallway_cells: Set[Cell] = set()

            for x in range(grid_width):
                for y in range(grid_height):
                    in_cap = False
                    if cap_range is not None:
                        _, c0, c1 = cap_range
                        in_cap = c0 <= x < c1

                    if in_cap:
                        if cap_owner == "open":
                            open_cells.add((x, y))
                    else:
                        if open_y0 <= y < open_y1:
                            open_cells.add((x, y))
                        elif hallway_y0 <= y < hallway_y1:
                            hallway_cells.add((x, y))
                        else:
                            pass

        # PRIVATE REGION = complement of (open + hallway)
        all_cells = {(x, y) for x in range(grid_width) for y in range(grid_height)}
        private_region_cells = all_cells - open_cells - hallway_cells

        # split open_cells into kitchen/living (delimiter inside open-space, not a wall)
        xs_open = [x for x, _ in open_cells]
        ys_open = [y for _, y in open_cells]
        ox0, ox1 = min(xs_open), max(xs_open) + 1
        oy0, oy1 = min(ys_open), max(ys_open) + 1

        living_cells: Set[Cell] = set()
        kitchen_cells: Set[Cell] = set()

        if vertical_layout:
            # horizontal delimiter; force it to cut within hallway span if possible
            if cap_range is not None:
                # use hallway y-range to ensure both rooms touch hallway area
                if cap_range[0] == "top":
                    cy0, cy1 = hallway_y0, hallway_y1
                else:
                    cy0, cy1 = hallway_y0, hallway_y1
            else:
                cy0, cy1 = 0, grid_height

            margin = 4
            if cy1 - cy0 <= 2 * margin + 2:
                split_y = (oy0 + oy1) // 2
            else:
                split_y = self.rng.randint(cy0 + margin, cy1 - margin)

            for x, y in open_cells:
                a_side = y < split_y  # "top"
                if kitchen_first:
                    (kitchen_cells if a_side else living_cells).add((x, y))
                else:
                    (living_cells if a_side else kitchen_cells).add((x, y))

            delimiter = {
                "orientation": "horizontal",
                "p0": (ox0, split_y),
                "p1": (ox1, split_y),
            }
        else:
            # vertical delimiter
            if cap_range is not None:
                cx0, cx1 = hallway_x0, hallway_x1
            else:
                cx0, cx1 = 0, grid_width

            margin = 5
            if cx1 - cx0 <= 2 * margin + 2:
                split_x = (ox0 + ox1) // 2
            else:
                split_x = self.rng.randint(cx0 + margin, cx1 - margin)

            for x, y in open_cells:
                a_side = x < split_x  # "left"
                if kitchen_first:
                    (kitchen_cells if a_side else living_cells).add((x, y))
                else:
                    (living_cells if a_side else kitchen_cells).add((x, y))

            delimiter = {
                "orientation": "vertical",
                "p0": (split_x, oy0),
                "p1": (split_x, oy1),
            }

        # allocate private rooms (bed/bath) within the private region.
        private_ids = [
            room_id
            for room_id, room_type in rooms_def
            if room_type in ("bedroom", "bathroom")
        ]
        self.rng.shuffle(private_ids)

        n = len(private_ids)
        private_rooms_cells: Dict[str, Set[Cell]] = {rid: set() for rid in private_ids}

        # build bands within the hallway span (the part where hallway exists),
        # and optionally extend one band into the "cap" when cap_owner=="private".
        if vertical_layout:
            # determine x-range of private side within hallway span
            if open_side == "left":
                core_x0, core_x1 = hallway_x1, grid_width
            else:
                core_x0, core_x1 = 0, hallway_x0

            # determine y span where private exists
            if cap_owner == "open" and cap_range is not None:
                # cap is all open => private only in hallway y-range
                span_y0, span_y1 = hallway_y0, hallway_y1
                cap_y0, cap_y1 = None, None
            elif cap_owner == "private" and cap_range is not None:
                # cap is private => rooms must fit hallway but one room can extend into cap
                span_y0, span_y1 = hallway_y0, hallway_y1
                cap_y0, cap_y1 = cap_range[1], cap_range[2]
            else:
                span_y0, span_y1 = 0, grid_height
                cap_y0, cap_y1 = None, None

            span_h = span_y1 - span_y0
            min_height = max(8, span_h // (n + 2))

            cuts = sorted(
                self.rng.sample(
                    range(span_y0 + min_height, span_y1 - min_height),
                    k=max(0, n - 1),
                )
            )
            ys = [span_y0] + cuts + [span_y1]
            bands = [(ys[i], ys[i + 1]) for i in range(len(ys) - 1)]

            # merge/split to match n
            while len(bands) > n:
                i = self.rng.randrange(0, len(bands) - 1)
                bands[i] = (bands[i][0], bands[i + 1][1])
                bands.pop(i + 1)
            while len(bands) < n:
                i = self.rng.randrange(0, len(bands))
                y0, y1 = bands[i]
                if y1 - y0 <= 2 * min_height:
                    break
                mid = (y0 + y1) // 2
                bands[i] = (y0, mid)
                bands.insert(i + 1, (mid, y1))

            for rid, (y0, y1) in zip(private_ids, bands):
                for x in range(core_x0, core_x1):
                    for y in range(y0, y1):
                        private_rooms_cells[rid].add((x, y))

            # if cap is private, extend a band to cover the cap (full width).
            if cap_owner == "private" and cap_range is not None:
                # attach cap to the nearest band (top cap -> first band, bottom cap -> last band)
                attach_rid = (
                    private_ids[0] if cap_range[0] == "top" else private_ids[-1]
                )
                for x in range(grid_width):
                    for y in range(cap_y0, cap_y1):
                        private_rooms_cells[attach_rid].add((x, y))

        else:
            # horizontal layout: slice bands along x
            if open_side == "top":
                core_y0, core_y1 = hallway_y1, grid_height
            else:
                core_y0, core_y1 = 0, hallway_y0

            if cap_owner == "open" and cap_range is not None:
                span_x0, span_x1 = hallway_x0, hallway_x1
                cap_x0, cap_x1 = None, None
            elif cap_owner == "private" and cap_range is not None:
                span_x0, span_x1 = hallway_x0, hallway_x1
                cap_x0, cap_x1 = cap_range[1], cap_range[2]
            else:
                span_x0, span_x1 = 0, grid_width
                cap_x0, cap_x1 = None, None

            span_w = span_x1 - span_x0
            min_width = max(10, span_w // (n + 2))

            cuts = sorted(
                self.rng.sample(
                    range(span_x0 + min_width, span_x1 - min_width),
                    k=max(0, n - 1),
                )
            )
            xs = [span_x0] + cuts + [span_x1]
            bands = [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]

            while len(bands) > n:
                i = self.rng.randrange(0, len(bands) - 1)
                bands[i] = (bands[i][0], bands[i + 1][1])
                bands.pop(i + 1)
            while len(bands) < n:
                i = self.rng.randrange(0, len(bands))
                x0, x1 = bands[i]
                if x1 - x0 <= 2 * min_width:
                    break
                mid = (x0 + x1) // 2
                bands[i] = (x0, mid)
                bands.insert(i + 1, (mid, x1))

            for rid, (x0, x1) in zip(private_ids, bands):
                for x in range(x0, x1):
                    for y in range(core_y0, core_y1):
                        private_rooms_cells[rid].add((x, y))

            if cap_owner == "private" and cap_range is not None:
                attach_rid = (
                    private_ids[0] if cap_range[0] == "left" else private_ids[-1]
                )
                for x in range(cap_x0, cap_x1):
                    for y in range(grid_height):
                        private_rooms_cells[attach_rid].add((x, y))

        # ensure private region is fully covered by private rooms
        covered_private = (
            set().union(*private_rooms_cells.values()) if private_rooms_cells else set()
        )
        missing = private_region_cells - covered_private
        if missing:
            # assign missing cells to a random private room to keep coverage consistent.
            # (This can happen due to edge-case sizing.)
            if private_ids:
                private_rooms_cells[self.rng.choice(private_ids)] |= missing

        # carve WIC(s) from bedrooms in a corner away from the hallway-facing side
        wic_defs = program["wics"]  # (wic_id, type, bedroom_id)
        wic_rooms_cells: Dict[str, Set[Cell]] = {}

        def carve_corner_wic(
            bed_cells: Set[Cell],
            # "left"|"right"|"up"|"down" meaning hallway-facing side, so we carve away from it
            away_side: str,
        ) -> Optional[Set[Cell]]:
            if len(bed_cells) < 80:
                return None

            xs = [x for x, _ in bed_cells]
            ys = [y for _, y in bed_cells]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)

            bw = x1 - x0 + 1
            bh = y1 - y0 + 1

            carve_w = max(4, int(bw * self.rng.uniform(0.25, 0.40)))
            carve_h = max(5, int(bh * self.rng.uniform(0.30, 0.55)))

            # candidate corners away from hallway-facing side (so WIC is away from door wall)
            # corners expressed as (use_left, use_top)
            if away_side == "left":
                # right-top, right-bottom
                corners = [(False, True), (False, False)]
            elif away_side == "right":
                # left-top, left-bottom
                corners = [(True, True), (True, False)]
            elif away_side == "up":
                # bottom-left, bottom-right
                corners = [(True, False), (False, False)]
            else:  # "down"
                # top-left, top-right
                corners = [(True, True), (False, True)]

            use_left, use_top = self.rng.choice(corners)

            cx0 = x0 if use_left else (x1 - carve_w + 1)
            cy0 = y0 if use_top else (y1 - carve_h + 1)
            cx1 = cx0 + carve_w
            cy1 = cy0 + carve_h

            carve = {(x, y) for x in range(cx0, cx1) for y in range(cy0, cy1)}
            carve = carve.intersection(bed_cells)

            # must be meaningful size and "touch the wall" (corner placement does that;
            # still enforce that some cells lie on bbox boundary)
            if len(carve) < 20:
                return None
            return carve

        # determine hallway-facing direction for private rooms
        # (this is the side where bedroom door to hallway will be, so WIC should be away from it)
        if vertical_layout:
            hallway_facing = "left" if open_side == "left" else "right"
        else:
            hallway_facing = "up" if open_side == "top" else "down"

        for wic_id, _t, bed_id in wic_defs:
            bed_cells = private_rooms_cells.get(bed_id, set())
            carve = carve_corner_wic(bed_cells, away_side=hallway_facing)
            if not carve:
                continue
            private_rooms_cells[bed_id] = bed_cells - carve
            wic_rooms_cells[wic_id] = carve

        # compose final rooms
        rooms: Dict[str, Room] = {
            "L": Room("L", "living", living_cells),
            "K": Room("K", "kitchen", kitchen_cells),
            "H": Room("H", "hallway", hallway_cells),
        }
        for rid, cells in private_rooms_cells.items():
            rooms[rid] = Room(rid, room_types[rid], cells)
        for rid, cells in wic_rooms_cells.items():
            rooms[rid] = Room(rid, "WIC", cells)

        hints = {
            "delimiter": delimiter,
            "open_side": open_side,
            "vertical_layout": vertical_layout,
            "hallway_cap_owner": cap_owner,
            "hallway_cap_anchor": cap_anchor,
        }
        return rooms, hints

    def place_doors(
        self,
        rooms: Dict[str, Room],
        program: Dict,
        grid_width: int,
        grid_height: int,
    ) -> List[Door]:
        occupancy = build_occupancy(rooms)
        doors: List[Door] = []

        def add_one_door(room_a: str, room_b: str) -> None:
            edges = shared_room_boundary_edges(occupancy, room_a, room_b)
            if not edges:
                return
            orientation, p0, p1 = self.rng.choice(edges)
            style = "sliding" if self.rng.random() < 0.35 else "arc"
            doors.append(
                Door(
                    rooms=(room_a, room_b),
                    style=style,
                    orientation=orientation,
                    p0=p0,
                    p1=p1,
                )
            )

        def remove_one_door(room_a: str, room_b: str) -> None:
            door = [
                door for door in doors if room_a in door.rooms and room_b in door.rooms
            ][0]
            if not door:
                return
            doors.remove(door)

        # required doors from program
        for room_a, room_b in program["door_edges"]:
            if room_a not in rooms or room_b not in rooms:
                continue
            add_one_door(room_a, room_b)

        # doors directly between adjacent bedrooms and bathrooms
        # (only if they share a boundary)
        bedroom_ids = [rid for rid, r in rooms.items() if r.type == "bedroom"]
        bathroom_ids = [rid for rid, r in rooms.items() if r.type == "bathroom"]
        hallway_id = "H"

        existing = {tuple(sorted(d.rooms)) for d in doors}
        for bedroom in bedroom_ids:
            for bathroom in bathroom_ids:
                key = tuple(sorted((bedroom, bathroom)))
                if key in existing:
                    continue
                edges = shared_room_boundary_edges(occupancy, bedroom, bathroom)
                if not edges:
                    continue
                # probability knob
                if self.rng.random() < 0.35:
                    add_one_door(bedroom, bathroom)
                    existing.add(key)
                    key_to_remove = tuple(sorted((bathroom, hallway_id)))
                    existing.remove(key_to_remove)
                    remove_one_door(bathroom, hallway_id)

        # exterior entrance door on hallway OR living room
        entrance_room = program["knobs"].get("entrance_room", hallway_id)
        if entrance_room in rooms:
            ext_edges = exterior_boundary_edges(
                occupancy, entrance_room, grid_width, grid_height
            )
            if ext_edges:
                orientation, p0, p1 = self.rng.choice(ext_edges)
                doors.append(
                    Door(
                        rooms=(entrance_room, "OUT"),
                        style="arc",
                        orientation=orientation,
                        p0=p0,
                        p1=p1,
                    )
                )

        return doors

    def place_furniture_between_unconnected(
        self,
        rooms: Dict[str, Room],
        doors: List[Door],
        grid_width: int,
        grid_height: int,
        max_items: int = 6,
    ) -> List[Furniture]:
        occupancy = build_occupancy(rooms)
        door_set = {tuple(sorted(d.rooms)) for d in doors}

        # collect candidate boundaries (room pairs with shared wall but no door)
        # then sample random boundary edges and keep those that separate unconnected rooms.
        room_candidates: List[Tuple[str, str, str, Cell, Cell]] = []

        # scan all cell adjacencies
        for (x, y), room_a in occupancy.items():
            # right neighbor boundary
            if x + 1 < grid_width and (x + 1, y) in occupancy:
                room_b = occupancy[(x + 1, y)]
                if room_a != room_b and tuple(sorted((room_a, room_b))) not in door_set:
                    room_candidates.append(
                        (room_a, room_b, "vertical", (x + 1, y), (x + 1, y + 1))
                    )
            # down neighbor boundary
            if y + 1 < grid_height and (x, y + 1) in occupancy:
                room_b = occupancy[(x, y + 1)]
                if room_a != room_b and tuple(sorted((room_a, room_b))) not in door_set:
                    room_candidates.append(
                        (room_a, room_b, "horizontal", (x, y + 1), (x + 1, y + 1))
                    )

        self.rng.shuffle(room_candidates)
        furn: List[Furniture] = []
        for room_a, room_b, orientation, p0, p1 in room_candidates:
            if len(furn) >= max_items:
                break
            # place furniture in one of the rooms (random)
            room_to_place_in = room_a if self.rng.random() < 0.5 else room_b
            furniture_type = "wardrobe" if self.rng.random() < 0.55 else "cupboard"
            furn.append(
                Furniture(
                    type=furniture_type,
                    room_id=room_to_place_in,
                    wall_orientation=orientation,
                    wall_p0=p0,
                    wall_p1=p1,
                )
            )
        return furn

    def validate(
        self,
        rooms: Dict[str, Room],
        doors: List[Door],
    ) -> None:
        occupancy = build_occupancy(rooms)  # overlap check

        # door boundary check: must separate exactly the two intended rooms
        for d in doors:
            a, b = d.rooms
            p0 = d.p0

            if d.orientation == "vertical":
                x = p0[0]
                y = p0[1]
                left_cell = (x - 1, y)
                right_cell = (x, y)
                room_a = occupancy.get(left_cell)
                room_b = occupancy.get(right_cell)
            else:
                x = p0[0]
                y = p0[1]
                up_cell = (x, y - 1)
                down_cell = (x, y)
                room_a = occupancy.get(up_cell)
                room_b = occupancy.get(down_cell)

            # interior door
            if "OUT" not in (a, b):
                if room_a is None or room_b is None:
                    raise ValueError(f"Door not on interior boundary: {d}")
                if set((room_a, room_b)) != set((a, b)):
                    raise ValueError(
                        f"Door separates {room_a}<->{room_b} but declared {a}<->{b}: {
                            d
                        }"
                    )
                continue

            # exterior door: exactly one side must be outside
            inside = b if a == "OUT" else a
            if (room_a is None) == (room_b is None):
                raise ValueError(
                    f"Exterior door must have exactly one side outside: {d} (ra={
                        room_a
                    }, room_b={room_b})"
                )

            actual_inside = room_a if room_b is None else room_b
            if actual_inside != inside:
                raise ValueError(
                    f"Exterior door inside-room mismatch: expected {inside}, got {
                        actual_inside
                    }: {d}"
                )

    def render_png(
        self,
        rooms: Dict[str, Room],
        doors: List[Door],
        furniture: List[Furniture],
        hints: Dict,
        knobs: Dict,
        out_path: str,
        cell_px: int = 10,
        margin_px: int = 10,
    ) -> None:
        # compute grid bounds
        # (Assume consistent grid; infer from cells)
        all_cells = set()
        for r in rooms.values():
            all_cells.update(r.cells)
        max_x = max(x for x, _ in all_cells) + 1
        max_y = max(y for _, y in all_cells) + 1
        grid_width, grid_height = max_x, max_y

        img_width = grid_width * cell_px + 2 * margin_px
        img_height = grid_height * cell_px + 2 * margin_px
        img = Image.new("RGBA", (img_width, img_height), RGBA_COLORS["WHITE"])
        draw = ImageDraw.Draw(img)

        # try a default font
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # room colors
        palette = {
            "living": RGBA_COLORS["BLUE"],
            "kitchen": RGBA_COLORS["YELLOW"],
            "hallway": RGBA_COLORS["GREY"],
            "bedroom": RGBA_COLORS["GREEN"],
            "bathroom": RGBA_COLORS["PINK"],
            "WIC": RGBA_COLORS["PINK"],
        }

        def cell_to_px(x: int, y: int) -> Tuple[int, int, int, int]:
            x0 = margin_px + x * cell_px
            y0 = margin_px + y * cell_px
            return (x0, y0, x0 + cell_px, y0 + cell_px)

        # fill room cells
        for room in rooms.values():
            col = palette.get(room.type, RGBA_COLORS["GREY"])
            for x, y in room.cells:
                draw.rectangle(cell_to_px(x, y), fill=col, outline=None)

        # draw walls between different rooms + outer boundary walls
        occupancy = build_occupancy(rooms)
        wall_col = RGBA_COLORS["BLACK"]
        draw.line(
            [(margin_px, margin_px), ((grid_width + 1) * cell_px, margin_px)],
            fill=wall_col,
            width=2,
        )
        draw.line(
            [(margin_px, margin_px), (margin_px, (grid_height + 1) * cell_px)],
            fill=wall_col,
            width=2,
        )

        # draw grid-edge walls by scanning cells
        for (x, y), room_a in occupancy.items():
            # check right edge
            if x + 1 >= grid_width or (x + 1, y) not in occupancy:
                # outer boundary
                xpx = margin_px + (x + 1) * cell_px
                y0 = margin_px + y * cell_px
                y1 = y0 + cell_px
                draw.line([(xpx, y0), (xpx, y1)], fill=wall_col, width=2)
            else:
                room_b = occupancy[(x + 1, y)]
                if room_a != room_b:
                    xpx = margin_px + (x + 1) * cell_px
                    y0 = margin_px + y * cell_px
                    y1 = y0 + cell_px
                    draw.line([(xpx, y0), (xpx, y1)], fill=wall_col, width=2)

            # check bottom edge
            if y + 1 >= grid_height or (x, y + 1) not in occupancy:
                ypx = margin_px + (y + 1) * cell_px
                x0 = margin_px + x * cell_px
                x1 = x0 + cell_px
                draw.line([(x0, ypx), (x1, ypx)], fill=wall_col, width=2)
            else:
                room_b = occupancy[(x, y + 1)]
                if room_a != room_b:
                    ypx = margin_px + (y + 1) * cell_px
                    x0 = margin_px + x * cell_px
                    x1 = x0 + cell_px
                    draw.line([(x0, ypx), (x1, ypx)], fill=wall_col, width=2)

        # draw open-space delimiter (dotted line between Living and Kitchen)
        d = hints["delimiter"]
        if d["orientation"] == "horizontal":
            y = d["p0"][1]
            ypx = margin_px + y * cell_px
            x0 = margin_px + d["p0"][0] * cell_px
            x1 = margin_px + d["p1"][0] * cell_px
            step = 8
            seg = 4
            for x in range(x0 + 2, x1 - 2, step):
                draw.line(
                    [(x, ypx), (min(x + seg, x1), ypx)],
                    fill=RGBA_COLORS["WHITE"],
                    width=2,
                )
        else:  # vertical
            x = d["p0"][0]
            xpx = margin_px + 1 + x * cell_px
            y0 = margin_px + d["p0"][1] * cell_px
            y1 = margin_px + d["p1"][1] * cell_px
            step = 8
            seg = 4
            for y in range(y0 + 2, y1 - 2, step):
                draw.line(
                    [(xpx, min(y + seg, y1)), (xpx, y)],
                    fill=RGBA_COLORS["WHITE"],
                    width=2,
                )

        def edge_to_px(orientation: str, p0: Cell, p1: Cell) -> Tuple[Cell, Cell]:
            if orientation == "vertical":
                x = p0[0]
                y0 = p0[1]
                y1 = p1[1]
                xpx = margin_px + x * cell_px
                return (xpx, margin_px + y0 * cell_px), (xpx, margin_px + y1 * cell_px)
            else:
                y = p0[1]
                x0 = p0[0]
                x1 = p1[0]
                ypx = margin_px + y * cell_px
                return (margin_px + x0 * cell_px, ypx), (margin_px + x1 * cell_px, ypx)

        for d in doors:
            pxa, pxb = edge_to_px(d.orientation, d.p0, d.p1)

            # erase wall
            draw.line([pxa, pxb], fill=RGBA_COLORS["WHITE"], width=6)

            # draw door style
            if d.style == "sliding":
                # draw a parallel short line
                if d.orientation == "vertical":
                    (x, y0), (_, y1) = pxa, pxb
                    draw.line(
                        [(x + 3, y0 + cell_px), (x + 3, y1 + cell_px)],
                        fill=wall_col,
                        width=2,
                    )
                else:
                    (x0, y), (x1, _) = pxa, pxb
                    draw.line(
                        [(x0 + cell_px, y + 3), (x1 + cell_px, y + 3)],
                        fill=wall_col,
                        width=2,
                    )
            else:
                # arc swing approximation (simple quarter circle)
                # choose arc center near midpoint
                mx = (pxa[0] + pxb[0]) // 2
                my = (pxa[1] + pxb[1]) // 2
                r = int(cell_px * 0.7)
                bbox = [mx - r, my - r, mx + r, my + r]
                draw.arc(bbox, start=0, end=90, fill=wall_col, width=2)

        # kitchen island diversity knob
        if knobs.get("kitchen_island"):
            # place island between kitchen and living room
            kitchen = rooms["K"]
            kitchen_first = knobs.get("kitchen_first")
            xs = [x for x, _ in kitchen.cells]
            ys = [y for _, y in kitchen.cells]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            island_width, island_height = 6, 4
            vertical_cell_offset = 5 if kitchen_first else -5
            x0 = max(0, cx - (island_width // 2))
            y0 = max(0, vertical_cell_offset + (cy - island_height // 2))
            x1 = min(grid_width - 1, x0 + island_width)
            y1 = min(grid_height - 1, y0 + island_height)
            draw.rectangle(
                (
                    margin_px + x0 * cell_px,
                    margin_px + y0 * cell_px,
                    margin_px + x1 * cell_px,
                    margin_px + y1 * cell_px,
                ),
                outline=wall_col,
                fill=RGBA_COLORS["GREY"],
                width=2,
            )

        # furniture: draw small blocks near the wall and mark an "opening"
        for furn in furniture:
            room = rooms.get(furn.room_id)
            if not room:
                continue
            # take wall midpoint and place furniture just inside room by 1 cell
            (a0, a1) = edge_to_px(furn.wall_orientation, furn.wall_p0, furn.wall_p1)
            mx = (a0[0] + a1[0]) // 2
            my = (a0[1] + a1[1]) // 2
            width = (
                int(cell_px * 2.2)
                if furn.wall_orientation == "vertical"
                else int(cell_px)
            )
            height = (
                int(cell_px)
                if furn.wall_orientation == "vertical"
                else int(cell_px * 2.2)
            )
            draw.rectangle(
                [
                    mx + 1,
                    my + 1,
                    mx + 1 + height,
                    my + 1 + width,
                ],
                outline=wall_col,
                width=1,
                fill=RGBA_COLORS["GREY"],
            )
            # opening indicator on wall
            # arc swing approximation (simple quarter circle)
            r = int(cell_px * 0.7)
            x_adjustment = (
                height / 2 if furn.wall_orientation == "horizontal" else height
            )
            y_adjustment = width / 2 if furn.wall_orientation == "vertical" else width
            bbox = [
                mx + 1 + x_adjustment - r,
                my + 1 + y_adjustment - r,
                mx + 1 + x_adjustment + r,
                my + 1 + y_adjustment + r,
            ]
            draw.arc(bbox, start=0, end=90, fill=wall_col, width=1)

        # labels (hallway label may be hidden)
        for room in rooms.values():
            if room.type == "hallway" and not knobs.get("hallway_label_visible", True):
                continue
            xs = [x for x, _ in room.cells]
            ys = [y for _, y in room.cells]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            txt = f"{room.type}:{room.id}"
            draw.text(
                (margin_px + cx * cell_px + 2, margin_px + cy * cell_px + 2),
                txt,
                fill=(0, 0, 0, 255),
                font=font,
            )

        img.save(out_path)

    def generate_one(
        self,
        out_dir: str,
        index: int,
        seed: int,
        grid_width: int = 80,
        grid_height: int = 60,
        cell_px: int = 10,
    ) -> Tuple[str, str]:
        os.makedirs(out_dir, exist_ok=True)

        program = self.sample_program()
        rooms, hints = self.realize_geometry(
            program, grid_width=grid_width, grid_height=grid_height
        )
        doors = self.place_doors(rooms, program, grid_width, grid_height)

        furniture = self.place_furniture_between_unconnected(
            rooms, doors, grid_width=grid_width, grid_height=grid_height
        )

        occupancy = build_occupancy(rooms)

        wall_adjacency = wall_adjacency_from_occupancy(
            occupancy, grid_width=grid_width, grid_height=grid_height
        )

        self.validate(rooms, doors)

        # write outputs
        png_path = os.path.join(out_dir, f"sample_{index:06d}.png")
        json_path = os.path.join(out_dir, f"sample_{index:06d}.json")

        self.render_png(
            rooms=rooms,
            doors=doors,
            furniture=furniture,
            hints=hints,
            knobs=program["knobs"],
            out_path=png_path,
            cell_px=cell_px,
        )

        # JSON
        rooms_json = []
        for rid, r in rooms.items():
            xs = [x for x, _ in r.cells]
            ys = [y for _, y in r.cells]
            bbox = [
                min(xs),
                min(ys),
                max(xs) + 1,
                max(ys) + 1,
            ]  # x0,y0,x1,y1 excl
            rooms_json.append(
                {
                    "id": r.id,
                    "type": r.type,
                    "bbox": bbox,
                }
            )

        doors_json = []
        for d in doors:
            doors_json.append(
                {
                    "rooms": list(d.rooms),
                    "style": d.style,
                    "orientation": d.orientation,
                    "wall_segment": {"p0": list(d.p0), "p1": list(d.p1)},
                }
            )

        adjacency_json = []
        for rid in sorted(rooms.keys()):
            adjacency_json.append(
                {"id": rid, "adjacent": sorted(wall_adjacency.get(rid, set()))}
            )

        furniture_json = []
        for f in furniture:
            furniture_json.append(
                {
                    "type": f.type,
                    "room_id": f.room_id,
                    "wall_orientation": f.wall_orientation,
                    "wall_segment": {
                        "p0": list(f.wall_p0),
                        "p1": list(f.wall_p1),
                    },
                }
            )

        payload = {
            "meta": {
                "seed": seed,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "cell_px": cell_px,
                "knobs": program["knobs"],
            },
            "rooms": rooms_json,
            "doors": doors_json,
            "adjacency": adjacency_json,
            "furniture": furniture_json,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return png_path, json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="out_samples")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--grid_width", type=int, default=80)
    ap.add_argument("--grid_height", type=int, default=60)
    ap.add_argument("--cell_px", type=int, default=10)
    args = ap.parse_args()

    gen = FloorplanGenerator(seed=args.seed)
    for i in range(args.n):
        png_path, json_path = gen.generate_one(
            out_dir=args.out,
            index=i,
            grid_width=args.grid_width,
            grid_height=args.grid_height,
            cell_px=args.cell_px,
            seed=args.seed,
        )
        print(f"Wrote: {png_path} | {json_path}")


if __name__ == "__main__":
    main()
