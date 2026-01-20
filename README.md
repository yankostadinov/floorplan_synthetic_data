# Floorplan Generator

Part of a research conducted into LLM performance in analysis of residential floor plans:
[Research into LLM 2D floorplan understanding](https://www.notion.so/2D-Floorplan-Understanding-2ee188dee74d804485abffd8512daf65)
Generates simple grid-based apartment floorplans as:

- `PNG` preview image
- `JSON` with doors, adjacency, and furniture

## Requirements

- Python 3.10+
- Pillow

Install dependencies:

```bash
pip install pillow
```

## Usage

```bash
python generate.py --out out_samples --n 10 --seed 123 --grid_width 80 --grid_height 60 --cell_px 10
```

## Outputs (per sample):

    out_samples/sample_000000.png
    out_samples/sample_000000.json

## JSON contents (high level)

    meta: grid size and generation knobs
    rooms: each room has:
        id, type
        bbox (x0,y0,x1,y1) in grid coordinates
    doors: door segments between rooms
    adjacency: wall-sharing adjacency (rooms that share a wall; not necessarily a door)
    door_adjacency: door connectivity graph
    furniture: decorative items placed along boundaries between unconnected rooms

## What is randomized?

With a fixed --seed, the following are generated deterministically:

### Program (layout “requirements”)

    Number of bedrooms and bathrooms
    Which bedrooms get a WIC
    Whether the kitchen has an island
    Whether hallway label is shown
    Which side is “open space” (left/right/top/bottom)
    Hallway cap behavior (cap owner and anchor)
    Which room gets the exterior entrance door (hallway or living)
    Whether kitchen is the “first” side of the open-space split

### Geometry (exact shapes)

    Hallway thickness and length
    Open-space thickness
    Open-space split position (kitchen vs living delimiter)
    Private-room band cuts and assignment order
    WIC carve size and which corner is carved

### Doors / furniture

    Exact door location along the shared boundary
    Door style (arc vs sliding)
    Optional bedroom↔bathroom direct doors (and removal of bathroom↔hallway door)
    Furniture placement along boundaries without doors
    style (arc vs sliding)
    Optional bedroom↔bathroom direct doors (and removal of bathroom↔hallway door)
    Furniture placement along boundaries without doors


## References
[Research into LLM 2D floorplan understanding](https://www.notion.so/2D-Floorplan-Understanding-2ee188dee74d804485abffd8512daf65)
