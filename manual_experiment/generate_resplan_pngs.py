import os
import pickle
import matplotlib.pyplot as plt
import random

from resplan.resplan_utils import normalize_keys, plot_plan

DATA_PATH = "resplan/ResPlan.pkl"


print("Loading ResPlan data...")
with open(DATA_PATH, "rb") as f:
    plans = pickle.load(f)

print(f"Loaded {len(plans)} plans")


PLANS_TO_PLOT_COUNT = 10
PLANS_OUTPUT_DIRECTORY = "plans"
print(
    f"Plotting {PLANS_TO_PLOT_COUNT} plans and saving to /{PLANS_OUTPUT_DIRECTORY}..."
)

os.makedirs(PLANS_OUTPUT_DIRECTORY, exist_ok=True)
plans_to_plot = random.sample(plans, 10)
for index, plan in enumerate(plans_to_plot):
    # Normalize common key typos (e.g., balacony -> balcony) in-place for safety
    normalize_keys(plan)
    plotted_plan = plot_plan(plan, title=f"Plan {index}")
    out_path = os.path.join(PLANS_OUTPUT_DIRECTORY, f"resplan_{index}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
plt.close()
