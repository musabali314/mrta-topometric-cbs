"""
eval_policies.py
----------------

Evaluate:
    - scoba_cbs
    - greedy_astar
    - rl1   (Q-learning)

Outputs:
    * pretty console table
    * CSV results file (optional)
    * optional plots

Uses: run_episode() from main_simulation.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from main_simulation import run_episode


# =====================================================================
#   CONFIG
# =====================================================================

POLICIES = ["scoba_cbs", "greedy_astar", "rl1"]

NUM_EPISODES = 10      # run 10 seeds per policy
SEED_BASE = 100        # starting seed
NUM_AGENTS = 3
NUM_TASKS = 6

SAVE_CSV = True
CSV_FILE = "eval_results.csv"

MAKE_PLOTS = True


# =====================================================================
#   RUN SINGLE POLICY MULTIPLE SEEDS
# =====================================================================

# =====================================================================
#   RUN SINGLE POLICY MULTIPLE SEEDS
# =====================================================================

def eval_policy(policy_name):
    metrics_list = []

    print(f"\n\n==== Evaluating {policy_name} ====\n")

    for k in range(NUM_EPISODES):
        seed = SEED_BASE + k
        print(f"[{policy_name}] Episode {k+1}/{NUM_EPISODES} (seed={seed})")

        m = run_episode(
            policy_type=policy_name,
            seed=seed,
            visualize=False,
            headless=True,
            num_agents=NUM_AGENTS,
            num_tasks=NUM_TASKS,
            dt=0.20,
            max_steps=1000,
            task_reach_eps=0.4,
            debug_cbs=False
        )

        # ----------------------------
        # ADD DERIVED METRICS HERE
        # ----------------------------
        m["utility_per_distance"] = (
            m["total_utility"] / m["total_distance"]
            if m["total_distance"] > 0 else 0.0
        )

        m["utility_per_step"] = (
            m["total_utility"] / m["steps"]
            if m["steps"] > 0 else 0.0
        )

        N = NUM_AGENTS
        m["collision_free_rate"] = (
            1.0 - min(1.0, m["collisions"] / (m["steps"] * N + 1e-6))
        )

        # Placeholder — until real deadlines are handled
        m["deadline_satisfaction"] = m["fraction_tasks_completed"]

        metrics_list.append(m)

    return metrics_list


# =====================================================================
#   AGGREGATE METRICS
# =====================================================================

def summarize(metrics_list):
    summary = {}
    sample = metrics_list[0]

    for key, value in sample.items():
        if isinstance(value, (int, float)):
            arr = np.array([m[key] for m in metrics_list], dtype=float)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

    return summary


# =====================================================================
#   PRETTY PRINT SUMMARY
# =====================================================================

def print_summary(policy_name, summary):
    print("\n-------------------------------------------------")
    print(f"  SUMMARY FOR {policy_name}")
    print("-------------------------------------------------")

    fields_to_show = [
        "tasks_completed",
        "fraction_tasks_completed",
        "total_utility",
        "collisions",
        "steps",
        "total_distance",
        "utility_per_distance",
        "utility_per_step",
        "collision_free_rate",
        "deadline_satisfaction",
    ]

    for f in fields_to_show:
        d = summary[f]
        print(
            f"{f:25s}: mean={d['mean']:.3f}  "
            f"std={d['std']:.3f}  min={d['min']:.3f}  max={d['max']:.3f}"
        )


# =====================================================================
#   BATCH EVALUATION WRAPPER (needed by figures.py)
# =====================================================================

def evaluate_many(policies, N=10):
    """
    Runs N episodes per policy and returns a flat list:
        [
           {"policy": p, "tasks_completed": ..., ...},
           ...
        ]
    Used by figures.py to generate comparison plots.
    """
    results = []

    for policy_name in policies:
        print(f"\n=== Evaluate_Many: {policy_name} ===")

        for k in range(N):
            seed = SEED_BASE + k
            print(f"  [{policy_name}] episode {k+1}/{N} (seed={seed})")

            m = run_episode(
                policy_type=policy_name,
                seed=seed,
                visualize=False,
                headless=True,
                num_agents=NUM_AGENTS,
                num_tasks=NUM_TASKS,
                dt=0.20,
                max_steps=1000,
                task_reach_eps=0.4,
                debug_cbs=False
            )

            out = dict(m)
            out["policy"] = policy_name
            out["seed"] = seed
            results.append(out)

    return results



# =====================================================================
#   MAIN
# =====================================================================

def main():
    all_results = {}   # policy → metrics_list

    for p in POLICIES:
        results = eval_policy(p)
        all_results[p] = results

        summary = summarize(results)
        print_summary(p, summary)

    # ---------------------------------------
    # Save CSV of all episodes
    # ---------------------------------------
    if SAVE_CSV:
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            # header
            header = ["policy", "seed", "tasks_completed", "total_tasks",
                      "fraction_tasks_completed", "total_utility",
                      "collisions", "steps", "total_distance"]
            writer.writerow(header)

            # rows
            for p in POLICIES:
                for i, m in enumerate(all_results[p]):
                    seed = SEED_BASE + i
                    row = [
                        p, seed,
                        m["tasks_completed"], m["total_tasks"],
                        m["fraction_tasks_completed"],
                        m["total_utility"],
                        m["collisions"],
                        m["steps"],
                        m["total_distance"],
                    ]
                    writer.writerow(row)

        print(f"\nCSV saved to: {CSV_FILE}")

    # ---------------------------------------
    # Optional: PLOTS
    # ---------------------------------------
    if MAKE_PLOTS:
        plot_metrics(all_results)

    print("\n=== EVAL COMPLETE ===")


# =====================================================================
#   PLOTS (simple + publication-style)
# =====================================================================

def plot_metrics(all_results):
    """
    Creates simple bar charts comparing:
        - tasks completed
        - total utility
    """
    def avg(policy, key):
        arr = [m[key] for m in all_results[policy]]
        return np.mean(arr)

    # Metrics to compare
    metrics = ["tasks_completed", "total_utility"]

    for metric in metrics:
        plt.figure(figsize=(6, 4))
        vals = [avg(p, metric) for p in POLICIES]

        plt.bar(POLICIES, vals, color=["steelblue", "darkorange", "seagreen"])
        plt.title(f"Average {metric} (N={NUM_EPISODES})")
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()


# =====================================================================
#   ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
