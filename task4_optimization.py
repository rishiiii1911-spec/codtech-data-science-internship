# =============================================================================
# CODTECH INTERNSHIP - TASK 4: OPTIMIZATION MODEL
# Tools   : PuLP, pandas, matplotlib
# Problem : Production Planning — maximize profit subject to resource constraints
#
# BUSINESS SCENARIO:
#   A factory produces two products: Tables and Chairs.
#   Goal: Find how many of each to produce to MAXIMIZE total profit,
#         given limited Wood, Labor, and Machine hours.
#
# pip install pulp pandas matplotlib
# =============================================================================

import pulp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

print("=" * 60)
print("  CODTECH Task 4 — Business Optimization with PuLP")
print("=" * 60)


# ──────────────────────────────────────────────
# STEP 1: DEFINE THE PROBLEM
# ──────────────────────────────────────────────
print("\n📋  PROBLEM SETUP")
print("-" * 40)

# Product data
products = ["Table", "Chair"]

# Profit per unit (₹)
profit = {"Table": 500, "Chair": 300}

# Resource required per unit
resources_per_unit = {
    #                Table  Chair
    "Wood (kg)":   {"Table": 10, "Chair": 5},
    "Labor (hrs)": {"Table": 4,  "Chair": 3},
    "Machine (hrs)":{"Table": 3, "Chair": 2},
}

# Available resources
available = {
    "Wood (kg)":    200,
    "Labor (hrs)":  120,
    "Machine (hrs)": 90,
}

# Print the problem summary as a table
data = {
    "Profit (₹/unit)": [profit[p] for p in products],
    "Wood (kg)":        [resources_per_unit["Wood (kg)"][p] for p in products],
    "Labor (hrs)":      [resources_per_unit["Labor (hrs)"][p] for p in products],
    "Machine (hrs)":    [resources_per_unit["Machine (hrs)"][p] for p in products],
}
df_problem = pd.DataFrame(data, index=products)
print("\n  Resource requirements per unit:")
print(df_problem.to_string())
print(f"\n  Available resources: {available}")


# ──────────────────────────────────────────────
# STEP 2: BUILD THE LINEAR PROGRAMMING MODEL
# ──────────────────────────────────────────────
print("\n\n🔧  BUILDING LP MODEL ...")

# Create the LP problem (maximization)
prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)

# Decision variables: how many Tables and Chairs to produce (must be >= 0)
x_table = pulp.LpVariable("Tables", lowBound=0, cat="Integer")
x_chair = pulp.LpVariable("Chairs", lowBound=0, cat="Integer")

# Objective function: maximize total profit
prob += profit["Table"] * x_table + profit["Chair"] * x_chair, "Total_Profit"

# Constraints: resource usage must not exceed availability
for resource, req in resources_per_unit.items():
    prob += (
        req["Table"] * x_table + req["Chair"] * x_chair <= available[resource],
        f"{resource}_constraint"
    )

print("  Objective  : Maximize  500·Tables + 300·Chairs")
print("  Subject to :")
for resource, req in resources_per_unit.items():
    print(f"    {req['Table']}·Tables + {req['Chair']}·Chairs ≤ {available[resource]}  ({resource})")
print("    Tables, Chairs ≥ 0  (integer)")


# ──────────────────────────────────────────────
# STEP 3: SOLVE THE MODEL
# ──────────────────────────────────────────────
print("\n\n⚙️   SOLVING ...")
status = prob.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output
print(f"  Solver status : {pulp.LpStatus[prob.status]}")


# ──────────────────────────────────────────────
# STEP 4: DISPLAY RESULTS
# ──────────────────────────────────────────────
print("\n\n📊  OPTIMAL SOLUTION")
print("-" * 40)

tables_opt = int(pulp.value(x_table))
chairs_opt = int(pulp.value(x_chair))
profit_opt = int(pulp.value(prob.objective))

print(f"  Tables to produce : {tables_opt}")
print(f"  Chairs to produce : {chairs_opt}")
print(f"  Maximum Profit    : ₹{profit_opt:,}")

# Resource utilisation
print("\n  Resource utilisation:")
util_rows = []
for resource, req in resources_per_unit.items():
    used     = req["Table"] * tables_opt + req["Chair"] * chairs_opt
    capacity = available[resource]
    pct      = used / capacity * 100
    util_rows.append({"Resource": resource, "Used": used, "Available": capacity, "Utilisation": f"{pct:.1f}%"})
    print(f"    {resource:<16}: {used:>4} / {capacity}  ({pct:.1f}%)")

df_util = pd.DataFrame(util_rows)


# ──────────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ──────────────────────────────────────────────
print("\n\n📈  Generating visualizations ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("CODTECH Task 4 — Production Optimization Results", fontsize=13, fontweight="bold")

# --- Chart 1: Optimal production mix (bar chart) ---
axes[0].bar(["Tables", "Chairs"], [tables_opt, chairs_opt],
            color=["#2980b9", "#e67e22"], edgecolor="black", width=0.5)
axes[0].set_title("Optimal Production Mix")
axes[0].set_ylabel("Units to Produce")
axes[0].set_ylim(0, max(tables_opt, chairs_opt) * 1.3)
for i, v in enumerate([tables_opt, chairs_opt]):
    axes[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold", fontsize=13)
axes[0].grid(axis="y", alpha=0.3)

# --- Chart 2: Resource utilisation (horizontal bar) ---
resources_list = [r["Resource"] for r in util_rows]
used_list      = [r["Used"]     for r in util_rows]
avail_list     = [r["Available"]for r in util_rows]

y_pos = range(len(resources_list))
axes[1].barh(y_pos, avail_list, color="#bdc3c7", label="Available", edgecolor="black")
axes[1].barh(y_pos, used_list,  color="#e74c3c", label="Used",      edgecolor="black")
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(resources_list)
axes[1].set_title("Resource Utilisation")
axes[1].set_xlabel("Units")
axes[1].legend()
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("optimization_results.png", dpi=150, bbox_inches="tight")
print("  💾  Saved → optimization_results.png")


# ──────────────────────────────────────────────
# STEP 6: SENSITIVITY / INSIGHTS
# ──────────────────────────────────────────────
print("\n\n💡  BUSINESS INSIGHTS")
print("-" * 40)
print(f"  1. Produce {tables_opt} Tables and {chairs_opt} Chairs for maximum profit of ₹{profit_opt:,}.")
print(f"  2. Revenue breakdown:")
print(f"       Tables : {tables_opt} × ₹500 = ₹{tables_opt * 500:,}")
print(f"       Chairs : {chairs_opt} × ₹300 = ₹{chairs_opt * 300:,}")
bottleneck = df_util[df_util["Utilisation"] == "100.0%"]["Resource"].tolist()
if bottleneck:
    print(f"  3. Bottleneck resource(s): {', '.join(bottleneck)}")
    print(f"     → Investing in more {bottleneck[0]} capacity could increase profit further.")
else:
    print("  3. No resource is fully exhausted — there may be room for more production.")

print("\n" + "=" * 60)
print("  Task 4 complete! 🎉")
print("  Files generated:")
print("    • optimization_results.png  ← production mix & utilisation charts")
print("=" * 60)
