import os
import numpy as np
from scipy import stats


# =========================
# PATHS
# =========================
Model1_PATH = r""
Model2_PATH = r""


# =========================
# FUNCTION TO LOAD AUCs
# =========================
def load_auc_values(folder):

    auc_01_2 = []
    auc_2_34 = []
    auc_01_34 = []

    for file in os.listdir(folder):

        if not file.endswith("_AUC.txt"):
            continue

        filepath = os.path.join(folder, file)

        with open(filepath, "r") as f:
            lines = f.readlines()

        for line in lines:

            if "KL01_vs_KL2" in line:
                auc_01_2.append(float(line.split(":")[1]))

            elif "KL2_vs_KL34" in line:
                auc_2_34.append(float(line.split(":")[1]))

            elif "KL01_vs_KL34" in line:
                auc_01_34.append(float(line.split(":")[1]))

    return (
        np.array(auc_01_2),
        np.array(auc_2_34),
        np.array(auc_01_34)
    )


# =========================
# LOAD BOTH MODELS
# =========================
mlc_01_2, mlc_2_34, mlc_01_34 = load_auc_values(Model1_PATH)
fem_01_2, fem_2_34, fem_01_34 = load_auc_values(Model2_PATH)


# =========================
# CORRECTED t-TEST
# =========================
def corrected_t_test(x, y, k=5):

    diff = x - y
    n = len(diff)

    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)

    correction = (1 / n) + (1 / (k - 1))

    t_stat = mean_diff / np.sqrt(correction * var_diff)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

    return mean_diff, t_stat, p_value


# =========================
# STORE DATA CLEANLY
# =========================
data = {
    "KL01 vs KL2": (mlc_01_2, fem_01_2),
    "KL2 vs KL34": (mlc_2_34, fem_2_34),
    "KL01 vs KL34": (mlc_01_34, fem_01_34)
}


# =========================
# RUN TESTS
# =========================
alpha = 0.01

print("\n✅ AUC-based significance test (MLC vs FEM)\n")

for name, (mlc_vals, fem_vals) in data.items():

    mean_diff, t_stat, p_value = corrected_t_test(mlc_vals, fem_vals)

    print(f"{name}:")
    print(f"  Number of folds   = {len(mlc_vals)}")
    print(f"  Mean AUC Model 1      = {np.mean(mlc_vals)*100:.2f}")
    print(f"  Mean AUC Model 2      = {np.mean(fem_vals)*100:.2f}")
    print(f"  Mean difference   = {mean_diff*100:.2f}")
    print(f"  t-statistic       = {t_stat:.4f}")
    print(f"  p-value           = {p_value:.6f}")

    if p_value < alpha:
        print("  ✅ Significant difference")
    else:
        print("  ❌ Not significant")

    if mean_diff > 0:
        print("  👉 Model 1 performs better\n")
    else:
        print("  👉 Model 2 performs better\n")
