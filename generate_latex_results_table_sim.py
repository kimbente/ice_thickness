import os
import pandas as pd

# Define SIM results directory
RESULTS_DIR = "results_sim"

# Define the models and simulation names in the order we want
models = ["dfNN", "dfGP", "dfNGP", "PINN", "GP"]
simulations = ["curve", "branching", "deflection", "ridges", "edge"]
roman_numerals = ["I", "II", "III", "IV", "V"]

# Initialize the LaTeX lines list
latex_lines = []

# Iterate over simulates data experiments
for idx, sim_name in enumerate(simulations):
    roman = roman_numerals[idx]
    latex_lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{roman}. {sim_name.capitalize()}}}}} \\")
    latex_lines.append(r"\midrule")

    # STEP 1: find best (lowest) NLL (ignoring n.a.)
    # NOTE: We will format this as green
    best_nll_mean = None
    best_model = None

    for model in models:
        file_path = os.path.join(RESULTS_DIR, model, f"{sim_name}_{model}_metrics_summary.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        mean_row = df[df.iloc[:, 0] == "mean"].iloc[0]

        if "Test full NLL" in mean_row.index and model not in ["dfNN", "PINN"]:
            nll_mean = mean_row["Test full NLL"]
            if (best_nll_mean is None) or (nll_mean < best_nll_mean):
                best_nll_mean = nll_mean
                best_model = model

    # STEP 2: Loop over models and make rows
    for model in models:
        print(f"Processing {model} for simulation {sim_name}...")
        file_path = os.path.join(RESULTS_DIR, model, f"{sim_name}_{model}_metrics_summary.csv")

        df = pd.read_csv(file_path)
        mean_row = df[df.iloc[:, 0] == "mean"].iloc[0]
        std_row = df[df.iloc[:, 0] == "std"].iloc[0]

        # Truncate numeric strings to 4 characters, no scientific notation
        rmse_mean = "{:.4f}".format(mean_row["Test RMSE"])[:5]
        rmse_std = "{:.4f}".format(std_row["Test RMSE"])[:5]
        mae_mean = "{:.4f}".format(mean_row["Test MAE"])[:5]
        mae_std = "{:.4f}".format(std_row["Test MAE"])[:5]
        mad_mean = "{:.4f}".format(mean_row["Test MAD"])[:5]
        mad_std = "{:.4f}".format(std_row["Test MAD"])[:5]

        # NLL handling
        if "Test full NLL" in mean_row.index:
            nll_mean_val = mean_row["Test full NLL"]
            nll_std_val = std_row["Test full NLL"]
            nll_mean = "{:.4f}".format(nll_mean_val)[:5]
            nll_std = "{:.4f}".format(nll_std_val)[:5]
            nll_str_raw = f"{nll_mean} \\footnotesize{{± {nll_std}}}"

            # Highlight if this is the best NLL
            if model == best_model:
                nll_str = r"\textcolor{OliveGreen}{" + nll_str_raw + "}"
            else:
                nll_str = nll_str_raw
        else:
            nll_str = r"\footnotesize{n.a.}"

        # MAD cleanup for zeros
        if float(mad_mean) == 0.0:
            mad_mean = "0.0"
        if float(mad_std) == 0.0:
            mad_std = "0.0"

        if float(mad_mean) > 0:
            mad_str = r"\textcolor{BrickRed}{\footnotesize{" + f"{mad_mean} ± {mad_std}" + "}}"
        else:
            mad_str = f"{mad_mean} \\footnotesize{{± {mad_std}}}"

        # Compose LaTeX row
        row = (
            f"{model} & "
            f"{nll_str} & "
            f"{rmse_mean} \\footnotesize{{± {rmse_std}}} & "
            f"{mae_mean} \\footnotesize{{± {mae_std}}} & "
            f"{mad_str} \\\\"
        )
        latex_lines.append(row)

        # Insert midrule after dfNGP
        if model == "dfNGP":
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    if idx != len(simulations) - 1:
        latex_lines.append(r"\toprule")

# Save to file
with open("results_sim_latex_table.txt", "w") as f:
    for line in latex_lines:
        f.write(line + "\n")

print("LaTeX table generated: results_sim_latex_table.txt")