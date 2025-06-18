import os
import pandas as pd

# Define SIM results directory
RESULTS_DIR = "results_sim"

# Define the models and simulation names in the order we want
models = ["dfNN", "dfGP", "dfGPcm", "dfNGP", "PINN", "GP"] # "dfGP2"
simulations = ["curve", "branching", "deflection", "ridges", "edge"]
roman_numerals = ["I", "II", "III", "IV", "V"]

# Initialize the LaTeX lines list
latex_lines = []

# Iterate over simulations
for idx, sim_name in enumerate(simulations):
    roman = roman_numerals[idx]
    latex_lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{roman}. {sim_name.capitalize()}}}}} \\")
    latex_lines.append(r"\midrule")

    # Find the best (lowest) NLL and RMSE (ignoring n.a.)
    best_nlpd_mean = None
    best_model_nlpd = None
    best_rmse_mean = None
    best_model_rmse = None

    for model in models:
        file_path = os.path.join(RESULTS_DIR, model, f"{sim_name}_{model}_metrics_summary.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        mean_row = df[df.iloc[:, 0] == "mean"].iloc[0]

        # Best NLL
        if "Test NLPD" in mean_row.index and model not in ["dfNN", "PINN"]:
            nlpd_mean = mean_row["Test NLPD"]
            if (best_nlpd_mean is None) or (nlpd_mean < best_nlpd_mean):
                best_nlpd_mean = nlpd_mean
                best_model_nlpd = model

        # Best RMSE
        rmse_mean = mean_row["Test RMSE"]
        if (best_rmse_mean is None) or (rmse_mean < best_rmse_mean):
            best_rmse_mean = rmse_mean
            best_model_rmse = model

    # Loop over models to generate LaTeX rows
    for model in models:
        print(f"Processing {model} for simulation {sim_name}...")
        file_path = os.path.join(RESULTS_DIR, model, f"{sim_name}_{model}_metrics_summary.csv")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue

        df = pd.read_csv(file_path)
        mean_row = df[df.iloc[:, 0] == "mean"].iloc[0]
        std_row = df[df.iloc[:, 0] == "std"].iloc[0]

        # Truncate numeric strings to 5 characters, no scientific notation
        rmse_mean_val = mean_row["Test RMSE"]
        rmse_std_val = std_row["Test RMSE"]
        rmse_mean = "{:.4f}".format(rmse_mean_val)[:5]
        rmse_std = "{:.4f}".format(rmse_std_val)[:5]

        mae_mean = "{:.4f}".format(mean_row["Test MAE"])[:5]
        mae_std = "{:.4f}".format(std_row["Test MAE"])[:5]
        mad_mean = "{:.4f}".format(mean_row["Test MAD"])[:5]
        mad_std = "{:.4f}".format(std_row["Test MAD"])[:5]

        # NLL handling
        if "Test NLPD" in mean_row.index:
            nlpd_mean_val = mean_row["Test NLPD"]
            nlpd_std_val = std_row["Test NLPD"]
            nlpd_mean = "{:.4f}".format(nlpd_mean_val)[:5]
            nlpd_std = "{:.4f}".format(nlpd_std_val)[:5]
            nlpd_str_raw = f"{nlpd_mean} \\footnotesize{{± {nlpd_std}}}"

            # Highlight if this is the best NLL
            if model == best_model_nlpd:
                nlpd_str = r"\textcolor{OliveGreen}{" + nlpd_str_raw + "}"
            else:
                nlpd_str = nlpd_str_raw
        else:
            nlpd_str = r"\footnotesize{n.a.}"

        # Full NLL handling
        if "Test QCE" in mean_row.index:
            qce_mean_val = mean_row["Test QCE"]
            qce_std_val = std_row["Test QCE"]
            qce_mean = "{:.4f}".format(qce_mean_val)[:5]
            qce_std = "{:.4f}".format(qce_std_val)[:5]
            qce_str = f"{qce_mean} \\footnotesize{{± {qce_std}}}"
        else:
            qce_str = r"\footnotesize{n.a.}"

        # Highlight if this is the best RMSE
        rmse_str_raw = f"{rmse_mean} \\footnotesize{{± {rmse_std}}}"
        if model == best_model_rmse:
            rmse_str = r"\textcolor{OliveGreen}{" + rmse_str_raw + "}"
        else:
            rmse_str = rmse_str_raw

        # MAD cleanup for zeros
        if float(mad_mean) == 0.0:
            mad_mean = "0.0"
        if float(mad_std) == 0.0:
            mad_std = "0.0"

        if float(mad_mean) > 0:
            mad_str = r"\textcolor{BrickRed}{\footnotesize{" + f"{mad_mean} ± {mad_std}" + "}}"
        else:
            mad_str = f"{mad_mean} \\footnotesize{{± {mad_std}}}"

        # Compose LaTeX row with new column order: RMSE, MAE, NLL, MAD
        row = (
            f"{model} & "
            f"{rmse_str} & "
            f"{mae_mean} \\footnotesize{{± {mae_std}}} & "
            f"{nlpd_str} & "
            f"{qce_str} & "  # new column
            f"{mad_str} \\\\"
        )
        latex_lines.append(row)

        # Insert midrule after dfNGP (last divergence-free model)
        if model == "dfNGP":
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    if idx != len(simulations) - 1:
        latex_lines.append(r"\toprule")

# Save to file
with open("generated_latex_results_table_sim_sparse.txt", "w") as f:
    for line in latex_lines:
        f.write(line + "\n")

print("LaTeX table generated: generated_latex_results_table_sim_sparse.txt")
