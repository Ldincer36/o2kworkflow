#!/usr/bin/env python3
"""
process_files.py
Process every .csv under project/data/,
compute baseline (anti-mysin column or last column),
baseline-subtract row 7,
save plot to plots/,
and save modified CSV to results/ with _BASELINE appended.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


ROW_IDX_POSITION = 7



def process_file(csv_path: Path):
    print(f"\nProcessing: {csv_path.name}")
    sample_dir = csv_path.parent
    plots_dir = sample_dir / "plots"
    results_dir = sample_dir/ "results"
    plots_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    print(f"Plots → {plots_dir}")
    print(f"Results → {results_dir}")
    try:
        df = pd.read_csv(csv_path, sep="\t", skiprows=1)
        col7 = df.iloc[:, 7]
        col7_name = df.columns[7]

        col8 = df.iloc[:, 8]
        col8_name = df.columns[8]
        df = df.iloc[:, 9:-1]
        df.columns = df.columns.str.strip()
        df = df.iloc[:-1]

        # Find anti-myosin column
        anti_candidates = df.columns[df.columns.str.upper().str.contains("ANTI|MYSIN")]
        if len(anti_candidates) > 0:
            baseline_col = anti_candidates[0]
            print(f"Using anti-myosin column: {baseline_col}")
        else:
            baseline_col = df.columns[0]
            print("No anti-myosin column found; using last column as baseline.")

        if ROW_IDX_POSITION >= len(df):
            print("Row index out of range — skipping.")
            return

        baseline_value = pd.to_numeric(
            df.iloc[ROW_IDX_POSITION][baseline_col],
            errors="coerce"
        )

        if pd.isna(baseline_value):
            print("Baseline value invalid — skipping.")
            return

        row_numeric = df.iloc[ROW_IDX_POSITION].apply(pd.to_numeric, errors="coerce")
        new_row = row_numeric - baseline_value
        new_row.name = "baseline_subtracted"

        df.loc[new_row.name] = new_row
        df.insert(0, col8_name, col8.iloc[:len(df)].values)
        df.insert(0, col7_name, col7.iloc[:len(df)].values)
        df.iloc[-1, 0] = "Baseline Adjusted O2 slope neg"

       
        
        # -----------------
        # Save new CSV file
        # -----------------
        
        output_csv = results_dir / f"{csv_path.stem}_BASELINE.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV: {output_csv}")

        # -----------------
        # Plot
        # -----------------
        plt.style.use("seaborn-v0_8-whitegrid")

        plt.figure(figsize=(10, 6))

        x_labels = new_row.index.astype(str)
        x_positions = np.arange(len(x_labels))

        line_color = "#1f77b4"
        baseline_color = "#d62728"

        # Main line
        plt.plot(
            x_positions,
            new_row.values,
            marker='o',
            markersize=6,
            linewidth=2.5,
            label="O2K Flux (baseline subtracted)",
            color=line_color
        )

        # Baseline
        plt.axhline(
            y=0,
            color=baseline_color,
            linestyle="--",
            linewidth=1.8,
            label="Baseline"
)

        # Labels & title
        plt.xticks(x_positions, x_labels, rotation=45, ha='right')
        plt.ylabel("O2K Flux", weight="bold")
        plt.title(f"O2K Flux – {csv_path.stem}", weight="bold")

        # Remove top/right spines (ggplot-style)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Light grid only on y-axis
        plt.grid(True, axis='y', linestyle="--", alpha=0.3)
        plt.grid(False, axis='x')

        # Legend
        plt.legend(frameon=False)

        plt.tight_layout()

        output_plot = plots_dir / f"{csv_path.stem}_plot.png"
        plt.savefig(output_plot, dpi=300)
        plt.close()

        print(f"Saved plot: {output_plot}")

    except Exception as e:
        print(f"Failed {csv_path.name}: {e}")



def main():
    if not DATA_DIR.exists():
        print("Data directory not found.")
        return

    csv_files = [
    f for f in DATA_DIR.rglob("*.csv")
    if "results" not in f.parts
    and "plots" not in f.parts
    and "_BASELINE" not in f.name
]

    if not csv_files:
        print("No CSV files found in data/")
        return

    for file in csv_files:
        process_file(file)

    print("\nAll files processed successfully.")



if __name__ == "__main__":
    main()

