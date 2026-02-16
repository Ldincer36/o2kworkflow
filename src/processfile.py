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
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

ROW_IDX_POSITION = 7

PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def process_file(csv_path: Path):
    print(f"\nProcessing: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path, sep="\t", skiprows=1)
        df = df.iloc[:, 9:-1]
        df.columns = df.columns.str.strip()
        df = df.iloc[:-1]

        # Find anti-myosin column
        anti_candidates = df.columns[df.columns.str.upper().str.contains("ANTI|MYSIN")]
        if len(anti_candidates) > 0:
            baseline_col = anti_candidates[0]
            print(f"Using anti-myosin column: {baseline_col}")
        else:
            baseline_col = df.columns[-1]
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

        # -----------------
        # Save new CSV file
        # -----------------
        output_csv = RESULTS_DIR / f"{csv_path.stem}_BASELINE.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV: {output_csv}")

        # -----------------
        # Plot
        # -----------------
        plt.figure(figsize=(10, 6))

        x_labels = new_row.index.astype(str)
        x_positions = np.arange(len(x_labels))

        plt.plot(
            x_positions,
            new_row.values,
            marker='o',
            linewidth=2,
            label="O2K Flux (baseline subtracted)",
            color="blue"
        )

        plt.axhline(
            y=0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Baseline (0 after subtraction)"
        )

        plt.xticks(x_positions, x_labels, rotation=45, ha='right')
        plt.ylabel("O2K Flux")
        plt.title(f"O2K Flux - {csv_path.stem}")

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_plot = PLOTS_DIR / f"{csv_path.stem}_plot.png"
        plt.savefig(output_plot, dpi=300)
        plt.close()

        print(f"Saved plot: {output_plot}")

    except Exception as e:
        print(f"Failed {csv_path.name}: {e}")


def main():
    if not DATA_DIR.exists():
        print("Data directory not found.")
        return

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in data/")
        return

    for file in csv_files:
        process_file(file)

    print("\nAll files processed successfully.")


if __name__ == "__main__":
    main()

