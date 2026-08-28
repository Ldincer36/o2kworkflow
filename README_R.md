# o2kworkflow — R / RStudio port

This is an R translation of the Python code in `src/` (`parser.py`,
`processfile.py`, `gatp.py`), packaged as an RStudio project so it can be
opened and run directly in RStudio alongside the existing Python code.

## Getting started

1. Open `o2kworkflow.Rproj` in RStudio (double-click it, or File → Open
   Project). This sets your working directory to the project root and
   makes relative paths like `data/` work the same way the Python
   scripts' `Path(__file__).resolve().parent.parent` did.
2. In the R console, run:
   ```r
   source("R/00_setup.R")
   ```
   This installs (once) and loads the packages the project needs:
   `dplyr`, `ggplot2`, `stringr`, `here`.
3. Open `run_analysis.R` for the three entry points (batch baseline
   processing, single-file exploration, and the ATP/CK thermodynamics
   calculator) — each is commented out so you can run only what you need.

## File map

| Python              | R                    | Purpose                                                                 |
|---------------------|----------------------|--------------------------------------------------------------------------|
| `src/processfile.py`| `R/process_file.R`   | Batch baseline-subtract raw DatLab CSVs under `data/`, save plots + adjusted CSVs |
| `src/parser.py`      | `R/parser.R`          | Explore a single "flux export" CSV: extract flux, summarize states, plot, compute ratios |
| `src/gatp.py`        | `R/gatp.R`            | ATP/creatine-kinase clamp thermodynamics (dG' of ATP hydrolysis)         |
| —                    | `R/figure4_conductance.R` | Respiratory conductance from raw CK-clamp titration exports (see below) |
| —                    | `R/00_setup.R`        | Installs/loads required packages                                        |
| —                    | `run_analysis.R`      | Top-level entry point, sources everything above                         |

`data/`, `notebooks/`, and `src/` are untouched — this just adds an R/
folder and the project file alongside them, so both languages can keep
working from the same `data/` directory.

## Verification

Before delivering this, I ran the R translation against your own data and
compared it to the outputs your Python scripts had already produced:

- **`process_file.R`** was run on all 11 raw CSVs in `data/original_test/`
  and `data/test_data/`, and every output value matched the existing
  `*_BASELINE.csv` files in your `results/` folders to full double-precision
  (the only differences were how many digits `write.csv` prints — R shows
  ~14-15 significant digits by default vs. pandas' ~16, which is a
  formatting difference, not a computation difference).
- **`gatp.R`**: `gatp(c(1, 2, 3, 6, 3, 15))` was compared against
  `gatp.py`'s output for the same input and matched to 10+ significant
  figures on all six returned values.
- **`parser.R`** could *not* be validated against real data — none of the
  CSVs currently in `data/` have the "flux export" shape it expects
  (columns like `chamber`, `plot name`, `event name`, `time [min]`,
  `value`; the files you have are the tab-separated `mark_statistics`
  exports that `processfile.py`/`process_file.R` handle instead). The
  translation is a faithful line-by-line port, but test it against a real
  export before trusting it for a figure.

## Respiratory conductance (`R/figure4_conductance.R`)

This recreates the "conductance" panel from Fig. 4A of Wimberly et al.
2025 (the mitochondrial diagnostics/aging paper you shared): for each
sample, JO2 is measured across a creatine-kinase (CK) clamp PCr
titration, converted to an energy demand (ΔG_ATP), and the slope of
JO2 vs. ΔG_ATP ("conductance") is fit per sample.

**Input format matters**: this works on the raw, continuous O2k
`*.ANALYZED.csv` exports (one row per ~2 sec, with event marks like
`PCR1`, `PCR2`, ... embedded in an `Event Name` column) — not the
`mark_statistics` summary files that `process_file.R` handles. Point
`process_titration_dir()` at a folder of these raw exports.

**How it works:**
1. `read_o2k_trace()` reads the file (stripping the UTF-8 BOM some
   DatLab exports put on the header, which otherwise corrupts the first
   column name).
2. `extract_titration_flux()` finds the `PCR<n>` event marks, splits the
   trace into one window per titration step, and takes the mean JO2
   (chambers A and B) over the last 45 seconds of each window (before
   the next addition) as that step's steady-state flux — configurable
   via the `tail_seconds` argument.
3. `get_pcr_sequence()` determines the cumulative PCr concentration (mM)
   at each step: **the default sequence is 1, 3, 6, 9, 12 mM** (from
   your lab's "Mito Protocols" spreadsheet, Pyruvate + Malate protocol),
   used for any file whose name doesn't start with `x`. A file whose
   name *does* start with `x` gets a custom sequence instead, read from
   a dash-separated list at the very end of the filename (before the
   extension) — e.g. `xMyRun-1-3-6-9-12.ANALYZED.csv` → `c(1,3,6,9,12)`.
4. ΔG_ATP for that sequence — default or custom — is computed with
   `gatp()` (not read from the spreadsheet's own ΔG_ATP column; see the
   caveat below).
5. `parse_chamber_protein_mg()` reads chamber protein content (mg) from
   `..._A_<mg>_B_<mg>...` in the filename, and JO2 is normalized to
   pmol·s⁻¹·mg⁻¹.
6. `compute_conductance()` fits `JO2 ~ ΔG_ATP` per sample × chamber and
   returns the slope. `plot_jo2_vs_demand()` and `plot_conductance()`
   draw the two panels.

**Caveat you should know about**: I checked `gatp()`'s ΔG_ATP against
your spreadsheet's own tabulated values for the default (1-3-6-9-12 mM)
protocol, and they only agree at the very first step:

| Cumulative PCr (mM) | Spreadsheet ΔG_ATP (kJ/mol) | gatp() (kJ/mol) |
|---|---|---|
| 1  | −54.16 | −54.16 |
| 3  | −57.86 | −57.09 |
| 6  | −59.63 | −58.24 |
| 9  | −61.24 | −59.24 |
| 12 | −62.47 | −59.96 |

This isn't a bug in the R port — it's a faithful, verified translation
of `gatp.py`, and gives the same answer regardless of whether a
concentration is reached in one jump or several increments. The
spreadsheet's numbers were evidently produced by a different, unverified
method. Per your instruction, this script trusts `gatp()` over the
spreadsheet for both the default and any custom sequence — worth
resolving which is actually correct before this goes into a real figure,
since it shifts the x-axis by a few kJ/mol at the higher-PCr steps.

**Validated against your 4 real Plantaris CK-clamp files**: the pipeline
ran cleanly end-to-end (5 titration steps × 2 chambers × 4 samples = 40
rows), and the resulting JO2-vs-ΔG_ATP curve has the expected shape and
direction (JO2 rising as ΔG_ATP becomes less negative), matching the
qualitative pattern in the paper's Fig. 4A. I could not, however,
validate the absolute numbers against any independent source (there's no
existing "correct" conductance value for these files to check against),
so treat the output as a working first pass to sanity-check, not a
verified-to-the-decimal result the way `process_file.R` and the
`gatp(c(1,2,3,6,3,15))` example were.

**Assumptions worth double-checking**:
- The steady-state window is the *last 45 seconds* before each PCr
  addition (or before the substrate switch for the final step). This is
  a reasonable-looking default given your traces, but you may want a
  different window length or an explicit "exclude the first N seconds
  after each addition" rule.
- All four sample files were treated as one group (n=4 biological
  replicates) since nothing in the filenames or your instructions
  indicated separate conditions to compare — the current script doesn't
  split by any group; you'd add a grouping column yourself if you have
  conditions to compare (e.g. genotype, age).
- Non-titration marks (`CYTOC`, `CK+PCR+ATP`, `LOn`/`LOff`) are ignored
  for windowing purposes; `SUCC`/`ROT`/`ANTIA` close out the final
  window if present, otherwise the last window runs to the end of the
  file.

## Translation notes (things that differ subtly from the Python)

- **0- vs. 1-indexing.** Python's column/row positions (e.g. `df.iloc[:, 7]`,
  `ROW_IDX_POSITION = 7`) are all 0-indexed; the R code shifts each by one
  and comments the original Python index inline in `R/process_file.R`.
- **Population vs. sample standard deviation.** `numpy`'s `.std()` defaults
  to population std (divide by N), while R's built-in `sd()` divides by
  N-1. `R/parser.R` includes a `pop_sd()` helper used in
  `summarize_replicates()` so the numbers match Python's exactly — switch
  to `sd()` there if you'd rather have the sample estimate.
- **pandas/matplotlib → dplyr/ggplot2.** All data wrangling was rewritten
  with `dplyr` verbs and all plots with `ggplot2` rather than translated
  as literal loops, since that's the idiomatic way to write this in R —
  the logic (what's filtered, computed, and plotted) is unchanged.
- **`scipy.constants.R`** (the gas constant) doesn't have a base-R
  equivalent, so `R/gatp.R` hardcodes it as `R_GAS <- 8.31446261815324`
  (J/(mol·K)), the same value scipy uses.
- **Float-keyed dictionary lookup.** `gatp.py` looks up a free-Mg value
  from a dict keyed by an exact PCr concentration (float equality, which
  is fragile even in Python) and falls back to `np.interp`. `gatp.R` uses
  a small numeric tolerance for the "exact match" case and `approx(...,
  rule = 2)` for interpolation, which is more robust and behaves
  identically for the inputs tested.

## Packages used

`dplyr`, `ggplot2`, `stringr`, `here` (all installed by `R/00_setup.R`).
`tools` (for `file_path_sans_ext`) is part of base R.
