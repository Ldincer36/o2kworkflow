# R/process_file.R
#


library(dplyr)
library(stringr)
library(ggplot2)

ROW_IDX_POSITION <- 8

process_file <- function(csv_path) {
  cat(sprintf("\nProcessing: %s\n", basename(csv_path)))

  sample_dir <- dirname(csv_path)
  plots_dir <- file.path(sample_dir, "plots")
  results_dir <- file.path(sample_dir, "results")
  dir.create(plots_dir, showWarnings = FALSE)
  dir.create(results_dir, showWarnings = FALSE)
  cat(sprintf("Plots -> %s\n", plots_dir))
  cat(sprintf("Results -> %s\n", results_dir))

  tryCatch({
    raw <- read.delim(csv_path, sep = "\t", skip = 1,
                       check.names = FALSE, stringsAsFactors = FALSE)

    # Python: col7 = df.iloc[:, 7] (0-indexed) -> R column 8.
    col7 <- raw[[8]]
    col7_name <- names(raw)[8]
    # Python: col8 = df.iloc[:, 8] (0-indexed) -> R column 9.
    col8 <- raw[[9]]
    col8_name <- names(raw)[9]

    # Python: df = df.iloc[:, 9:-1] -> R columns 10 through ncol-1.
    df <- raw[, 10:(ncol(raw) - 1), drop = FALSE]
    names(df) <- str_trim(names(df))
    df <- df[-nrow(df), , drop = FALSE]

    # Find anti-myosin column (case-insensitive).
    anti_idx <- which(str_detect(toupper(names(df)), "ANTI|MYSIN"))
    if (length(anti_idx) > 0) {
      baseline_col <- names(df)[anti_idx[1]]
      cat(sprintf("Using anti-myosin column: %s\n", baseline_col))
    } else {
      baseline_col <- names(df)[1]
      cat("No anti-myosin column found; using last column as baseline.\n")
    }

    if (ROW_IDX_POSITION > nrow(df)) {
      cat("Row index out of range — skipping.\n")
      return(invisible(FALSE))
    }

    baseline_value <- suppressWarnings(as.numeric(df[[baseline_col]][ROW_IDX_POSITION]))
    if (is.na(baseline_value)) {
      cat("Baseline value invalid — skipping.\n")
      return(invisible(FALSE))
    }

    row_numeric <- suppressWarnings(as.numeric(unlist(df[ROW_IDX_POSITION, ])))
    names(row_numeric) <- names(df)
    new_row <- row_numeric - baseline_value

    df[nrow(df) + 1, ] <- as.list(new_row)
    n <- nrow(df)

    df <- cbind(setNames(data.frame(col8[seq_len(n)], stringsAsFactors = FALSE), col8_name), df)
    df <- cbind(setNames(data.frame(col7[seq_len(n)], stringsAsFactors = FALSE), col7_name), df)
    df[n, 1] <- "Baseline Adjusted O2 slope neg"

    output_csv <- file.path(results_dir,
                             paste0(tools::file_path_sans_ext(basename(csv_path)), "_BASELINE.csv"))
    write.csv(df, output_csv, row.names = FALSE)
    cat(sprintf("Saved CSV: %s\n", output_csv))

    # ---- Plot ----
    plot_df <- data.frame(x = factor(names(new_row), levels = names(new_row)),
                           y = as.numeric(new_row))

    p <- ggplot(plot_df, aes(x, y, group = 1)) +
      geom_line(color = "#1f77b4", linewidth = 1.1) +
      geom_point(color = "#1f77b4", size = 2.2) +
      geom_hline(yintercept = 0, color = "#d62728", linetype = "dashed", linewidth = 0.9) +
      labs(x = NULL, y = "O2K Flux",
           title = paste("O2K Flux –", tools::file_path_sans_ext(basename(csv_path)))) +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            panel.grid.major.x = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold"),
            axis.title.y = element_text(face = "bold"),
            legend.position = "none")

    output_plot <- file.path(plots_dir,
                              paste0(tools::file_path_sans_ext(basename(csv_path)), "_plot.png"))
    ggsave(output_plot, plot = p, width = 10, height = 6, dpi = 300)
    cat(sprintf("Saved plot: %s\n", output_plot))

    TRUE
  }, error = function(e) {
    cat(sprintf("Failed %s: %s\n", basename(csv_path), conditionMessage(e)))
    FALSE
  })
}

#' Process every raw CSV under <project_root>/data (skips results/, plots/,

run_all <- function(data_dir = here::here("data")) {
  if (!dir.exists(data_dir)) {
    cat("Data directory not found.\n")
    return(invisible(NULL))
  }

  csv_files <- list.files(data_dir, pattern = "\\.csv$", recursive = TRUE, full.names = TRUE)
  csv_files <- csv_files[!grepl("(^|[/\\\\])(results|plots|titrations)([/\\\\]|$)", dirname(csv_files))]
  csv_files <- csv_files[!grepl("_BASELINE", basename(csv_files))]
  csv_files <- csv_files[!grepl("\\.ANALYZED\\.csv$", basename(csv_files), ignore.case = TRUE)]

  if (length(csv_files) == 0) {
    cat("No CSV files found in data/\n")
    return(invisible(NULL))
  }

  for (f in csv_files) process_file(f)

  cat("\nAll files processed successfully.\n")
}
