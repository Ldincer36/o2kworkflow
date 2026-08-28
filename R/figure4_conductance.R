# R/figure4_conductance.R
#
# Recreates the "respiratory conductance" panel of Wimberly et al. 2025

library(dplyr)
library(ggplot2)
library(stringr)

DEFAULT_PCR_MM     <- c(1, 3, 6, 9, 12)
DEFAULT_DG_ATP_KJ  <- gatp(diff(c(0, DEFAULT_PCR_MM))) / 1000

#' Read a raw O2k "ANALYZED.csv" export, stripping the UTF-8 BOM some

read_o2k_trace <- function(path) {
  raw <- readLines(path, encoding = "UTF-8", warn = FALSE)
  raw[1] <- sub("^\xef\xbb\xbf", "", raw[1], useBytes = TRUE)
  read.csv(text = paste(raw, collapse = "\n"), check.names = FALSE,
            stringsAsFactors = FALSE)
}

#' Pull chamber protein content (mg) from a "..._A_<mg>_B_<mg>..." filename.
parse_chamber_protein_mg <- function(filename) {
  base <- basename(filename)
  m <- regmatches(base, regexec("_A_([0-9]+(?:\\.[0-9]+)?)_B_([0-9]+(?:\\.[0-9]+)?)", base))[[1]]
  if (length(m) != 3) {
    stop(sprintf(
      "Could not find a '_A_<mg>_B_<mg>' pattern in filename: %s", filename
    ))
  }
  list(A = as.numeric(m[2]), B = as.numeric(m[3]))
}

#' Determine the cumulative PCr sequence (mM) and dG_ATP (kJ/mol) at each

get_pcr_sequence <- function(filename) {
  base <- basename(filename)
  base_noext <- sub("\\.[Aa][Nn][Aa][Ll][Yy][Zz][Ee][Dd]\\.[Cc][Ss][Vv]$", "", base)
  base_noext <- sub("\\.[Cc][Ss][Vv]$", "", base_noext)

  if (!grepl("^x", base_noext)) {
    return(list(pcr_mM = DEFAULT_PCR_MM, dG_ATP_kJ = DEFAULT_DG_ATP_KJ,
                source = "default sequence (1,3,6,9,12 mM), dG_ATP via gatp()"))
  }

  m <- regmatches(base_noext, regexpr("(-[0-9.]+)+$", base_noext))
  if (length(m) == 0 || identical(m, character(0)) || m == "") {
    warning(sprintf(
      "'%s' starts with 'x' but no dash-separated sequence was found at the end of the filename; falling back to the default protocol.",
      filename
    ))
    return(list(pcr_mM = DEFAULT_PCR_MM, dG_ATP_kJ = DEFAULT_DG_ATP_KJ,
                source = "default (fallback -- no sequence parsed)"))
  }

  pcr_mM <- as.numeric(strsplit(sub("^-", "", m), "-")[[1]])
  if (any(is.na(pcr_mM))) {
    stop(sprintf("Could not parse a numeric PCr sequence from '%s'.", filename))
  }
  additions <- diff(c(0, pcr_mM))
  dG_ATP_kJ <- gatp(additions) / 1000
  list(pcr_mM = pcr_mM, dG_ATP_kJ = dG_ATP_kJ,
       source = "custom sequence via gatp()")
}

#' Split a raw trace into titration-step windows using PCR-numbered event
extract_titration_flux <- function(df, tail_seconds = 45) {
  time_col  <- 1L  # Time [min]
  event_col <- 2L  # Event Name
  jo2_A_col <- 7L  # 1A: O2 slope neg. [pmol/s/mL]
  jo2_B_col <- 8L  # 1B: O2 slope neg. [pmol/s/mL]

  events <- df[!is.na(df[[event_col]]) & df[[event_col]] != "", c(time_col, event_col)]
  names(events) <- c("time_min", "event")

  pcr_events <- events[grepl("^PCR[0-9]+$", events$event, ignore.case = TRUE), ]
  pcr_events$step_num <- as.integer(gsub("[^0-9]", "", toupper(pcr_events$event)))
  pcr_events <- pcr_events[order(pcr_events$step_num), ]

  if (nrow(pcr_events) == 0) {
    stop("No 'PCR<n>' event marks found in this trace -- is this a CK-clamp titration file?")
  }

  end_marks <- c("SUCC", "ROT", "ANTIA")
  end_event <- events[toupper(events$event) %in% end_marks, ]

  n_steps <- nrow(pcr_events) + 1L
  window_start <- c(min(df[[time_col]], na.rm = TRUE), pcr_events$time_min)
  window_end   <- c(pcr_events$time_min,
                     if (nrow(end_event) > 0) min(end_event$time_min) else max(df[[time_col]], na.rm = TRUE))

  out <- vector("list", n_steps)
  for (i in seq_len(n_steps)) {
    hi <- window_end[i]
    lo <- hi - tail_seconds / 60
    seg <- df[df[[time_col]] > lo & df[[time_col]] <= hi, , drop = FALSE]
    if (nrow(seg) == 0) {
      seg <- df[df[[time_col]] >= window_start[i] & df[[time_col]] <= window_end[i], , drop = FALSE]
    }
    out[[i]] <- data.frame(
      step = i,
      window_start_min = window_start[i],
      window_end_min = window_end[i],
      n_points = nrow(seg),
      JO2_A = mean(seg[[jo2_A_col]], na.rm = TRUE),
      JO2_B = mean(seg[[jo2_B_col]], na.rm = TRUE)
    )
  }
  do.call(rbind, out)
}

#' Process one raw titration file end-to-end: read, window, normalize by

process_titration_file <- function(path, tail_seconds = 45) {
  df <- read_o2k_trace(path)
  seq_info <- get_pcr_sequence(path)
  flux <- extract_titration_flux(df, tail_seconds = tail_seconds)

  if (nrow(flux) != length(seq_info$pcr_mM)) {
    stop(sprintf(
      "%s: found %d titration window(s) in the trace but the %s has %d concentration(s) -- check the PCR mark count against the sequence length.",
      basename(path), nrow(flux), seq_info$source, length(seq_info$pcr_mM)
    ))
  }

  protein_mg <- parse_chamber_protein_mg(path)
  sample_id <- sub("\\.[Aa][Nn][Aa][Ll][Yy][Zz][Ee][Dd]\\.[Cc][Ss][Vv]$", "", basename(path))
  sample_id <- sub("\\.[Cc][Ss][Vv]$", "", sample_id)

  rbind(
    data.frame(sample = sample_id, chamber = "A", step = flux$step,
               PCr_mM = seq_info$pcr_mM, dG_ATP_kJ_mol = seq_info$dG_ATP_kJ,
               JO2_raw_pmol_s_mL = flux$JO2_A, protein_mg = protein_mg$A,
               n_points = flux$n_points, window_start_min = flux$window_start_min,
               window_end_min = flux$window_end_min, sequence_source = seq_info$source),
    data.frame(sample = sample_id, chamber = "B", step = flux$step,
               PCr_mM = seq_info$pcr_mM, dG_ATP_kJ_mol = seq_info$dG_ATP_kJ,
               JO2_raw_pmol_s_mL = flux$JO2_B, protein_mg = protein_mg$B,
               n_points = flux$n_points, window_start_min = flux$window_start_min,
               window_end_min = flux$window_end_min, sequence_source = seq_info$source)
  ) %>% mutate(JO2_pmol_s_mg = JO2_raw_pmol_s_mL / protein_mg)
}

#' Process every "*.ANALYZED.csv" titration file in a directory 
process_titration_dir <- function(dir, tail_seconds = 45) {
  files <- list.files(dir, pattern = "\\.[Aa][Nn][Aa][Ll][Yy][Zz][Ee][Dd]\\.[Cc][Ss][Vv]$",
                       full.names = TRUE)
  if (length(files) == 0) {
    stop(sprintf("No '*.ANALYZED.csv' files found in %s", dir))
  }
  bind_rows(lapply(files, process_titration_file, tail_seconds = tail_seconds))
}

#' Fit JO2 ~ dG_ATP per sample x chamber and return the slope ("conductance") and intercept for each.
compute_conductance <- function(long_data) {
  long_data %>%
    group_by(sample, chamber) %>%
    summarise(
      conductance = coef(lm(JO2_pmol_s_mg ~ dG_ATP_kJ_mol))[["dG_ATP_kJ_mol"]],
      intercept   = coef(lm(JO2_pmol_s_mg ~ dG_ATP_kJ_mol))[["(Intercept)"]],
      n_steps     = n(),
      .groups = "drop"
    )
}

#' Plot JO2 vs. dG_ATP (mean +/- SD across chamber replicates at each

plot_jo2_vs_demand <- function(long_data, title = "Pyruvate/Malate") {
  summary_df <- long_data %>%
    group_by(dG_ATP_kJ_mol) %>%
    summarise(mean_JO2 = mean(JO2_pmol_s_mg), sd_JO2 = sd(JO2_pmol_s_mg), .groups = "drop")

  ggplot(summary_df, aes(dG_ATP_kJ_mol, mean_JO2)) +
    geom_errorbar(aes(ymin = mean_JO2 - sd_JO2, ymax = mean_JO2 + sd_JO2), width = 0.3) +
    geom_point(size = 2.5) +
    geom_line() +
    labs(x = expression(Delta*G[ATP]~(kJ/mol)), y = expression(JO[2]~(pmol%.%s^-1%.%mg^-1)),
         title = title) +
    theme_minimal()
}

#' Plot per-sample conductance as a dot plot, styled like Fig. 4A's

plot_conductance <- function(conductance_df) {
  stats <- conductance_df %>%
    summarise(mean_c = mean(conductance), sd_c = sd(conductance))

  ggplot(conductance_df, aes(x = 1, y = conductance)) +
    geom_jitter(width = 0.08, size = 2.5, color = "#1f77b4") +
    geom_errorbar(data = stats, inherit.aes = FALSE,
                  aes(x = 1, ymin = mean_c - sd_c, ymax = mean_c + sd_c), width = 0.15) +
    geom_point(data = stats, inherit.aes = FALSE, aes(x = 1, y = mean_c),
               shape = 95, size = 12, color = "red") +
    scale_x_continuous(limits = c(0.5, 1.5), breaks = NULL) +
    labs(x = NULL, y = "Conductance (pmol O2/s/mg per kJ/mol)",
         title = "Respiratory Conductance") +
    theme_minimal()
}

#' End-to-end run: process every raw titration file in `dir`, save the
#' JO2-vs-demand and conductance plots to `<dir>/plots/`, and save the
#' underlying long-format flux data and per-sample conductance table to
#' `<dir>/results/`
#' process_file.R uses for the baseline-subtraction pipeline.
#' @param dir directory of raw "*.ANALYZED.csv" titration files 
#' @param title plot title / substrate label for the JO2-vs-demand plot.
#' @param tail_seconds steady-state window length (seconds) before each
#' @return invisibly, a list with the long flux data and conductance table.
run_titration_analysis <- function(dir = here::here("data", "titrations"),
                                    title = "Pyruvate/Malate",
                                    tail_seconds = 45) {
  plots_dir <- file.path(dir, "plots")
  results_dir <- file.path(dir, "results")
  dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

  long <- process_titration_dir(dir, tail_seconds = tail_seconds)
  cond <- compute_conductance(long)

  write.csv(long, file.path(results_dir, "titration_flux.csv"), row.names = FALSE)
  write.csv(cond, file.path(results_dir, "conductance.csv"), row.names = FALSE)
  cat(sprintf("Saved data: %s\n", file.path(results_dir, "titration_flux.csv")))
  cat(sprintf("Saved data: %s\n", file.path(results_dir, "conductance.csv")))

  p1 <- plot_jo2_vs_demand(long, title = title)
  p2 <- plot_conductance(cond)

  ggsave(file.path(plots_dir, "jo2_vs_demand.png"), p1, width = 6, height = 5, dpi = 300)
  ggsave(file.path(plots_dir, "conductance.png"), p2, width = 4, height = 5, dpi = 300)
  cat(sprintf("Saved plot: %s\n", file.path(plots_dir, "jo2_vs_demand.png")))
  cat(sprintf("Saved plot: %s\n", file.path(plots_dir, "conductance.png")))

  invisible(list(long = long, conductance = cond))
}
