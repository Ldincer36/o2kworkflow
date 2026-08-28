# R/parser.R


library(dplyr)
library(ggplot2)

#' Load a DatLab CSV export.
load_csv <- function(path) {
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

#' Extract flux values for a specific chamber and plot.
extract_flux <- function(df, chamber, plot_name = "O2 flux") {
  df %>% filter(chamber == !!chamber, `plot name` == !!plot_name)
}

#' Get key respiration states (LEAK, OXPHOS, ETS) for both chambers.
#' Returns a list with State / `Chamber 1` / `Chamber 2`, mirroring the
#' dict structure the Python version returned.
get_state_flux <- function(df) {
  states <- c(LEAK = "Add oligomycin", OXPHOS = "Add ADP", ETS = "Add FCCP")
  chambers <- c("Chamber 1", "Chamber 2")

  summary <- list(State = names(states))
  for (chamber in chambers) summary[[chamber]] <- numeric(length(states))

  for (i in seq_along(states)) {
    event <- states[[i]]
    for (chamber in chambers) {
      value <- df %>%
        filter(chamber == !!chamber,
               `plot name` == "O2 flux",
               `event name` == !!event) %>%
        pull(value)
      summary[[chamber]][i] <- value[1]
    }
  }
  summary
}

#' Compute OXPHOS/LEAK and ETS/OXPHOS ratios from a get_state_flux() summary.
compute_ratios <- function(summary) {
  chambers <- c("Chamber 1", "Chamber 2")
  efficiency <- list(`OXPHOS/LEAK` = numeric(length(chambers)),
                      `ETS/OXPHOS` = numeric(length(chambers)))

  for (i in seq_along(chambers)) {
    vals <- summary[[chambers[i]]]
    LEAK <- vals[1]; OXPHOS <- vals[2]; ETS <- vals[3]
    efficiency$`OXPHOS/LEAK`[i] <- OXPHOS / LEAK
    efficiency$`ETS/OXPHOS`[i]  <- ETS / OXPHOS
  }
  efficiency
}

#' Plot O2 flux over time for both chambers, with event labels — ggplot2
#' equivalent of parser.py's plot_flux().
plot_flux <- function(df) {
  ch1 <- extract_flux(df, "Chamber 1")
  ch2 <- extract_flux(df, "Chamber 2")

  events <- df %>%
    filter(`plot name` == "O2 flux") %>%
    distinct(`time [min]`, `event name`)

  y_max <- max(ch1$value, ch2$value, na.rm = TRUE)

  ggplot() +
    geom_vline(data = events, aes(xintercept = `time [min]`),
               color = "gray50", linetype = "dashed", alpha = 0.5) +
    geom_text(data = events,
               aes(x = `time [min]` + 0.1, y = y_max + 2, label = `event name`),
               angle = 45, hjust = 0, size = 3) +
    geom_line(data = ch1, aes(`time [min]`, value, color = "Chamber 1")) +
    geom_point(data = ch1, aes(`time [min]`, value, color = "Chamber 1"), shape = 16) +
    geom_line(data = ch2, aes(`time [min]`, value, color = "Chamber 2")) +
    geom_point(data = ch2, aes(`time [min]`, value, color = "Chamber 2"), shape = 15) +
    scale_color_manual(name = NULL, values = c("Chamber 1" = "blue", "Chamber 2" = "red")) +
    labs(x = "Time (min)", y = "O2 flux (pmol·s⁻¹·mg⁻¹)",
         title = "O2 Flux Over Time - Both Chambers") +
    theme_minimal()
}

#' Plot key respiration states (from get_state_flux()) as a grouped bar chart.
plot_bar <- function(summary) {
  states_labels <- summary$State
  long <- data.frame(
    State   = factor(rep(states_labels, 2), levels = states_labels),
    Chamber = rep(c("Chamber 1", "Chamber 2"), each = length(states_labels)),
    Value   = c(summary$`Chamber 1`, summary$`Chamber 2`)
  )

  ggplot(long, aes(State, Value, fill = Chamber)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_text(aes(label = sprintf("%.1f", Value)),
              position = position_dodge(width = 0.7), vjust = -0.4, size = 3) +
    scale_fill_manual(values = c("Chamber 1" = "blue", "Chamber 2" = "red")) +
    labs(x = "Respiration State", y = "O2 flux (pmol·s⁻¹·mg⁻¹)",
         title = "Key Respiration States by Chamber") +
    theme_minimal()
}

#' Plot respiration efficiency ratios (from compute_ratios()) as a bar chart.
plot_ratios <- function(efficiency) {
  ratios_labels <- names(efficiency)
  long <- data.frame(
    Ratio   = factor(rep(ratios_labels, 2), levels = ratios_labels),
    Chamber = rep(c("Chamber 1", "Chamber 2"), each = length(ratios_labels)),
    Value   = c(sapply(efficiency, `[`, 1), sapply(efficiency, `[`, 2))
  )

  ggplot(long, aes(Ratio, Value, fill = Chamber)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_text(aes(label = sprintf("%.2f", Value)),
              position = position_dodge(width = 0.7), vjust = -0.4, size = 3) +
    scale_fill_manual(values = c("Chamber 1" = "blue", "Chamber 2" = "red")) +
    labs(x = "Respiration Efficiency", y = "Ratio",
         title = "Mitochondrial Respiration Ratios") +
    theme_minimal()
}

#' Population standard deviation, matching numpy's default `.std()` (ddof=0).

pop_sd <- function(x) sqrt(mean((x - mean(x))^2))

#' Summarize respiration states across replicates: mean and (population)
summarize_replicates <- function(df) {
  states <- c(LEAK = "Add oligomycin", OXPHOS = "Add ADP", ETS = "Add FCCP")
  chambers <- unique(df$chamber)

  summary_stats <- list()
  for (ch in chambers) {
    means <- numeric(length(states))
    sds   <- numeric(length(states))
    for (i in seq_along(states)) {
      values <- df %>%
        filter(chamber == ch,
               `plot name` == "O2 flux",
               `event name` == states[[i]]) %>%
        pull(value)
      means[i] <- mean(values)
      sds[i]   <- pop_sd(values)
    }
    summary_stats[[ch]] <- list(State = names(states), Mean = means, Std = sds)
  }
  summary_stats
}

  plot_bar_with_error_save <- function(summary_stats, filename) {
  chambers <- names(summary_stats)
  states_labels <- summary_stats[[chambers[1]]]$State

  long <- do.call(rbind, lapply(chambers, function(ch) {
    data.frame(Chamber = ch,
               State = factor(summary_stats[[ch]]$State, levels = states_labels),
               Mean = summary_stats[[ch]]$Mean,
               Std = summary_stats[[ch]]$Std)
  }))

  p <- ggplot(long, aes(State, Mean, fill = Chamber)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_errorbar(aes(ymin = Mean - Std, ymax = Mean + Std),
                  position = position_dodge(width = 0.7), width = 0.2) +
    labs(x = "Respiration State", y = "O2 flux (pmol·s⁻¹·mg⁻¹)",
         title = "Respiration States Across Replicates") +
    theme_minimal()

  ggsave(filename, plot = p, width = 8, height = 5)
  invisible(p)
}

