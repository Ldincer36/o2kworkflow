# R/00_setup.R
#
# Installs (once) and loads the packages this project depends on.
# Source this file first after opening o2kworkflow.Rproj in RStudio.

required_packages <- c("dplyr", "ggplot2", "stringr", "here")

missing <- required_packages[!vapply(required_packages, requireNamespace,
                                      logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  install.packages(missing)
}

invisible(lapply(required_packages, library, character.only = TRUE))
