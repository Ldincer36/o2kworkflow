# R/gatp.R


R_GAS <- 8.31446261815324  # J / (mol * K), matches scipy.constants.R

equilibrium_constant <- function(kref, dH, product_charges, reactant_charges, name) {
  list(kref = kref, dH = dH, prod = product_charges, react = reactant_charges, name = name)
}

# ---- constants (ionic strength = 0 reference values) ----
is0_Ka_atp      <- equilibrium_constant(2.512e-08, -6.30, c(1, 4), c(3), "Ka_atp")
is0_Kb_mg_atp   <- equilibrium_constant(1.514e6,   22.90, c(2), c(4, 2), "Kb_mg_atp")
is0_Kb_mg_hatp  <- equilibrium_constant(4.266e3,   16.90, c(1), c(2, 3), "Kb_mg_hatp")
is0_Ka_adp      <- equilibrium_constant(6.607e-08, -5.60, c(1, 3), c(2), "Ka_adp")
is0_Kb_mg_adp   <- equilibrium_constant(4.466e4,   19.0,  c(1), c(2, 3), "Kb_mg_adp")
is0_Kb_mg_hadp  <- equilibrium_constant(3.163e2,   12.50, NA_real_, c(2, 2), "Kb_mg_hadp")
is0_Ka_pho      <- equilibrium_constant(6.026e-8,  12.2,  c(2), NA_real_, "Ka_pho")
is0_Kb_mg_pho   <- equilibrium_constant(5.128e2,   8.19,  NA_real_, c(2, 2), "Kb_mg_pho")
is0_Ka_pcr      <- equilibrium_constant(8.854e-6,  2.66,  c(2), NA_real_, "Ka_pcr")
is0_Kb_mg_pcr   <- equilibrium_constant(2.320e2,   8.19,  NA_real_, c(2, 2), "Kb_mg_pcr")

Kref_CK  <- equilibrium_constant(2.58e8,  -17.55, c(4), c(3, 2, 1), "Kref_CK")
Kref_ATP <- equilibrium_constant(2.946e-1, -20.50, c(3, 2, 1), c(4), "Kref_ATP")

#' Debye-Huckel activity coefficient ratio (Gamma) for a reaction's charged
#' species at a given temperature and ionic strength.
solve_gamma <- function(products, reactants, temp, ionic_strength) {
  Am <- 3 * (-16.39023 + (261.3371 / temp) + 3.3689633 * log(temp) -
               1.437167 * (temp / 100) + 0.111995 * (temp / 100)^2)
  B <- 1.6

  if (is.na(reactants[1])) {
    reactant_gammas <- 1
  } else {
    reactant_gammas <- exp((-Am * sqrt(ionic_strength) * reactants^2) /
                              (1 + B * sqrt(ionic_strength)))
  }

  if (is.na(products[1])) {
    product_gammas <- 1
  } else {
    product_gammas <- exp((-Am * sqrt(ionic_strength) * products^2) /
                             (1 + B * sqrt(ionic_strength)))
  }

  prod(product_gammas) / prod(reactant_gammas)
}

#' Van't Hoff temperature adjustment of an equilibrium constant.
vant_hoff <- function(temp1, temp2, Kref1, dH) {
  exp((-1000 * dH / R_GAS) * (1 / temp2 - 1 / temp1) + log(Kref1))
}

CK_concentration <- function(atp, cr, adp, pcr) {
  list(atp = atp, cr = cr, adp = adp, pcr = pcr)
}

#' Solve the CK equilibrium quadratic for the very first titration step.
calc_step_one_conc <- function(Keq_CK, ATP, Cr, PCr) {
  a <- Keq_CK - 1
  b <- Keq_CK * PCr + ATP + Cr
  c <- -ATP * Cr

  x <- (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)

  CK_concentration(atp = ATP - x, cr = Cr - x, adp = x, pcr = PCr + x)
}

#' Ionic strength at each PCr concentration (Tris-PCr + Tris-ATP buffer).

compute_ionic_strengths <- function(pcr_series) {
  basal_IS <- 0.168
  atp <- 0.005
  tris_atp <- 2 * atp
  atp_is <- 0.5 * (atp * (-4)^2 + tris_atp * 1^2)

  sapply(pcr_series, function(pcr) {
    tris <- 2 * pcr
    pcr_is <- 0.5 * (pcr * (-2)^2 + tris * 1^2)
    basal_IS + atp_is + pcr_is
  })
}

#' Solve the CK equilibrium quadratic for a subsequent titration step,

calc_step_conc <- function(Keq_CK, ATP, Cr, ADP, PCr_pre, PCr_add) {
  a <- Keq_CK - 1
  b <- Keq_CK * (ADP - PCr_pre - PCr_add) - ATP - Cr
  c <- Keq_CK * ADP * PCr_pre + Keq_CK * ADP * PCr_add - ATP * Cr

  disc <- b^2 - 4 * a * c
  if (disc < 0) disc <- 0  # numerical safety

  x <- (-b - sqrt(disc)) / (2 * a)

  CK_concentration(atp = ATP + x, cr = Cr + x, adp = ADP - x,
                    pcr = PCr_pre + (PCr_add - x))
}

#' Concentrations at every titration step, given target (cumulative) PCr

calc_step_concentrations <- function(Keq_CK, pcr_series) {
  ATP <- 0.005
  Cr  <- 0.005

  step_concentrations <- vector("list", length(pcr_series))
  step_concentrations[[1]] <- calc_step_one_conc(Keq_CK[1], ATP, Cr, pcr_series[1])

  if (length(pcr_series) > 1) {
    for (i in 2:length(pcr_series)) {
      prev <- step_concentrations[[i - 1]]
      PCr_add <- pcr_series[i] - prev$pcr
      step_concentrations[[i]] <- calc_step_conc(
        Keq_CK[i], prev$atp, prev$cr, prev$adp, prev$pcr, PCr_add
      )
    }
  }
  step_concentrations
}

#' Apparent ATP-hydrolysis equilibrium constant at each titration step.
calc_ATP_Keq_steps <- function(ionicStrengths, tempK, ATP_kref_titration, pH, freeMg) {
  proton <- 10^(-pH)
  temp25 <- 273.15 + 25

  if (length(ionicStrengths) != length(freeMg)) {
    stop("Size mismatch: the length of ionicStrengths and freeMg are not equal")
  }

  atp_Ks <- list(is0_Ka_atp, is0_Kb_mg_atp, is0_Kb_mg_hatp, is0_Ka_adp,
                 is0_Kb_mg_adp, is0_Kb_mg_hadp, is0_Ka_pho, is0_Kb_mg_pho)

  Keq_ATP <- numeric(length(ionicStrengths))

  for (index in seq_along(ionicStrengths)) {
    ion_str <- ionicStrengths[index]
    mg <- freeMg[index]
    new_Kabs <- list()

    for (k1 in atp_Ks) {
      k2 <- vant_hoff(temp25, tempK, k1$kref, k1$dH)
      k3 <- k2 / solve_gamma(k1$prod, k1$react, tempK, ion_str)
      new_Kabs[[k1$name]] <- k3
    }

    react <- 1 + proton / new_Kabs[["Ka_atp"]] + new_Kabs[["Kb_mg_atp"]] * mg +
      new_Kabs[["Kb_mg_hatp"]] * proton * mg / new_Kabs[["Ka_atp"]]

    prod_adp <- 1 + proton / new_Kabs[["Ka_adp"]] + new_Kabs[["Kb_mg_adp"]] * mg +
      new_Kabs[["Kb_mg_hadp"]] * proton * mg / new_Kabs[["Ka_adp"]]

    prod_pho <- 1 + proton / new_Kabs[["Ka_pho"]] + new_Kabs[["Kb_mg_pho"]] * mg

    Keq_ATP[index] <- (ATP_kref_titration[index] * prod_adp * prod_pho) / (react * proton)
  }

  Keq_ATP
}

#' Apparent creatine-kinase equilibrium constant at each titration step.
calc_CK_Keq_steps <- function(ionicStrengths, tempK, CK_kref_titration, pH, freeMg) {
  proton <- 10^(-pH)
  temp25 <- 273.15 + 25

  if (length(ionicStrengths) != length(freeMg)) {
    stop("Size mismatch: the length of ionicStrengths and freeMg are not equal")
  }

  ck_Ks <- list(is0_Ka_atp, is0_Kb_mg_atp, is0_Kb_mg_hatp, is0_Ka_adp,
                is0_Kb_mg_adp, is0_Kb_mg_hadp, is0_Ka_pcr, is0_Kb_mg_pcr)

  Keq_CK <- numeric(length(ionicStrengths))

  for (index in seq_along(ionicStrengths)) {
    ion_str <- ionicStrengths[index]
    mg <- freeMg[index]
    new_Kabs <- list()

    for (k1 in ck_Ks) {
      k2 <- vant_hoff(temp25, tempK, k1$kref, k1$dH)
      k3 <- k2 / solve_gamma(k1$prod, k1$react, tempK, ion_str)
      new_Kabs[[k1$name]] <- k3
    }

    products <- 1 + proton / new_Kabs[["Ka_atp"]] + new_Kabs[["Kb_mg_atp"]] * mg +
      new_Kabs[["Kb_mg_hatp"]] * proton * mg / new_Kabs[["Ka_atp"]]

    react_adp <- 1 + proton / new_Kabs[["Ka_adp"]] + new_Kabs[["Kb_mg_adp"]] * mg +
      new_Kabs[["Kb_mg_hadp"]] * proton * mg / new_Kabs[["Ka_adp"]]

    react_pcr <- 1 + proton / new_Kabs[["Ka_pcr"]] + new_Kabs[["Kb_mg_pcr"]] * mg

    Keq_CK[index] <- CK_kref_titration[index] * proton * products / (react_adp * react_pcr)
  }

  Keq_CK
}

#' Compute the apparent Gibbs free energy of ATP hydrolysis (dG', J/mol)
#' at every step of a CK-clamp titration.
#'
#' @param pcr_additions_mM numeric vector of PCr *additions* at each step,
#'   in millimolar (not cumulative concentrations).
#' @return numeric vector of dG' (J/mol), one value per titration step.
gatp <- function(pcr_additions_mM) {
  ref_pcr <- c(1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30) / 1000

  temp37 <- 273.15 + 37
  temp25 <- 273.15 + 25

  freeMg_ref <- c(
    0.000407972, 0.000395923, 0.000377038, 0.000364626,
    0.0003487,   0.00033737,  0.000323291, 0.000309397,
    0.000299109, 0.000292253, 0.000281548
  )

  phosphate <- 0.010
  pH <- 7.2
  basal_IS <- 0.168

  pcr_series <- pcr_additions_mM / 1000

  # ---- free Mg lookup/interpolation across the titration ----
  current_pcr <- 0
  freeMg <- numeric(length(pcr_series))
  for (i in seq_along(pcr_series)) {
    current_pcr <- current_pcr + pcr_series[i]
    match_idx <- which(abs(ref_pcr - current_pcr) < 1e-12)
    if (length(match_idx) > 0) {
      freeMg[i] <- freeMg_ref[match_idx[1]]
    } else {
      freeMg[i] <- approx(ref_pcr, freeMg_ref, xout = current_pcr, rule = 2)$y
    }
  }

  # ---- ionic strength across the titration ----
  atp <- 0.005
  tris_atp <- 2 * atp
  atp_is <- 0.5 * (atp * (-4)^2 + tris_atp * 1^2)

  current_pcr <- pcr_series[1]
  tris <- 2 * current_pcr
  pcr_is <- 0.5 * (current_pcr * (-2)^2 + tris * 1^2)
  ionicStrengths <- basal_IS + atp_is + pcr_is

  if (length(pcr_series) > 1) {
    for (pcr_add in pcr_series[2:length(pcr_series)]) {
      current_pcr <- current_pcr + pcr_add
      tris <- 2 * current_pcr
      pcr_is <- 0.5 * (current_pcr * (-2)^2 + tris * 1^2)
      ionicStrengths <- c(ionicStrengths, basal_IS + atp_is + pcr_is)
    }
  }

  cat("IONIC STRENGTH\n")
  for (idx in seq_along(ionicStrengths)) {
    cat(sprintf("%-8d%.6g\n", idx, ionicStrengths[idx] * 1000))
  }

  # ---- CK Keq ----
  CK_k2 <- vant_hoff(temp25, temp37, Kref_CK$kref, Kref_CK$dH)
  CK_kref_titration <- sapply(ionicStrengths, function(ion_str) {
    CK_k2 / solve_gamma(Kref_CK$prod, Kref_CK$react, temp37, ion_str)
  })
  Keq_CK <- calc_CK_Keq_steps(ionicStrengths, temp37, CK_kref_titration, pH, freeMg)

  cat("Creatine Kinase Krefs\n")
  for (idx in seq_along(CK_kref_titration)) cat(sprintf("%-8d%.6g\n", idx, CK_kref_titration[idx]))
  cat("Keq CK\n")
  for (idx in seq_along(Keq_CK)) cat(sprintf("%-8d%.6g\n", idx, Keq_CK[idx]))

  # ---- ATP Keq ----
  ATP_k2 <- vant_hoff(temp25, temp37, Kref_ATP$kref, Kref_ATP$dH)
  ATP_kref_titration <- sapply(ionicStrengths, function(ion_str) {
    ATP_k2 / solve_gamma(Kref_ATP$prod, Kref_ATP$react, temp37, ion_str)
  })
  Keq_ATP <- calc_ATP_Keq_steps(ionicStrengths, temp37, ATP_kref_titration, pH, freeMg)

  cat("ATP hydrolysis Krefs\n")
  for (idx in seq_along(ATP_kref_titration)) cat(sprintf("%-8d%.6g\n", idx, ATP_kref_titration[idx]))

  cat(sprintf("%-8s%s\n", "Step", "ATP Hydrolysis K'"))
  cat(strrep("-", 25), "\n", sep = "")
  for (idx in seq_along(Keq_ATP)) cat(sprintf("%-8d%.6g\n", idx, Keq_ATP[idx]))

  # ---- standard-state dG (dG_std) ----
  t37 <- 273.15 + 37
  dG_std_ATP <- sapply(Keq_ATP, function(keq) -R_GAS * t37 * log(keq))

  cat(sprintf("%-8s%s\n", "Step", "dG (standard) of ATP Hydrolysis (kJ/mol)"))
  cat(strrep("-", 39), "\n", sep = "")
  for (idx in seq_along(dG_std_ATP)) cat(sprintf("%-8d%.6g\n", idx, dG_std_ATP[idx] / 1000))

  # ---- concentrations at each step (using additions directly, as gatp.py does) ----
  ATP0 <- 0.005
  Cr0  <- 0.005
  pcr_additions <- pcr_additions_mM / 1000

  step_concentrations <- vector("list", length(pcr_additions))
  step_concentrations[[1]] <- calc_step_one_conc(Keq_CK[1], ATP0, Cr0, pcr_additions[1])

  if (length(pcr_additions) > 1) {
    for (i in 2:length(pcr_additions)) {
      prev <- step_concentrations[[i - 1]]
      step_concentrations[[i]] <- calc_step_conc(
        Keq_CK[i], prev$atp, prev$cr, prev$adp, prev$pcr, pcr_additions[i]
      )
    }
  }

  cat(sprintf("%60s\n", "Concentrations over the CK Clamp (mM)"))
  cat(sprintf("%12s%12s%12s%12s%12s\n", "Step", "ATP", "ADP", "PCr", "Cr"))
  cat(strrep("-", 60), "\n", sep = "")
  for (idx in seq_along(step_concentrations)) {
    step <- step_concentrations[[idx]]
    cat(sprintf("%12d%12.6g%12.6g%12.6g%12.6g\n", idx,
                step$atp * 1000, step$adp * 1000, step$pcr * 1000, step$cr * 1000))
  }

  # ---- dG' at each step ----
  dG_ATP <- numeric(length(step_concentrations))
  for (i in seq_along(step_concentrations)) {
    step <- step_concentrations[[i]]
    dG_ATP[i] <- dG_std_ATP[i] + R_GAS * temp37 * log(step$adp * phosphate / step$atp)
  }

  dG_ATP
}

