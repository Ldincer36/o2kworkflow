#open proj file, then uncomment code to run sections

source("R/00_setup.R")
source("R/parser.R")
source("R/process_file.R")
source("R/gatp.R")
source("R/figure4_conductance.R")

#baselin subtract
 run_all()


#thermo calculator
 dG_ATP <- gatp(c(1, 2, 3, 6, 3, 15))
 print(dG_ATP)

#figure4
 run_titration_analysis()
