library(tidyverse)
library(Matrix)
params <- read_rds("data/ca-convolution-mat.rds")
y <- read_rds("data/deconvolved_ca_ga.rds") |>
  filter(time_value <= ymd("2023-03-01"), geo_value == "ca") |>
  pull(cases)
x <- seq_along(y)
cmats <- params$Cmat
cmat <- reduce(cmats, `+`)

Rcpp::sourceCpp("src/estim_path.cpp")
tt <- admm_test(5000, 3, y, x, cmat, 1000, 1, 1e-3)
t1 <- estim_path_single(y, x, cmat, 3, double(10), nsol = 10L, lambdamax = 10000)


# Backfitting -------------------------------------------------------------

library(rlang)
cc <- inject(cbind(!!!cmats)) # hard to pass list of mats to cpp
t2 <- backfitting_test(3, length(cmats), y, x, cc, 10000, 1, 1e-3, 25, 200)
