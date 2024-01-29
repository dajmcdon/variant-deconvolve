library(tidyverse)
library(Matrix)
params <- read_rds("data/ca-convolution-mat.rds")
y <- read_rds("data/deconvolved_ca_ga.rds") |>
  filter(time_value <= ymd("2021-12-01"), geo_value == "ca") |>
  pull(cases)
x <- seq_along(y)
cmats <- params$Cmat
cmat <- reduce(cmats, `+`)

Rcpp::sourceCpp("src/estim_path.cpp")
t1 <- estim_path_single(y, x, cmat, 3, double(50), nsol = 50L, lambdamax = 10000)
