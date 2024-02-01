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
tt <- admm_testing(10000, 3, y, x, cmat, 1000, 1, 1e-3)
t1 <- estim_path_single(y, x, cmat, 3, double(50), nsol = 50L, lambdamax = 10000)

plot(y, ty = "l")
for (i in 1:30) {
  tt <- admm_testing(i, 3, y, x, cmat, 10000, 1, 1e-3)
  lines(tt$theta, col = i + 1)
}
