library(tidyverse)
ca_report <- read_rds("data/report_delay_ca_ga.rds") |>
  filter(state == "CA")
vmix <- read_rds("data/variant_mix.rds") |>
  filter(State == "CA")
v <- vmix |> select(Alpha:Other) |> as.matrix()
d <- 1:nrow(v) / (nrow(v) - 1)


# Here, my data is daily, not only the observations
vmix_s <- fitted(nnet::multinom(v ~ poly(d, degree = 3)))
plotter <- function(predmat) {
  as_tibble(predmat) |> 
    mutate(time = 1:nrow(predmat)) |>
    pivot_longer(-time) |> 
    ggplot(aes(time, y=value, fill = name)) + 
    geom_area(position = "stack") + 
    theme_bw() + 
    scale_fill_viridis_d()
}
plotter(vmix_s)

report <- ca_report |>
  pivot_wider(names_from = delay, values_from = dist) |>
  select(-(Date:state)) |>
  as.matrix()

# period up to 2021-12-01
ed <- which(vmix$Date == "2021-12-01")
report <- report[1:ed, ]
vmix_s <- vmix_s[1:ed, ]

# variant-specific delays
discretize_gamma <- function(x, shape = 2.5, scale = 2.5, rate = 1 / scale) {
  stopifnot(shape > 0, scale > 0, all(x >= 0))
  pgm <- stats::pgamma(x, shape = shape, scale = scale)
  pgm <- c(0, pgm)
  pgm <- diff(pgm)
  list(pgm / sum(pgm))
}

mv_from_lognormal <- function(lmean, lsd) {
  c(exp(lmean + lsd^2 / 2), exp(lmean + lsd^2 / 2) * sqrt(exp(lsd^2) - 1))
}

max_inc_days <- 21
support <- 0:max_inc_days # 0 or 1??

b1 <- 5.1^2 / 2.7^2
b2 <- 2.7^2 / 5.1

dmsd <- mv_from_lognormal(1.25, 0.34)
inc_pars <- tribble(
  ~Variant, ~Shape, ~Scale,
  "Ancestral", 3.05, 1.95,
  "Alpha", 4.94^2 / 2.19^2, 2.19^2 / 4.94,
  "Omicron", 3.03^2 / 1.33^2, 1.33^2 / 3.03,
  "Delta", dmsd[1]^2 / dmsd[2]^2, dmsd[2]^2 / dmsd[1],
  "Beta", b1, b2,
  "Epsilon", b1, b2,
  "Gamma", b1, b2,
  "Iota", b1, b2
)

inc_delays <- inc_pars |>
  mutate(delay = discretize_gamma(support, Shape, Scale))

threshold <- function(x, tol = 1e-6) {
  x[x<tol] <- 0
  x
}
tf <- function(x) {
  cc <- threshold(convolve(x, rev(inc), type = "o"))
  cc / sum(cc)
}

incubation_plus_reportdelay <- function(inc) {
  list(t(apply(report, 1, function(x) {
    cc <- threshold(convolve(x, rev(unlist(inc)), type = "o"))
    cc / sum(cc)
  })))
}

inc_convolved <- inc_delays |> 
  rowwise() |>
  mutate(convolved = incubation_plus_reportdelay(delay))

probs <- apply(vmix_s, 2, function(x) list(x)) |> 
  as_tibble() |> 
  pivot_longer(everything(), names_to = "Variant", values_to = "probs")

listy <- left_join(inc_convolved, probs)
library(Matrix)
make_cmat <- function(conv, p) { ## correct??
  dims <- dim(conv)
  conv <- conv[,dims[2]:1]
  ix <- rep(1:dims[1], times = dims[2])
  jx <- ix + rep(0:(dims[2] - 1), each = dims[1])
  Cmat <- sparseMatrix(i = ix, j = jx, x = c(conv))
  Cmat <- t(Cmat[,-c(1:(dims[2] - 1))])
  list(drop0(p * Cmat))
}

listy <- listy |>
  mutate(Cmat = make_cmat(convolved, probs))

saveRDS(listy, "data/ca-convolution-mat.rds")
