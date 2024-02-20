library(tidyverse)
library(covidcast)
library(Matrix)
library(reticulate)
pd <- import("pandas")

##############################################################################################################################################
# Some initial settings
start_date = as.Date("2020-03-01") 
end_date =  as.Date("2023-03-01") 

# For op (number of days for negative delays)
daysbefore = 7

##############################################################################################################################################
# Create folders for each state's results in data folder
setwd(paste0("/Users/admin/Downloads/variant-deconvolve/data"))
for(state in state.abb){
  dir <- file.path("data", state) 
  if (!dir.exists(dir)) dir.create(dir)
}

##############################################################################################################################################
# Source necessary files to run tf cv
setwd(paste0("/Users/admin/Downloads/variant-deconvolve/"))
source("utils-arg.R")
source("utils-enlist.R")
source("cv_estimate_tf.R") 

##############################################################################################################################################
# Load covidcast case data (for plotting later on)

cases_df <- covidcast_signal("jhu-csse", "confirmed_incidence_num",
                          start_day = start_date, end_day = end_date,
                          as_of = as.Date("2023-07-06"),
                          geo_type = "state") %>%
  select(geo_value, time_value, version = issue, cases = value) %>% 
  mutate(geo_value = toupper(geo_value))
####################################################################################################################################
# Helper functions

make_cmat <- function(conv, daysbefore = 0) { 
  dims <- dim(conv)
  conv <- conv[,dims[2]:1]
  ix <- rep(1:dims[1], times = dims[2])
  jx <- ix + rep(0:(dims[2] - 1), each = dims[1])
  Cmat <- sparseMatrix(i = ix, j = jx, x = c(conv))
  Cmat <- Cmat[,-c(1:(dims[2] - 1 - daysbefore))]
  Cmat <- Cmat[,1:(ncol(Cmat) - daysbefore)] #%%
  list(drop0(Cmat)) 
}

plotter <- function(predmat) {
  as_tibble(predmat) |> 
    mutate(time = 1:nrow(predmat)) |>
    pivot_longer(-time) |> 
    ggplot(aes(time, y=value, fill = name)) + 
    geom_area(position = "stack") + 
    theme_bw() + 
    scale_fill_viridis_d()
}

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

##############################################################################################################################################
# Settings for incubation periods
b1 <- 5.1^2 / 2.7^2
b2 <- 2.7^2 / 5.1

dmsd <- mv_from_lognormal(1.25, 0.34)
inc_pars <- tribble(
  ~Variant, ~Shape, ~Scale,
  "Other", 3.05, 1.95,
  "Alpha", 4.94^2 / 2.19^2, 2.19^2 / 4.94,
  "Omicron", 3.03^2 / 1.33^2, 1.33^2 / 3.03,
  "Delta", dmsd[1]^2 / dmsd[2]^2, dmsd[2]^2 / dmsd[1],
  "Beta", b1, b2,
  "Epsilon", b1, b2,
  "Gamma", b1, b2,
  "Iota", b1, b2
)
##############################################################################################################################################

state = "CA"

setwd(paste0("/Users/admin/Downloads/naive_delay_dist/data/naive_delay_distributions/", state))
state_op_list <- pd$read_pickle(paste0(state, "_Empshrink_delay_distribution_feb8_op.p"))
state_report = reshape::melt(state_op_list) %>% 
  rename(Date = L1, dist = value, delay = indices) %>%
  mutate(delay = delay - daysbefore - 1, Date = as.Date(Date), state = state)

setwd(paste0("/Users/admin/Downloads/variant-deconvolve/"))
vmix <- read_rds("data/seq_prop_df.rds") |>
  filter(State == state)

# Here, data is daily, not only the observations
vmix_s <- vmix |> select(Alpha:Other) |> as.matrix()

# plotter(vmix_s)

report <- state_report |>
  pivot_wider(names_from = delay, values_from = dist) |>
  select(-(Date:state)) |>
  as.matrix()

# period up to 2021-12-01
ed <- which(vmix$Date == end_date)
report <- report[1:ed, ]
vmix_s <- vmix_s[1:ed, ]

# variant-specific delays

max_inc_days <- 21
support <- 0:max_inc_days #%% Start at 0 with prob = 0

inc_delays <- inc_pars |>
  mutate(delay = discretize_gamma(support, Shape, Scale))

inc_convolved <- inc_delays |> 
  rowwise() |>
  mutate(convolved = incubation_plus_reportdelay(delay))

probs <- apply(vmix_s, 2, function(x) list(x)) |> 
  as_tibble() |> 
  pivot_longer(everything(), names_to = "Variant", values_to = "probs")

listy <- left_join(inc_convolved, probs)

listy <- listy |>
  mutate(Cmat = make_cmat(convolved, daysbefore = daysbefore)) #%% Make sure dimensions of this are slightly larger if using daysbefore parameter (so Cmat and probs in listy results are the same dim)

setwd(paste0("data/", state)) # Where the results and plots are to be saved for the state
write_rds(listy, "convolution-mat-list.rds")



# Deconvolutions
setwd(paste0("/Users/admin/Downloads/naive_delay_dist/data/naive_delay_distributions/", state))

# Positive specimen to report date
state_pr_list <- pd$read_pickle(paste0(state, "_Empshrink_delay_distribution_d60c_feb8_pr.p"))
state_pr_mat <- do.call(rbind, state_pr_list) # matrix and bind rows so 1096 x 61 or 82

Cmat <- make_cmat(state_pr_mat)[[1]]

state_cases = cases_df %>% filter(geo_value == state) %>% pull(cases)
if((length(state_cases) != nrow(Cmat)) | (length(state_cases) != ncol(Cmat))) cli::cli_abort("`Cmat` length is not the same as the number of days of cases.")

# CV estimate using tf
res_pr <- cv_estimate_tf(y = state_cases, 
                         cmat = Cmat,
                         error_measure = "mse")

# Plot of the thetas for all lambdas
matplot(res_pr$full_fit$thetas, type = "l")
# Plot line of thetas for lambda_min
matplot(res_pr$full_fit$thetas[,which(res_pr$lambda == res_pr$lambda.min)], type = "l")

final_thetas_pr = res_pr$full_fit$thetas[,which(res_pr$lambda == res_pr$lambda.min)]

for(var in inc_pars$Variant){
  if(length(final_thetas_pr) != length(probs[probs$Variant == var, ]$probs[[1]])) cli::cli_warn(paste0("Length of `final_thetas_pr` is not the same as the variant probabilities for a variant ", var)) 
}



# Infection onset to positive specimen date
# Load convolution data and Cmat by variant
setwd(paste0("data/", state)) # Where the results and plots are to be saved for the state
params <- read_rds("convolution-mat-list.rds")

final_thetas_op_list = vector(mode = "list", length = length(inc_pars$Variant))
names(final_thetas_op_list) <- inc_pars$Variant
for(var in inc_pars$Variant){
  cmat_var = params[params$Variant == var,]$Cmat[[1]]
  y_var = final_thetas_pr * probs[probs$Variant == var, ]$probs[[1]] #%% final_thetas_pr, 
  
  # Checks that dimensions of y_var and cmat are reasonable
  if(nrow(cmat_var) != ncol(cmat_var)) cli::cli_abort("`cmat` is not square. Check the dimensions.")
  if(nrow(cmat_var) != length(y_var)) cli::cli_abort("Nrow of `cmat` is not the same as the length of `y_var`. Check the dimensions and dates used of each.")
  if(ncol(cmat_var) != length(y_var)) cli::cli_abort("Ncol of `cmat` is not the same as the length of `y_var`. Check the dimensions and dates used of each.")
  
  # CV estimate using tf 
  res_op <- cv_estimate_tf(y_var, 
                           cmat = cmat_var,
                           error_measure = "mse")
  
  # Plot of the thetas for all lambdas
  matplot(res_op$full_fit$thetas, type = "l")
  # Plot line of thetas for lambda_min
  matplot(res_op$full_fit$thetas[,which(res_op$lambda == res_op$lambda.min)], type = "l")
  
  final_thetas_op = res_op$full_fit$thetas[,which(res_op$lambda == res_op$lambda.min)]
  final_thetas_op_list[[var]] = data.frame(time_value = seq(start_date, end_date, by = "day"), geo_value = state, infect = final_thetas_op)
}
final_thetas_op_df_state = bind_rows(final_thetas_op_list, .id="variant")

setwd(paste0("data/", state)) # Where the results and plots are to be saved for the state
write_rds(final_thetas_op_df_state, "final-thetas-df.rds")

# Produce plot of all infections by variant
ggplot(final_thetas_op_df, aes(time_value, infect, colour=variant)) +
  geom_line()
ggsave(filename = paste0(state, "infect_by_variant.png"))

# sum(final_thetas_op_df$infect) # too large compared to other two 12,000,000 --> 18,000,000
# sum(final_thetas_pr)
# sum(state_cases)
