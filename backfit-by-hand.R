thetas <- sapply(cmats, \(x) drop(t(x) %*% y))
zs <- matrix(0, m, ncomponents)
us <- matrix(0, m, ncomponents)
r <- y
for (j in 1:ncomponents) r = drop(r - cmats[[j]] %*% thetas[,j])
r_old <- r
for (i in 1:10) {
  for (j in 1:ncomponents) {
    py <- drop(cmats[[j]] %*% thetas[,j])
    r <- r + py
    o <- admm_test(200, 3, r, x, cmats[[j]], 1000, 1, 1e-3)
    thetas[,j] <- o$theta
    zs[,j] <- o$z
    us[,j] <- o$u
    py <- drop(cmats[[j]] %*% o$theta)
    r <- r - py
  }
  print(paste("i =", i, "norm =", sqrt(sum((r - r_old)^2)) / sqrt(n)))
  r_old = r
}
