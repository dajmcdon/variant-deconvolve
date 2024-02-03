#include <Eigen/Sparse>
#include <RcppEigen.h>
#include "admm.h"

using namespace Rcpp;

void backfitting(int korder,
                 Eigen::VectorXd y,
                 Rcpp::NumericVector x,
                 Eigen::SparseMatrix<double> Cmats,
                 Eigen::MatrixXd thetas,
                 Eigen::MatrixXd zs,
                 Eigen::MatrixXd us,
                 double rho,
                 double lam_z,
                 Eigen::SparseMatrix<double> DD,
                 double tol,
                 int biters,
                 int admmiters) {
  
  int n = y.size();
  int m = thetas.rows();
  double ncomponents = Cmats.cols() / m;
  double sqrtn = sqrt(n);
  Eigen::SparseMatrix<double> Cmat(n, m);
  
  Eigen::VectorXd yhat(n);
  Eigen::VectorXd r_diff(n);
  for (int j = 0; j < ncomponents; j++) {
    Cmat = Cmats(Eigen::placeholders::all, Eigen::seqN(j * m, m));
    yhat += Cmat * thetas.col(j);
  }
  Eigen::VectorXd r = y - yhat;
  Eigen::VectorXd r_old = r;
  Eigen::VectorXd pyhat(n);
  
  for (int iter = 0; iter < biters; iter++) {
    for (int j = 0; j < ncomponents; j++) {
      Cmat = Cmats(Eigen::placeholders::all, Eigen::seqN(j * m, m));
      pyhat = Cmat * thetas.col(j);
      r += pyhat; // r <- (y - (yhat - pyhat))
      Eigen::VectorXd th = thetas.col(j);
      Eigen::VectorXd zv = zs.col(j);
      Eigen::VectorXd uv = us.col(j);
      admm_gauss(admmiters, korder, r, x, Cmat, th,
                 zv, uv, rho, lam_z, DD, tol);
      pyhat = Cmat * th;
      r -= pyhat; // r <- (y - (yhat + pyhat))
      thetas.col(j) = th;
      zs.col(j) = zv;
      us.col(j) = uv;
    }
    r_diff = r - r_old;
    if (r_diff.norm() / sqrtn < tol) break;
  }
}