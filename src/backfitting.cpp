#include <Eigen/Sparse>
#include <RcppEigen.h>
#include "admm.h"
#include "utils.h"

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
  
  int n = thetas.rows();
  int m = zs.rows();
  int ncomponents = thetas.cols();
  int yn = y.size();
  double sqrtn = sqrt(n);
  // Eigen::SparseMatrix<double> Cmat(n, m);
  
  Eigen::VectorXd yhat(yn);
  Eigen::VectorXd r = y;
  Eigen::VectorXd r_diff(yn);
  for (int j = 0; j < ncomponents; j++) {
    r -= Cmats.middleCols(j * n, n) * thetas.col(j);
  }
  Eigen::VectorXd r_old = r;
  Eigen::VectorXd pyhat(yn);
  Eigen::SparseMatrix<double> Cmat;
  Eigen::VectorXd th(n);
  Eigen::VectorXd zv(m);
  Eigen::VectorXd uv(m);
  
  for (int iter = 0; iter < biters; iter++) {
    Rcout << "biter = " << iter << "\n";
    for (int j = 0; j < ncomponents; j++) {
      Cmat = Cmats.middleCols(j * n, n);
      pyhat = Cmat * thetas.col(j);
      r += pyhat; // r <- (y - (yhat - pyhat))
      th = thetas.col(j);
      zv = zs.col(j);
      uv = us.col(j);
      admm_gauss(admmiters, korder, r, x, Cmat, th,
                 zv, uv, rho, lam_z, DD, tol);
      thetas.col(j) = th;
      zs.col(j) = zv;
      us.col(j) = uv;
      pyhat = Cmat * th;
      r -= pyhat; // r <- (y - (yhat + pyhat))
    }
    r_diff = r - r_old;
    if (r_diff.norm() / sqrtn < tol) break;
    r_old = r;
  }
}

// [[Rcpp::export()]]
List backfitting_test(int korder,
                      int ncomponents,
                      Eigen::VectorXd y,
                      Rcpp::NumericVector x,
                      Eigen::SparseMatrix<double> Cmats,
                      double rho,
                      double lam_z,
                      double tol,
                      int biters,
                      int admmiters) {
  Eigen::SparseMatrix<double> DkDk;
  Eigen::SparseMatrix<double> Dk = get_Dtil(korder, x); 
  DkDk = Dk.transpose() * Dk;
  
  int m = Dk.rows();
  int n = Dk.cols();
  Eigen::MatrixXd thetas(n, ncomponents);
  Eigen::MatrixXd zs(m, ncomponents);
  Eigen::MatrixXd us(m, ncomponents);
  
  backfitting(korder, y, x, Cmats, thetas, zs, us, rho, lam_z, DkDk,
              tol, biters, admmiters);
  
  List out = List::create(
    Named("thetas") = thetas,
    Named("zs") = zs,
    Named("us") = us
  );
  return out;
}