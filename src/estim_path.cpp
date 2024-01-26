#include <RcppEigen.h>
#include <Eigen/Sparse>
#include "backfitting.h"
#include "utils.h"
#include "dptf.h"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(dspline)]]

typedef Eigen::COLAMDOrdering<int> Ord;

using Eigen::SparseMatrix;
using Eigen::SparseQR;
using Eigen::VectorXd;
SparseQR<SparseMatrix<double>, Ord> qr;

using namespace Rcpp;

// [[Rcpp::export]]
List estim_path(Eigen::VectorXd y,
                NumericVector x,
                List Cmats,
                int korder,
                NumericVector lambda,
                double lambdamax = -1,
                double lambdamin = -1,
                int nsol = 100,
                double rho = -1,
                int maxadmm_iter = 1e3,
                int maxbackfit_iter = 100,
                double tolerance = 1e-3,
                double lambda_min_ratio = 1e-4,
                int verbose = 0) {
  int ncomponents = Cmats.size();
  Eigen::SparseMatrix<double> Cmat = Cmats[1];
  int n = Cmat.cols();
  int m = Cmat.rows();

  // Placeholders for solutions
  Eigen::MatrixXd thetas(n, ncomponents);
  List all_thetas(nsol);
  for (int i = 0; i < nsol; i++) all_thetas[i] = thetas;
  NumericVector niter(nsol);

  // Build D matrices as needed
  Eigen::SparseMatrix<double> D;
  Eigen::SparseMatrix<double> Dk;
  Eigen::SparseMatrix<double> DkDk;
  D = get_D(korder, x);
  qr.compute(D.transpose());
  if (korder > 0) {
    Dk = get_Dtil(korder, x);
    DkDk = Dk.transpose() * Dk;
    int mm = Dk.rows();
  }

  // Generate lambda sequence if necessary
  if (abs(lambda[nsol - 1]) < tolerance / 100 && lambdamax <= 0) {
    Eigen::SparseMatrix<double> Cmat_sum;
    for (int i = 0; i < ncomponents; i++) {
      Eigen::SparseMatrix<double> Cmat = Cmats[i];
      Cmat_sum += Cmat.transpose() * Cmat;
    }
    VectorXd b(n - korder);
    VectorXd Cy = Cmat_sum * y;
    b = qr.solve(Cy);
    NumericVector bp = evec_to_nvec(b);
    lambdamax = max(abs(bp));
  }
  create_lambda(lambda, lambdamin, lambdamax, lambda_min_ratio, nsol);

  // ADMM parameters
  double _rho;

  // ADMM variables
  Eigen::MatrixXd beta(n, ncomponents);
  Eigen::MatrixXd alpha(m, ncomponents);
  Eigen::MatrixXd u(m, ncomponents);
  int iters = 0;
  int nsols = nsol;

  // Outer loop to compute solution path
  for (int i = 0; i < nsol; i++) {
    if (verbose > 0) Rcout << ".";
    Rcpp::checkUserInterrupt();

    _rho = (rho < 0) ? lambda[i] : rho;
    backfitting(korder, y, x, Cmats, beta, alpha, u, _rho,
                lambda[i] / _rho, DkDk, tolerance, maxbackfit_iter,
                maxadmm_iter);

    // Store solution
    all_thetas(i) = beta;

    // Verbose handlers
    if (verbose > 1) Rcout << niter(i);
    if (verbose > 2) Rcout << "(" << lambda(i) << ")";
    if (verbose > 0) Rcout << std::endl;
  }

  // Return
  List out = List::create(
    Named("thetas") = all_thetas,
    Named("lambda") = lambda,
    Named("korder") = korder
  );

  return out;
}
