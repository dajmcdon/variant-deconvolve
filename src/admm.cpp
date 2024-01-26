#include <RcppEigen.h>
#include <Eigen/Sparse>
#include "utils.h"
#include "dptf.h"
#include "admm.h"

typedef Eigen::COLAMDOrdering<int> Ord;

using Eigen::SparseMatrix;
using Eigen::SparseQR;
using Eigen::VectorXd;
SparseQR<SparseMatrix<double>, Ord> qradmm;

using namespace Rcpp;

/**
 * ADMM for Gaussian Deconvolution
 * @param M maximum iteration of the algos
 * @param n signal length
 * @param korder degree of penalty
 * @param y observed signals
 * @param x signal locations
 * @param Cmat convolution matrix (sparse)
 * @param theta primal variable of length `n`
 * @param z auxiliary variable of length `n-korder`
 * @param u dual variable of length `n-korder`
 * @param rho Lagrangian parameter of ADMM
 * @param lam_z hyperparameter of the auxiliary step of ADMM
 * @param DD D^T * D
 * @param tol tolerance of stopping criteria
 * @param iter interation index
 */
void admm_gauss(int M,
                int korder,
                Eigen::VectorXd y,
                NumericVector x,
                Eigen::SparseMatrix<double> Cmat,
                Eigen::VectorXd& theta,
                Eigen::VectorXd& z,
                Eigen::VectorXd& u,
                double rho,
                double lam_z,
                Eigen::SparseMatrix<double> DD,
                double tol) {
  double r_norm = 0.0;
  double s_norm = 0.0;
  int n = theta.size();
  double sqrtn = sqrt(n);
  Eigen::VectorXd z_old = z;
  int m = z.size();
  VectorXd Dth(m);
  VectorXd tmp_n(n);
  SparseMatrix<double> cDD = DD * n * rho + Cmat.transpose() * Cmat;
  Eigen::VectorXd Cty = Cmat * y;
  qradmm.compute(cDD);

  for (int iter = 0; iter < M; iter++) {
    if (iter % 1000 == 0) Rcpp::checkUserInterrupt();
    // solve for primal variable - theta:
    tmp_n = doDtv(z - u, korder, x) * n * rho;
    tmp_n += Cty;
    theta = qradmm.solve(tmp_n);
    // solve for alternating variable - z:
    Dth = doDv(theta, korder, x);
    Dth += u;
    z = dptf(Dth, lam_z);
    // update dual variable - u:
    Dth -= z;
    u += Dth;

    // primal residuals:
    r_norm = Dth.norm() / sqrtn;
    // dual residuals:
    tmp_n = doDtv(z - z_old, korder, x);
    s_norm = rho * tmp_n.norm() / sqrtn;
    // stopping criteria check:
    if (r_norm < tol && s_norm < tol) break;

    // auxiliary variables update:
    z_old = z;
  }
}
