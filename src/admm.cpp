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
  Eigen::VectorXd z_old(n);
  z_old.setZero();
  z_old += z;
  int m = z.size();
  VectorXd Dth(m);
  VectorXd tmp_n(n);
  VectorXd r(m);
  SparseMatrix<double> cDD = DD * rho + Cmat.transpose() * Cmat;
  
  // small ridge penalty?
  // for (int i = 0; i < n; i++) {
  //   cDD.diagonal()(i) += .001;
  // }
  Eigen::VectorXd Cty = Cmat.transpose() * y;
  qradmm.compute(cDD);
  
  // Rcout << cDD.nonZeros() << "\n";
  // Rcout << cDD.sum() << "\n";
  
  int niter = 0;
  for (int iter = 0; iter < M; iter++) {
    // tmp_n.setZero();
    // Dth.setZero();
    // r.setZero();
    niter++;
    if (iter % 1000 == 0) Rcpp::checkUserInterrupt();
    // solve for primal variable - theta:
    tmp_n = doDtv(z + u, korder, x) * rho;
    tmp_n += Cty;
    
    theta = qradmm.solve(tmp_n);
    // solve for alternating variable - z:
    Dth = doDv(theta, korder, x);
    //Dth -= u;
    //z.setZero();
    z = dptf(Dth - u, lam_z);
    // update dual variable - u:
    u -= Dth - z;
    u += z;

    // primal residuals:
    // r.setZero();
    r = Dth - z;
    r_norm = r.norm() / sqrtn;
    // dual residuals:
    tmp_n = doDtv(z - z_old, korder, x);
    s_norm = rho * tmp_n.norm() / sqrtn;
    // stopping criteria check:
    if (r_norm < tol && s_norm < tol) break;

    // auxiliary variables update:
    z_old = z;
    // Rcout << "niter = " << niter << ", r_norm = " << r_norm << ", s_norm = " << s_norm << "\n";
  }
}


// [[Rcpp::export()]]
List admm_testing(int M,
                  int korder,
                  Eigen::VectorXd y,
                  NumericVector x,
                  Eigen::SparseMatrix<double> Cmat,
                  double rho,
                  double lam_z,
                  double tol) {
  Eigen::SparseMatrix<double> DkDk;
  Eigen::SparseMatrix<double> Dk = get_Dtil(korder, x); 
  DkDk = Dk.transpose() * Dk;
  int mm = Dk.rows();
  int n = y.size();
  
  Eigen::VectorXd theta(n);
  Eigen::VectorXd z(mm);
  Eigen::VectorXd u(mm);
  
  theta.setZero();
  z.setZero();
  u.setZero();
  
  admm_gauss(M, korder, y, x, Cmat, theta, z, u, rho, lam_z, DkDk, tol);
  List out = List::create(
    Named("theta") = theta,
    Named("z") = z,
    Named("u") = u
  );
  return out;
}
