#ifndef __BACKFITTING_H
#define __BACKFITTING_H

void backfitting(int korder,
                 Eigen::VectorXd y,
                 Rcpp::NumericVector x,
                 Rcpp::List Cmats,
                 Eigen::MatrixXd thetas,
                 Eigen::MatrixXd zs,
                 Eigen::MatrixXd us,
                 double rho,
                 double lam_z,
                 Eigen::SparseMatrix<double> DD,
                 double tol,
                 int biters,
                 int admmiters);

#endif