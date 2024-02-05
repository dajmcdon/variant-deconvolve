#ifndef __BACKFITTING_H
#define __BACKFITTING_H

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
                 int admmiters);

Rcpp::List backfitting_test(int korder,
                            int ncomponents,
                            Eigen::VectorXd y,
                            Rcpp::NumericVector x,
                            Eigen::SparseMatrix<double> Cmats,
                            double rho,
                            double lam_z,
                            double tol,
                            int biters,
                            int admmiters);

#endif