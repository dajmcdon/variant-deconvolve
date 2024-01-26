#ifndef __ADMM_H
#define __ADMM_H

void admm_gauss(int M,
                int korder,
                Eigen::VectorXd y,
                Rcpp::NumericVector x,
                Eigen::SparseMatrix<double> Cmat,
                Eigen::VectorXd& theta,
                Eigen::VectorXd& z,
                Eigen::VectorXd& u,
                double rho,
                double lam_z,
                Eigen::SparseMatrix<double> DD,
                double tol);

#endif
