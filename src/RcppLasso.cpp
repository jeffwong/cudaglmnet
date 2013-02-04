// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Rcpp;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::Lower;
    
typedef Map<MatrixXd> MapMatd;
typedef Map<VectorXd> MapVecd;

//Compute crossprod(X)
inline MatrixXd AtA(const MatrixXd& A) {
    int n(A.cols());
    return MatrixXd(n,n).setZero().selfadjointView<Lower>()
    .rankUpdate(A.adjoint());
}

//Compute crossprod(X,Y)
inline MatrixXd AtB(const MatrixXd& A, const MatrixXd& B) {
    return A.adjoint() * B;
}

inline VectorXd softthreshold(VectorXd x, double lambda) {
    for (int i = 0; i < x.size(); i++) {
        if (x[i] <= lambda & x[i] >= -lambda) x[i] = 0;
        else x[i] = x[i] - lambda;
    }
    return x;
}

// [[Rcpp::export]]
List RcppLasso(NumericMatrix Xr, NumericVector yr, NumericMatrix Br, NumericVector lambdar,
               double stepsize, double maxIters) {
    const MapMatd X(as<MapMatd>(Xr));
    const MapVecd y(as<MapVecd>(yr));
    const MapVecd lambda(as<MapVecd>(lambdar));
    //const MapMatd B(as<MapMatd>(Br));
    MatrixXd B(MatrixXd(y.size(), lambda.size()));
    for (int i = 0; i < y.size(); i++) {
        for (int j = 0; j < lambda.size(); j++) {
            B(i,j) = Br(i,j);
        }
    } 

    const MatrixXd XtX(AtA(X));
    const MatrixXd Xty(AtB(X, y));

    for (int i = 0; i < lambda.size(); i++) {
        for (int iter = 0; iter < maxIters; iter++) {
            B.col(i) = softthreshold(B.col(i) + stepsize*(Xty - XtX*B.col(i)), lambda[i]);
        }
    }

    return List::create(Named("coefficients") = B);
}

