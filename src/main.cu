#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <cuda.h>
#include <cmath>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#define index(i,j,ld) (((j)*(ld))+(i))

typedef struct {
    int n,p,num_lambda;
    thrust::host_vector<float> lambda;
    thrust::device_ptr<float> X, y;
} data;

typedef struct {
    thrust::device_ptr<float> beta, beta_old, theta, theta_old, momentum;
} coef;

typedef struct {
    float nLL;
    thrust::device_ptr<float> eta, yhat, residuals, grad, U, diff_beta, diff_theta;
} opt;

typedef struct {
    int type, maxIt, reset;
    float gamma, t, thresh;
} misc;

struct square
{
    __host__ __device__
        float operator()(const float& x) const { 
            return x*x;
        }
};

struct soft_threshold
{
    const float lambda;

    soft_threshold(float _lambda) : lambda(_lambda) {}

    __host__ __device__
        float operator()(const float& x) const { 
            if (x > -lambda && x < lambda) return 0;
            else if (x > lambda) return x - lambda;
            else return x + lambda;
        }
};

struct saxpy
{
    const float a;

    saxpy(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

struct absolute_value
{
    __host__ __device__
        float operator()(const float& x) const { 
            if (x < 0) return (-1*x);
            else return x;
        }
};


  
extern "C"{


void activePathSol(float*, float*, int*, int*, float*, int*,
                   int*, float*, int*, float*, float*,
                   float*, int*);
void init(data*, coef*, opt*, misc*,
          float*, float*, int, int, float*, int,
          int, float*, int, float, float,
          float, int);
void pathSol(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, float* beta);
void singleSolve(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j);
float calcNegLL(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, thrust::device_ptr<float> pvector, int j);
void gradStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j);
void proxCalc(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j);
void nestStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter);
int checkStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j);
int checkCrit(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter);
void shutdown(data* ddata, coef* dcoef, opt* dopt, misc* dmisc);
float device_ptr2Norm(thrust::device_ptr<float> x, int length);
float device_ptrDot(thrust::device_ptr<float> x,
                    thrust::device_ptr<float> y,
                    int length);
float device_ptrMaxNorm(thrust::device_ptr<float> x, int length);
void device_ptrSoftThreshold(thrust::device_ptr<float> x, thrust::device_ptr<float>, int length, float lambda);
void device_ptrSgemv(thrust::device_ptr<float> A,
                     thrust::device_ptr<float> x,
                     thrust::device_ptr<float> b,
                          int n, int p);
void device_ptrCrossProd(thrust::device_ptr<float> X,
                         thrust::device_ptr<float> y,
                         thrust::device_ptr<float> b,
                         int n, int p);
thrust::device_ptr<float> makeDeviceVector(float* x, int size);
thrust::device_ptr<float> makeEmptyDeviceVector(int size);
 


  /*
    Entry point for R
    X is a matrix (represented as a 1d array) that is n by p
    y is a vector that is n by 1
  */
  void activePathSol(float* X, float* y, int* n, int* p, float* lambda, int* num_lambda,
                     int* type, float* beta, int* maxIt, float* thresh, float* gamma,
                     float* t, int* reset)
  { 
    //setup pointers
    data* ddata = (data*)malloc(sizeof(data));
    coef* dcoef = (coef*)malloc(sizeof(coef));
    opt* dopt = (opt*)malloc(sizeof(opt));
    misc* dmisc = (misc*)malloc(sizeof(misc));
 
    //allocate pointers, init cublas
    init(ddata, dcoef, dopt, dmisc,
         X, y, n[0], p[0], lambda, num_lambda[0],
         type[0], beta, maxIt[0], thresh[0], gamma[0],
         t[0], reset[0]);

    //solve
    pathSol(ddata, dcoef, dopt, dmisc, beta);

    //shutdown
    shutdown(ddata, dcoef, dopt, dmisc);
  }

  void init(data* ddata, coef* dcoef, opt* dopt, misc* dmisc,
            float* X, float* y, int n, int p, float* lambda, int num_lambda,
            int type, float* beta, int maxIt, float thresh, float gamma,
            float t, int reset)
  {
    cublasInit();

    /* Set data variables */

    ddata->X = makeDeviceVector(X, n*p);
    ddata->y = makeDeviceVector(y, n);
    ddata->lambda = thrust::host_vector<float>(lambda, lambda+num_lambda);
    ddata->n = n;
    ddata->p = p;
    ddata->num_lambda = num_lambda;

    /* Set coef variables */

    dcoef->beta = makeEmptyDeviceVector(p);
    dcoef->beta_old = makeEmptyDeviceVector(p);
    dcoef->theta = makeEmptyDeviceVector(p);
    dcoef->theta_old = makeEmptyDeviceVector(p);
    dcoef->momentum = makeEmptyDeviceVector(p);

    /* Set optimization variables */

    dopt->eta = makeEmptyDeviceVector(p);
    dopt->yhat = makeEmptyDeviceVector(n);
    dopt->residuals = makeEmptyDeviceVector(n);
    dopt->grad = makeEmptyDeviceVector(p);
    dopt->U = makeEmptyDeviceVector(p);
    dopt->diff_beta = makeEmptyDeviceVector(p);
    dopt->diff_theta = makeEmptyDeviceVector(p);

    /* Set misc variables */

    dmisc->type = type;
    dmisc->maxIt = maxIt;
    dmisc->gamma = gamma;
    dmisc->t = t;
    dmisc->reset = reset;
    dmisc->thresh = thresh;
  }

  void pathSol(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, float* beta)
  {
    int j = 0;
    for (j=0; j < ddata->num_lambda; j++){
      dcoef->beta_old = dcoef->beta;
      dcoef->theta_old = dcoef->theta;
      
      singleSolve(ddata, dcoef, dopt, dmisc, j);

      thrust::device_vector<float> dbeta(dcoef->beta, &dcoef->beta[ddata->p]);
      thrust::host_vector<float> hbeta = dbeta;
      int startIndex = j*ddata->p;
      int i = 0;
      for(i=0; i < ddata->p; i++){
        beta[startIndex+i] = hbeta[i];
      }
    }
  }

  void singleSolve(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j)
  {
    int iter = 0;
    while (checkCrit(ddata, dcoef, dopt, dmisc, j, iter) == 0)
    {
      calcNegLL(ddata, dcoef, dopt, dmisc, dcoef->beta, j);
      while (checkStep(ddata, dcoef, dopt, dmisc, j) == 0)
      {
        gradStep(ddata, dcoef, dopt, dmisc, j);
      }
      nestStep(ddata, dcoef, dopt, dmisc, j, iter);
      iter = iter + 1;
    }
  }

  float calcNegLL(data* ddata, coef* dcoef, opt* dopt, misc* dmisc,
                  thrust::device_ptr<float> pvector, int j)
  {
    device_ptrSgemv(ddata->X, pvector, dopt->eta, ddata->n, ddata->p);
    switch (dmisc->type)
    {
      case 0:  //normal
      {
        dopt->nLL = 0.5 * device_ptr2Norm(dopt->residuals, ddata->n);
        break;
      }
      default:  //default to normal
      { 
        dopt->nLL = 0.5 * device_ptr2Norm(dopt->residuals, ddata->n);
        break;
      }
    }
    return dopt->nLL;
  }

  void gradStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j)
  {
    switch (dmisc->type)
    {
      case 0:  //normal
      {
        //yhat = XB
        device_ptrSgemv(ddata->X, dcoef->beta,
                        dopt->yhat,
                        ddata->n, ddata->p);
        //residuals = y - yhat
        printf("grad step transform");
        thrust::transform(ddata->y, &ddata->y[ddata->n],
                          dopt->yhat,
                          dopt->residuals,
                          thrust::minus<float>());
        //grad = X^T residuals
        device_ptrCrossProd(ddata->X, dopt->residuals,
                            dopt->grad,
                            ddata->n, ddata->p);
        //U = -t * grad + beta
        thrust::transform(dopt->grad, &dopt->grad[ddata->p],
                          dcoef->beta,
                          dopt->U,
                          saxpy(-dmisc->t));
        proxCalc(ddata, dcoef, dopt, dmisc, j);
        break;
      }
      default:
      {
        break;
      }
    } 
  }

  void proxCalc(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j)
  {
    switch (dmisc->type)
    {
      case 0:  //normal
      {
        device_ptrSoftThreshold(dopt->U,
                                dcoef->theta,
                                ddata->p, ddata->lambda[j] * dmisc->t);
        break;
      }
      default:
      {
        break;
      }
    }
  }

  void nestStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter)
  {
    dcoef->beta_old = dcoef->beta;
    //momentum = theta - theta old
    /*thrust::transform(dcoef->theta, &dcoef->theta[ddata->p],
                      dcoef->theta_old,
                      dcoef->momentum,
                      thrust::minus<float>());
    */float scale = (float) (iter % dmisc->reset) / (iter % dmisc->reset + 3);
    //beta = theta + scale*momentum
    /*thrust::transform(dcoef->momentum, &dcoef->momentum[ddata->p],
                      dcoef->theta,
                      dcoef->beta,
                      saxpy(scale));
    */dcoef->theta_old = dcoef->theta;
  }

  int checkStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j)
  {
    float nLL = calcNegLL(ddata, dcoef, dopt, dmisc, dcoef->theta, j);
    //iprod is the dot product of diff and grad
    float iprod = device_ptrDot(dopt->diff_theta, dopt->grad, ddata->p);
    float sumSquareDiff = device_ptr2Norm(dopt->diff_theta, ddata->p);

    int check = (int)(nLL < ((dopt->nLL + iprod + sumSquareDiff) / (2 * dmisc->t)));
    if (check == 0) dmisc->t = dmisc->t * dmisc->gamma;
      
    return check;
  }

  int checkCrit(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter)
  {
    if (iter > dmisc->maxIt) return 1;
    else return 0;
    /*float move = device_ptrMaxNorm(dopt->diff_theta, ddata->p);  
    if ((iter > dmisc->maxIt) || (move < dmisc->thresh)) return 1;
    else return 0;*/
  }

  void shutdown(data* ddata, coef* dcoef, opt* dopt, misc* dmisc)
  {
    free(ddata); free(dcoef); free(dopt); free(dmisc);
    cublasShutdown();
  }

  /*
    MISC MATH FUNCTIONS
  */

  thrust::device_ptr<float> makeDeviceVector(float* x, int size)
  {
    thrust::device_vector<float> dx(x, x+size);
    return &dx[0];
  }

  thrust::device_ptr<float> makeEmptyDeviceVector(int size)
  {
    thrust::host_vector<float> x(size, 0);
    thrust::device_vector<float> dx = x;
    return &dx[0];
  }

  // ||x||_max
  float device_ptrMaxNorm(thrust::device_ptr<float> x, int length)
  {
    return thrust::transform_reduce(x, &x[length],
                                    absolute_value(), 0.0, thrust::maximum<float>());  
  }

  // ||x||_2^2
  float device_ptr2Norm(thrust::device_ptr<float> x, int length)
  {  
    return cublasSnrm2(length, thrust::raw_pointer_cast(x), 1);
  }

  // y = ax + y
  void device_ptrSaxpy(thrust::device_ptr<float> x,
                       thrust::device_ptr<float> y,
                       int length,float scale)
  {
    cublasSaxpy(length, scale, 
                thrust::raw_pointer_cast(x), 1,
                thrust::raw_pointer_cast(y), 1);
  }

  // <x,y>
  float device_ptrDot(thrust::device_ptr<float> x,
                      thrust::device_ptr<float> y,
                      int length)
  {  
    return cublasSdot(length, thrust::raw_pointer_cast(x), 1,
                      thrust::raw_pointer_cast(y), 1);
  }

  // b = X^T y
  void device_ptrCrossProd(thrust::device_ptr<float> X,
                           thrust::device_ptr<float> y,
                           thrust::device_ptr<float> b,
                           int n, int p)
  {
    cublasSgemv('t', n, p, 1,
                thrust::raw_pointer_cast(X), n,
                thrust::raw_pointer_cast(y), 1,
                0, thrust::raw_pointer_cast(b), 1); 
  }

  // b = Ax
  void device_ptrSgemv(thrust::device_ptr<float> A,
                       thrust::device_ptr<float> x,
                       thrust::device_ptr<float> b,
                       int n, int p)
  {
    cublasSgemv('n', n, p, 1,
                thrust::raw_pointer_cast(A), n,
                thrust::raw_pointer_cast(x), 1,
                0, thrust::raw_pointer_cast(b), 1);
  }

  // S(x, lambda)
  void device_ptrSoftThreshold(thrust::device_ptr<float> x,
                               thrust::device_ptr<float> dest,
                               int length, float lambda)
  {
    thrust::transform(x, &x[length],
                      dest,
                      soft_threshold(lambda));
  }

}

int main() {
  thrust::host_vector<float> X(1000,1);
  thrust::host_vector<float> y(100,1);
  thrust::host_vector<float> beta(10,1);

  int n = 100;
  int p = 10;
  float lambda = 1;
  int num_lambda = 1;
  int type = 0;
  int maxIt = 10;
  float thresh = 0.001;
  float gamma = 0.001;
  float t = 0.1;
  int reset = 5;

  activePathSol(&X[0], &y[0], &n, &p, &lambda, &num_lambda,
                &type, &beta[0], &maxIt, &thresh, &gamma,
                &t, &reset);
  return 0;
}
