#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <cuda.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#define index(i,j,ld) (((j)*(ld))+(i))

typedef struct {
    int n,p,num_lambda;
    thrust::device_ptr<float> X, y;
    thrust::host_vector<float> lambda;
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

struct absolute_value
{
    __host__ __device__
        float operator()(const float& x) const { 
            if (x < 0) return -x;
            else return x;
        }
};

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

/*
  Finds components of the grad that have absolute value > lambda
  saves into gpu_isActive
*/
__global__ void checkKKT(float* gpu_grad, int* gpu_isActive, float lambda, int p)
{
  int k = threadIdx.x + blockDim.x*blockIdx.x;
  if (k < p){
    if ((gpu_grad[k] < -lambda) || (gpu_grad[k] > lambda)) gpu_isActive[k] = 1;
  }
}

/*
  gpu_beta is a vector of length p
  thresholds gpu_beta at lambda
*/
__global__ void softKernel(float *gpu_beta, float lambda, int p)
{
  int k = threadIdx.x + blockDim.x*blockIdx.x;
  if(k < p){
    float beta_k = gpu_beta[k];
    if ((beta_k > -lambda) && (beta_k < lambda)) gpu_beta[k] = 0;
    else if (beta_k > lambda) beta_k = gpu_beta[k] - lambda;
    else if (beta_k < -lambda) beta_k = gpu_beta[k] + lambda;
  }
}


  
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
float calcNegLL(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, thrust::device_vector<float> pvector, int j);
void gradStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j);
void proxCalc(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j);
void nestStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter);
int checkStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j);
int checkCrit(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter);
void shutdown(data* ddata, coef* dcoef, opt* dopt, misc* dmisc);
float device_vector2Norm(thrust::device_vector<float> x);
void device_vectorSoftThreshold(thrust::device_vector<float> x, thrust::device_vector<float>, float lambda);
void device_vectorSgemv(thrust::device_vector<float> A,
                          thrust::device_vector<float> x,
                          thrust::device_vector<float> b,
                          int n, int p);
void device_vectorCrossProd(thrust::device_vector<float> X,
                              thrust::device_vector<float> y,
                              thrust::device_vector<float> b,
                              int n, int p);
 


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
    data* ddata = NULL; coef* dcoef = NULL; opt* dopt = NULL; misc* dmisc = NULL;

    //allocate pointers, init cublas
    init(ddata, dcoef, dopt, dmisc,
         X, y, n[0], p[0], lambda, num_lambda[0],
         type[0], beta, maxIt[0], thresh[0], gamma[0],
         t[0], reset[0]);

    //solve
    //pathSol(ddata, dcoef, dopt, dmisc, beta);

    //shutdown
    shutdown(ddata, dcoef, dopt, dmisc);
  }

  void init(data* ddata, coef* dcoef, opt* dopt, misc* dmisc,
            float* X, float* y, int n, int p, float* lambda, int num_lambda,
            int type, float* beta, int maxIt, float thresh, float gamma,
            float t, int reset)
  {
    int number_of_devices;
    cudaGetDeviceCount(&number_of_devices);
    cudaSetDevice(0);
    cublasInit();

    ddata = (data*)malloc(sizeof(data));
    dcoef = (coef*)malloc(sizeof(coef));
    dopt = (opt*)malloc(sizeof(opt));
    dmisc = (misc*)malloc(sizeof(misc));

    /* Set data variables */

    ddata->X = thrust::device_vector<float>(X, X+(n*p)).data();
    ddata->y = thrust::device_vector<float>(y, y+n).data();
    ddata->lambda = thrust::host_vector<float>(lambda, lambda+num_lambda);
    ddata->n = n;
    ddata->p = p;
    ddata->num_lambda = num_lambda;

    /* Set coef variables */

    /*dcoef->beta = thrust::device_vector<float>(p,0);
    dcoef->beta_old = thrust::device_vector<float>(p,0);
    dcoef->theta = thrust::device_vector<float>(p,0);
    dcoef->theta_old = thrust::device_vector<float>(p,0);
    dcoef->momentum = thrust::device_vector<float>(p,0);*/

    /* Set optimization variables */

    /*dopt->eta = thrust::device_vector<float>(p,0);
    dopt->yhat = thrust::device_vector<float>(n,0);
    dopt->residuals = thrust::device_vector<float>(n,0);
    dopt->grad = thrust::device_vector<float>(p,0);
    dopt->U = thrust::device_vector<float>(p,0);
    dopt->diff_beta = thrust::device_vector<float>(p,0);
    dopt->diff_theta = thrust::device_vector<float>(p,0);*/

    /* Set misc variables */

    /*dmisc->type = type;
    dmisc->maxIt = maxIt;
    dmisc->gamma = gamma;
    dmisc->t = t;
    dmisc->reset = reset;
    dmisc->thresh = thresh;*/
  }

  void pathSol(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, float* beta)
  {
    int j;
    for (j=0; j < ddata->num_lambda; j++){
      dcoef->beta_old = dcoef->beta;
      dcoef->theta_old = dcoef->theta;
      singleSolve(ddata, dcoef, dopt, dmisc, j);

      int startIndex = j*ddata->p;
      int i;
      for(i=0; i < ddata->p; i++){
        beta[startIndex+i] = dcoef->beta[i];
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
                  thrust::device_vector<float> pvector, int j)
  {
    device_vectorSgemv(ddata->X, pvector, dopt->eta, ddata->n, ddata->p);
    switch (dmisc->type)
    {
      case 0:  //normal
      {
        dopt->nLL = 0.5 * device_vector2Norm(dopt->residuals); break;
      }
      default:  //default to normal
      { 
        dopt->nLL = 0.5 * device_vector2Norm(dopt->residuals); break;
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
        device_vectorSgemv(ddata->X, dcoef->beta, dopt->yhat, ddata->n, ddata->p);
        //residuals = y - yhat
        thrust::transform(ddata->y.begin(), ddata->y.end(),
                          dopt->yhat.begin(),
                          dopt->residuals.begin(),
                          thrust::minus<float>());
        //grad = X^T residuals
        device_vectorCrossProd(ddata->X, dopt->residuals, dopt->grad, ddata->n, ddata->p);
        //U = -t * grad + beta
        thrust::transform(dopt->grad.begin(), dopt->grad.end(),
                          dcoef->beta.begin(),
                          dopt->U.begin(),
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
        device_vectorSoftThreshold(dopt->U, dcoef->theta, ddata->lambda[j] * dmisc->t);
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
    thrust::transform(dcoef->theta.begin(), dcoef->theta.end(),
                      dcoef->theta_old.begin(),
                      dcoef->momentum.begin(),
                      thrust::minus<float>());
    float scale = (float) (iter % dmisc->reset) / (iter % dmisc->reset + 3);
    //beta = theta + scale*momentum
    thrust::transform(dcoef->momentum.begin(), dcoef->momentum.end(),
                      dcoef->theta.begin(),
                      dcoef->beta.begin(),
                      saxpy(scale));
    dcoef->theta_old = dcoef->theta;
  }

  int checkStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j)
  {
    float nLL = calcNegLL(ddata, dcoef, dopt, dmisc, dcoef->theta, j);
    //iprod is the dot product of diff and grad
    float iprod = thrust::inner_product(dopt->diff_theta.begin(), dopt->diff_theta.end(),
                                        dopt->grad.begin(),
                                        0); 
    float sumSquareDiff = device_vector2Norm(dopt->diff_theta);

    int check = (int)(nLL < (dopt->nLL + iprod + sumSquareDiff) / (2 * dmisc->t));
    if (check == 0) dmisc->t = dmisc->t * dmisc->gamma;
      
    return check;
  }

  int checkCrit(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter)
  {
    absolute_value unary_op;
    thrust::maximum<float> binary_op;
    float init = 0;
    float move = thrust::transform_reduce(dopt->diff_beta.begin(), dopt->diff_beta.end(),
                                          unary_op, init, binary_op);
      
    return (iter > dmisc->maxIt) || (move < dmisc->thresh);
  }

  void shutdown(data* ddata, coef* dcoef, opt* dopt, misc* dmisc)
  {
    free(ddata); free(dcoef); free(dopt); free(dmisc);
    cublasShutdown();
  }

  /*
    MISC MATH FUNCTIONS
  */

  // ||x||_2^2
  float device_vector2Norm(thrust::device_vector<float> x)
  {  
    square unary_op;
    thrust::plus<float> binary_op;
    float init = 0;
    return thrust::transform_reduce(x.begin(), x.end(), unary_op, init, binary_op);
  }

  // b = X^T y
  void device_vectorCrossProd(thrust::device_vector<float> X,
                              thrust::device_vector<float> y,
                              thrust::device_vector<float> b,
                              int n, int p)
  {
    cublasSgemv('t', n, p, 1,
                thrust::raw_pointer_cast(&X[0]), n,
                thrust::raw_pointer_cast(&y[0]), 1,
                0, thrust::raw_pointer_cast(&b[0]), 1); 
  }

  // b = Ax
  void device_vectorSgemv(thrust::device_vector<float> A,
                          thrust::device_vector<float> x,
                          thrust::device_vector<float> b,
                          int n, int p)
  {
    cublasSgemv('n', n, p, 1,
                thrust::raw_pointer_cast(&A[0]), n,
                thrust::raw_pointer_cast(&x[0]), 1,
                0, thrust::raw_pointer_cast(&b[0]), 1);
  }

  // S(x, lambda)
  void device_vectorSoftThreshold(thrust::device_vector<float> x,
                                  thrust::device_vector<float> dest,
                                  float lambda)
  {
    thrust::transform(x.begin(), x.end(), dest.begin(), soft_threshold(lambda));
  }

}

int main() {
  return 0;
}
