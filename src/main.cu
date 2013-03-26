#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <sys/mman.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/generate.h>
#include <thrust/inner_product.h>

#define DEBUG 0
#define TIME 1

typedef struct {
    int n,p,num_lambda;
    float* lambda;
    thrust::device_ptr<float> X, y;
} data;

typedef struct {
    thrust::device_ptr<float> beta, beta_old, theta, theta_old, momentum;
} coef;

typedef struct {
    float nLL;
    thrust::device_ptr<float> eta, yhat, residuals, grad, U, diff_beta, diff;
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

struct absolute_value
{
    __host__ __device__
        float operator()(const float& x) const { 
            if (x < 0) return (-1*x);
            else return x;
        }
};

void printDevPtr(thrust::device_ptr<float> x, int size)
{
  thrust::device_vector<float> d(x, x+size);
  int i = 0;
  for (i = 0; i < size; i++) std::cout << "D[" << i << "] = " << d[i] << std::endl; 
}

void init(data*, coef*, opt*, misc*,
          float*, float*, int, int, float*, int,
          int, float*, int, float, float,
          float, int);
void pathSol(data*, coef*, opt*, misc*, float*, cublasStatus_t, cublasHandle_t);
void singleSolve(data*, coef*, opt*, misc*, int, cublasStatus_t, cublasHandle_t);
float calcNegLL(data*, coef*, opt*, misc*, thrust::device_ptr<float>, int, cublasStatus_t, cublasHandle_t);
void computeFit(data*, coef*, opt*, misc*, cublasStatus_t, cublasHandle_t);
void gradStep(data*, coef*, opt*, misc*, int, cublasStatus_t, cublasHandle_t);
void proxCalc(data*, coef*, opt*, misc*, int, cublasStatus_t, cublasHandle_t);
void nestStep(data*, coef*, opt*, misc*, int, int, cublasStatus_t, cublasHandle_t);
int checkStep(data*, coef*, opt*, misc*, int, cublasStatus_t, cublasHandle_t handle );
int checkCrit(data*, coef*, opt*, misc*, int, int, cublasStatus_t, cublasHandle_t);
void shutdown(data*, coef*, opt*, misc*);
void device_ptr2Norm(thrust::device_ptr<float>, float*, int, cublasStatus_t, cublasHandle_t);
void device_ptrDot(thrust::device_ptr<float>, thrust::device_ptr<float>,
                   float*, int, cublasStatus_t, cublasHandle_t);
float device_ptrMaxNorm(thrust::device_ptr<float>, int);
void device_ptrSoftThreshold(thrust::device_ptr<float>, thrust::device_ptr<float>, float, int);
void device_ptrSgemv(thrust::device_ptr<float>,
                          thrust::device_ptr<float>,
                          thrust::device_ptr<float>,
                          int, int,
                          cublasStatus_t, cublasHandle_t);
void device_ptrCrossProd(thrust::device_ptr<float>,
                         thrust::device_ptr<float>,
                         thrust::device_ptr<float>,
                         int, int,
                         cublasStatus_t, cublasHandle_t) ;
thrust::device_ptr<float> makeDeviceVector(float*, int);
thrust::device_ptr<float> makeEmptyDeviceVector(int);
void device_ptrCopy(thrust::device_ptr<float>,
                    thrust::device_ptr<float>,
                    int);

  void init(data* ddata, coef* dcoef, opt* dopt, misc* dmisc,
            float* X, float* y, int n, int p, float* lambda, int num_lambda,
            int type, float* beta, int maxIt, float thresh, float gamma,
            float t, int reset)
  {
    if (DEBUG) printf("Inside init\n");

    /* Set data variables */
    ddata->lambda = lambda;
    ddata->n = n;
    ddata->p = p;
    ddata->num_lambda = num_lambda;

    /* Set misc variables */

    dmisc->type = type;
    dmisc->maxIt = maxIt;
    dmisc->gamma = gamma;
    dmisc->t = t;
    dmisc->reset = reset;
    dmisc->thresh = thresh;
  }

  void pathSol(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, float* beta,
               cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside pathSol\n");
    int j = 0;
    for (j=0; j < ddata->num_lambda; j++){
      singleSolve(ddata, dcoef, dopt, dmisc, j, stat, handle);
      int startIndex = j*ddata->p;
      cudaMemcpy(beta + startIndex, thrust::raw_pointer_cast(dcoef->beta),
                 sizeof(float) * ddata->p, cudaMemcpyDeviceToHost);
    }
  }

  void singleSolve(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                   cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside singleSolve\n");
    int iter = 0;
    do
    {
      calcNegLL(ddata, dcoef, dopt, dmisc, dcoef->beta, j, stat, handle);
      computeFit(ddata, dcoef, dopt, dmisc, stat, handle);
      do
      {
        gradStep(ddata, dcoef, dopt, dmisc, j, stat, handle);
      } while (checkStep(ddata, dcoef, dopt, dmisc, j, stat, handle) == 0);
      nestStep(ddata, dcoef, dopt, dmisc, j, iter, stat, handle);
      iter = iter + 1;
    } while (checkCrit(ddata, dcoef, dopt, dmisc, j, iter, stat, handle) == 0);
    if (TIME) printf("Finished on iteration %i\n", iter);
  }

  float calcNegLL(data* ddata, coef* dcoef, opt* dopt, misc* dmisc,
                  thrust::device_ptr<float> pvector, int j,
                  cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside calcNegLL\n");

    device_ptrSgemv(ddata->X, pvector, dopt->eta, ddata->n, ddata->p, stat, handle);
    switch (dmisc->type)
    {
      case 0:  //normal
      {
        float nll = 0;
        device_ptr2Norm(dopt->residuals, &nll, ddata->n, stat, handle);
        dopt->nLL = 0.5 * nll;
        break;
      }
      default:  //default to normal
      { 
        float nll = 0;
        device_ptr2Norm(dopt->residuals, &nll, ddata->n, stat, handle);
        dopt->nLL = 0.5 * nll;
        break;
      }
    }
    if (DEBUG) printf("calcNegLL nll %f\n", dopt->nLL);
    return dopt->nLL;
  }

  void computeFit(data* ddata, coef* dcoef, opt* dopt, misc* dmisc,
                  cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside computeFit\n");
    switch (dmisc->type)
    {
      case 0:  //normal
      {
        //yhat = XB
        device_ptrSgemv(ddata->X, dcoef->beta, dopt->yhat, ddata->n, ddata->p, stat, handle);
        //residuals = y - yhat
        thrust::transform(ddata->y, ddata->y + ddata->n,
                          dopt->yhat,
                          dopt->residuals,
                          thrust::minus<float>());
        cudaThreadSynchronize();
        //grad = X^T residuals
        device_ptrCrossProd(ddata->X, dopt->residuals, dopt->grad, ddata->n,
                            ddata->p, stat, handle);
        break;
      }
      default:
      {
        break;
      }
    }
  }

  void gradStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside gradStep\n");
    switch (dmisc->type)
    {
      case 0:  //normal
      {
        //U = t * grad + beta
        thrust::transform(dopt->grad, dopt->grad + ddata->p, 
                          dcoef->beta,
                          dopt->U,
                          dmisc->t * thrust::placeholders::_1 + thrust::placeholders::_2);
        cudaThreadSynchronize();
        proxCalc(ddata, dcoef, dopt, dmisc, j, stat, handle);
        break;
      }
      default:
      {
        break;
      }
    } 
  }

  void proxCalc(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside proxCalc\n");
    switch (dmisc->type)
    {
      case 0:  //normal
      {
        device_ptrSoftThreshold(dopt->U, dcoef->theta, ddata->lambda[j] * dmisc->t, ddata->p);
        break;
      }
      default:
      {
        device_ptrSoftThreshold(dopt->U, dcoef->theta, ddata->lambda[j] * dmisc->t, ddata->p);
        break;
      }
    }
  }

  int checkStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside checkStep\n");
    float nLL = dopt->nLL;
    
    //diff = beta-theta
    thrust::transform(dcoef->beta, dcoef->beta + ddata->p,
                      dcoef->theta,
                      dopt->diff,
                      thrust::minus<float>());
    cudaThreadSynchronize();
    //iprod is the dot product of diff and grad
    float iprod=0; device_ptrDot(dopt->diff, dopt->grad, &iprod, ddata->p, stat, handle);
    float sumSquareDiff=0; device_ptr2Norm(dopt->diff, &sumSquareDiff, ddata->p, stat, handle);

    //int check = (int)(nLL < ((dopt->nLL + iprod + sumSquareDiff) / (2 * dmisc->t)));
    int check = (int)(nLL < (dopt->nLL + iprod + (sumSquareDiff / (2 * dmisc->t))));
    if (check == 0) dmisc->t = dmisc->t * dmisc->gamma;
    return check;
  }

  void nestStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside nestStep\n");
    //momentum = theta - theta old
    thrust::transform(dcoef->theta, dcoef->theta + ddata->p,
                      dcoef->theta_old,
                      dcoef->momentum,
                      thrust::minus<float>());
    cudaThreadSynchronize();
    int cycle = iter % dmisc->reset;
    float scale = ((float) cycle) / (cycle + 3);
    //beta = theta + scale*momentum
    thrust::transform(dcoef->momentum, dcoef->momentum + ddata->p,
                      dcoef->theta,
                      dcoef->beta,
                      scale * thrust::placeholders::_1 + thrust::placeholders::_2);
    cudaThreadSynchronize();
    device_ptrCopy(dcoef->theta, dcoef->theta_old, ddata->p);
  }

  int checkCrit(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside checkCrit\n");
    float move = device_ptrMaxNorm(dopt->diff, ddata->p); 
    if (DEBUG) printf("move %f\n", move);
    return ((iter > dmisc->maxIt) || (move < dmisc->thresh));
  }

  void shutdown(data* ddata, coef* dcoef, opt* dopt, misc* dmisc)
  {
    free(ddata); free(dcoef); free(dopt); free(dmisc);
  }

  /*
    MISC MATH FUNCTIONS
  */

  void device_ptrCopy(thrust::device_ptr<float> from,
                      thrust::device_ptr<float> to,
                      int size)
  {
    cudaMemcpy(thrust::raw_pointer_cast(to), thrust::raw_pointer_cast(from),
               sizeof(float) * size, cudaMemcpyDeviceToDevice);
  }

  // ||x||_max
  float device_ptrMaxNorm(thrust::device_ptr<float> x, int size)
  {
    return thrust::transform_reduce(x, x + size,
                                    absolute_value(), (float) 0, thrust::maximum<float>());  
  }

  // ||x||_2^2
  void device_ptr2Norm(thrust::device_ptr<float> x, float* result, int size,
                       cublasStatus_t stat, cublasHandle_t handle)
  {  
    //cublasSnrm2(handle, size, thrust::raw_pointer_cast(x), 1, result);
    //cudaThreadSynchronize();
    *result = sqrt(thrust::transform_reduce(x, x+size,
                   square(), (float) 0, thrust::plus<float>()));
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS snrm2 failed with error %i\n", stat);
    }
  }

  //result = <x,y>
  void device_ptrDot(thrust::device_ptr<float> x, thrust::device_ptr<float> y,
                     float* result, int size,
                     cublasStatus_t stat, cublasHandle_t handle)
  {  
    cublasSdot(handle, size, thrust::raw_pointer_cast(x), 1,
               thrust::raw_pointer_cast(y), 1, result);
    cudaThreadSynchronize();
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS sdot failed with error %i\n", stat);
    }
  }

  // b = X^T y
  void device_ptrCrossProd(thrust::device_ptr<float> X,
                           thrust::device_ptr<float> y,
                           thrust::device_ptr<float> b,
                           int n, int p,
                           cublasStatus_t stat, cublasHandle_t handle)
  {
    float alpha = 1; float beta = 0;
    stat = cublasSgemv(handle, CUBLAS_OP_T, n, p, &alpha,
                thrust::raw_pointer_cast(X), n,
                thrust::raw_pointer_cast(y), 1,
                &beta, thrust::raw_pointer_cast(b), 1);
    cudaThreadSynchronize();
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CrossProd using CUBLAS sgemv failed with error %i\n", stat);
    }
  }

  // b = Ax
  void device_ptrSgemv(thrust::device_ptr<float> A,
                       thrust::device_ptr<float> x,
                       thrust::device_ptr<float> b,
                       int n, int p,
                       cublasStatus_t stat, cublasHandle_t handle)
  {
    float alpha = 1; float beta = 0;
    stat = cublasSgemv(handle, CUBLAS_OP_N, n, p, &alpha,
                       thrust::raw_pointer_cast(A), n,
                       thrust::raw_pointer_cast(x), 1,
                       &beta, thrust::raw_pointer_cast(b), 1);
    cudaThreadSynchronize();
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS sgemv failed with error %i\n", stat);
    }
  }

  // S(x, lambda)
  void device_ptrSoftThreshold(thrust::device_ptr<float> x,
                               thrust::device_ptr<float> dest,
                               float lambda, int size)
  {
    thrust::transform(x, x + size,
                      dest, soft_threshold(lambda));
    cudaThreadSynchronize();
  }
  
extern "C"{

  /*
    Entry point for R
    X is a matrix (represented as a 1d array) that is n by p
    y is a vector that is n by 1
  */
  void activePathSol(float* X, float* y, int* n, int* p, float* lambda, int* num_lambda,
                     int* type, float* beta, int* maxIt, float* thresh, float* gamma,
                     float* t, int* reset)
  { 
    //setup timers
    clock_t starttime = clock();
    clock_t endtime;

    //setup pointers
    data* ddata = (data*)malloc(sizeof(data));
    coef* dcoef = (coef*)malloc(sizeof(coef));
    opt* dopt = (opt*)malloc(sizeof(opt));
    misc* dmisc = (misc*)malloc(sizeof(misc));

    /* Set key data variables X, y, beta */

    thrust::device_vector<float> dX(X, X+(n[0]*p[0]));
    thrust::device_vector<float> dy(y, y+n[0]);
    thrust::device_vector<float> dbeta(p[0]);

    ddata->X = dX.data();
    ddata->y = dy.data();
    dcoef->beta = dbeta.data();

    if (TIME) {
      endtime = clock() - starttime;
      starttime = clock();
      printf ("device memcpy took %d clicks (%f seconds).\n",
              (int)endtime,((float)endtime)/CLOCKS_PER_SEC);
    }

    /* Set coef variables */

    thrust::device_vector<float> dtheta(p[0]);
    thrust::device_vector<float> dtheta_old(p[0]);
    thrust::device_vector<float> dmomentum(p[0]);
    dcoef->theta = dtheta.data();
    dcoef->theta_old = dtheta_old.data();
    dcoef->momentum = dmomentum.data();

    /* Set optimization variables */

    thrust::device_vector<float> deta(n[0]);
    thrust::device_vector<float> dyhat(n[0]);
    thrust::device_vector<float> dresiduals(n[0]);
    thrust::device_vector<float> dgrad(p[0]);
    thrust::device_vector<float> dU(p[0]);
    thrust::device_vector<float> ddiff(p[0]);
    dopt->eta = deta.data();
    dopt->yhat = dyhat.data();
    dopt->residuals = dresiduals.data();
    dopt->grad = dgrad.data();
    dopt->U = dU.data();
    dopt->diff = ddiff.data();

    //allocate pointers
    init(ddata, dcoef, dopt, dmisc,
         X, y, *n, *p, lambda, *num_lambda,
         *type, beta, *maxIt, *thresh, *gamma,
         *t, *reset);
   
    //Set cublas variables
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    //solve
    device_ptrSgemv(ddata->X, dcoef->beta, dopt->yhat, ddata->n, ddata->p, stat, handle);
    thrust::transform(ddata->y, ddata->y + ddata->n,
                      dopt->yhat,
                      dopt->residuals,
                      thrust::minus<float>());
    pathSol(ddata, dcoef, dopt, dmisc, beta, stat, handle);
    //shutdown
    shutdown(ddata, dcoef, dopt, dmisc);
    cublasDestroy(handle);

    if (TIME) {
      endtime = clock() - starttime;
      printf ("cudaglmnet took %d clicks (%f seconds).\n",
              (int)endtime,((float)endtime)/CLOCKS_PER_SEC);
    }
  }

}

int main() {
  int* n = (int*)malloc(sizeof(int)); n[0] = 1000;
  int* p = (int*)malloc(sizeof(int)); p[0] = 10;
  int* num_lambda = (int*)malloc(sizeof(int)); num_lambda[0] = 10;
  int i = 0;
 
  thrust::host_vector<float> X(n[0]*p[0]);
  thrust::host_vector<float> y(n[0]);
  thrust::host_vector<float> beta(p[0] * num_lambda[0]);
  thrust::sequence(X.begin(), X.end());
  thrust::sequence(y.begin(), y.end());
  
  int* type = (int*)malloc(sizeof(int)); type[0] = 0;
  int* maxIt = (int*)malloc(sizeof(int)); maxIt[0] = 100;
  int* reset = (int*)malloc(sizeof(int)); reset[0] = 30;
  float* lambda = (float*)malloc(sizeof(float) * num_lambda[0]);
  for(i = 0; i < num_lambda[0]; i++) lambda[i] = i;
  float* thresh = (float*)malloc(sizeof(float)); thresh[0] = 0.00001;
  float* gamma = (float*)malloc(sizeof(float)); gamma[0] = 0.5;
  float* t = (float*)malloc(sizeof(float)); t[0] = 5;

  activePathSol(thrust::raw_pointer_cast(&X[0]),
                thrust::raw_pointer_cast(&y[0]),
                n, p, lambda, num_lambda,
                type, thrust::raw_pointer_cast(&beta[0]), maxIt, thresh, gamma,
                t, reset);
  if (DEBUG) { for(i = 0; i < beta.size(); i++) printf("beta[%i]: %f\n", i, beta[i]); }
  free(n); free(p); free(num_lambda);
  free(type); free(maxIt); free(reset);
  free(lambda); free(thresh); free(gamma); free(t);
  return 0;
}
