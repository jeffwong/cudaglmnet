#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>

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
void pathSol(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, float* beta,cublasStatus_t stat, cublasHandle_t handle );
void singleSolve(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,cublasStatus_t stat, cublasHandle_t handle );
float calcNegLL(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, thrust::device_ptr<float> pvector, int j,cublasStatus_t stat, cublasHandle_t handle );
void gradStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,cublasStatus_t stat, cublasHandle_t handle );
void proxCalc(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,cublasStatus_t stat, cublasHandle_t handle );
void nestStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter,cublasStatus_t stat, cublasHandle_t handle );
int checkStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,cublasStatus_t stat, cublasHandle_t handle );
int checkCrit(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter,cublasStatus_t stat, cublasHandle_t handle );
void shutdown(data* ddata, coef* dcoef, opt* dopt, misc* dmisc);
void device_ptr2Norm(thrust::device_ptr<float> x, float* result, int size, cublasStatus_t stat, cublasHandle_t handle );
void device_ptrDot(thrust::device_ptr<float> x, thrust::device_ptr<float> y,
                   float* result, int size, cublasStatus_t stat, cublasHandle_t handle );
float device_ptrMaxNorm(thrust::device_ptr<float> x, int size);
void device_ptrSoftThreshold(thrust::device_ptr<float> x, thrust::device_ptr<float>, float lambda, int size);
void device_ptrSgemv(thrust::device_ptr<float> A,
                          thrust::device_ptr<float> x,
                          thrust::device_ptr<float> b,
                          int n, int p,
                          cublasStatus_t stat, cublasHandle_t handle );
void device_ptrCrossProd(thrust::device_ptr<float> X,
                              thrust::device_ptr<float> y,
                              thrust::device_ptr<float> b,
                              int n, int p,
                            cublasStatus_t stat, cublasHandle_t handle) ;
thrust::device_ptr<float> makeDeviceVector(float* x, int size);
thrust::device_ptr<float> makeEmptyDeviceVector(int size);
void device_ptrCopy(thrust::device_ptr<float> from,
                    thrust::device_ptr<float> to,
                    int size);



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

    //allocate pointers
    init(ddata, dcoef, dopt, dmisc,
         X, y, *n, *p, lambda, *num_lambda,
         *type, beta, *maxIt, *thresh, *gamma,
         *t, *reset);
   
    //Set cublas variables
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      cublasDestroy(handle);
      return;
    }
 
    //solve
    //pathSol(ddata, dcoef, dopt, dmisc, beta, stat, handle);

    //shutdown
    shutdown(ddata, dcoef, dopt, dmisc);
    cublasDestroy(handle);
  }

  void init(data* ddata, coef* dcoef, opt* dopt, misc* dmisc,
            float* X, float* y, int n, int p, float* lambda, int num_lambda,
            int type, float* beta, int maxIt, float thresh, float gamma,
            float t, int reset)
  {
    /* Set data variables */
    ddata->X = makeDeviceVector(X, n*p);
    ddata->y = makeDeviceVector(y, n);
    ddata->lambda = thrust::host_vector<float>(lambda, lambda+num_lambda);
    ddata->n = n;
    ddata->p = p;
    ddata->num_lambda = num_lambda;

    /* Set coef variables */

    dcoef->beta = makeDeviceVector(beta, p);
    dcoef->beta_old = makeDeviceVector(beta, p);
    dcoef->theta = makeEmptyDeviceVector(p);
    dcoef->theta_old = makeEmptyDeviceVector(p);
    dcoef->momentum = makeEmptyDeviceVector(p);

    /* Set optimization variables */

    dopt->nLL = 0;
    dopt->eta = makeEmptyDeviceVector(n);
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

  void pathSol(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, float* beta,
               cublasStatus_t stat, cublasHandle_t handle)
  {
    int j = 0;
    for (j=0; j < ddata->num_lambda; j++){
      device_ptrCopy(dcoef->beta, dcoef->beta_old, ddata->p);
      device_ptrCopy(dcoef->theta, dcoef->theta_old, ddata->p);
      //singleSolve(ddata, dcoef, dopt, dmisc, j, stat, handle);

      int startIndex = j*ddata->p;
      //thrust::copy(dcoef->beta, dcoef->beta + ddata->p, beta + startIndex);
    }
  }

  void singleSolve(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                   cublasStatus_t stat, cublasHandle_t handle)
  {
    int iter = 0;
    do
    {
      printf("calcNegLL\n");
      //calcNegLL(ddata, dcoef, dopt, dmisc, dcoef->beta, j, stat, handle);
      /*do
      {
        printf("gradStep\n");
        gradStep(ddata, dcoef, dopt, dmisc, j, stat, handle);
        printf("checkStep\n");
      } while (checkStep(ddata, dcoef, dopt, dmisc, j, stat, handle) == 0);
      printf("nestStep\n");
      nestStep(ddata, dcoef, dopt, dmisc, j, iter, stat, handle);*/
      iter = iter + 1;
      printf("checkCrit");
    } while (checkCrit(ddata, dcoef, dopt, dmisc, j, iter, stat, handle));
  }

  int checkCrit(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    float move = 1;
    //device_ptrMaxNorm(dopt->diff_theta, ddata->p);  
    if ((iter > dmisc->maxIt) || (move < dmisc->thresh)) return 0;
    else return 1;
  }

  float calcNegLL(data* ddata, coef* dcoef, opt* dopt, misc* dmisc,
                  thrust::device_ptr<float> pvector, int j,
                  cublasStatus_t stat, cublasHandle_t handle)
  {
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
    return dopt->nLL;
  }

  int checkStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    float nLL = calcNegLL(ddata, dcoef, dopt, dmisc, dcoef->theta, j, stat, handle);
    
    //iprod is the dot product of diff and grad
    float iprod=0; device_ptrDot(dopt->diff_theta, dopt->grad, &iprod, ddata->p, stat, handle);
    float sumSquareDiff=0; device_ptr2Norm(dopt->diff_theta, &sumSquareDiff, ddata->p, stat, handle);

    int check = (int)(nLL < (dopt->nLL + iprod + sumSquareDiff) / (2 * dmisc->t));
    if (check == 0) dmisc->t = dmisc->t * dmisc->gamma;
    return check;
  }

  void gradStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                cublasStatus_t stat, cublasHandle_t handle)
  {
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
        //grad = X^T residuals
        device_ptrCrossProd(ddata->X, dopt->residuals, dopt->grad, ddata->n,
                            ddata->p, stat, handle);
        //U = -t * grad + beta
        thrust::transform(dopt->grad, dopt->grad + ddata->p,
                          dcoef->beta,
                          dopt->U,
                          saxpy(-dmisc->t));
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

  void nestStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j, int iter,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    device_ptrCopy(dcoef->beta, dcoef->beta_old, ddata->p);
    //momentum = theta - theta old
    thrust::transform(dcoef->theta, dcoef->theta + ddata->p,
                      dcoef->theta_old,
                      dcoef->momentum,
                      thrust::minus<float>());
    float scale = (float) (iter % dmisc->reset) / (iter % dmisc->reset + 3);
    //beta = theta + scale*momentum
    thrust::transform(dcoef->momentum, dcoef->momentum + ddata->p,
                      dcoef->theta,
                      dcoef->beta,
                      saxpy(scale));
    device_ptrCopy(dcoef->theta, dcoef->theta_old, ddata->p);
  }

  void shutdown(data* ddata, coef* dcoef, opt* dopt, misc* dmisc)
  {
    free(ddata); free(dcoef); free(dopt); free(dmisc);
  }

  /*
    MISC MATH FUNCTIONS
  */

  thrust::device_ptr<float> makeDeviceVector(float* x, int size)
  {
    thrust::host_vector<float> h(x, x+size);
    thrust::device_vector<float> d = h;
    return d.data();
  }

  thrust::device_ptr<float> makeEmptyDeviceVector(int size)
  {
    thrust::host_vector<float> h(size,0);
    thrust::device_vector<float> d = h;
    return d.data();
  }

  void device_ptrCopy(thrust::device_ptr<float> from,
                      thrust::device_ptr<float> to,
                      int size)
  {
    /*thrust::device_vector<float> A(from, from + size);
    thrust::device_vector<float> B(to, to + size);
    B = A;*/
    to = from;
  }

  // ||x||_max
  float device_ptrMaxNorm(thrust::device_ptr<float> x, int size)
  {
    thrust::device_vector<float> d(x, x + size);
    return thrust::reduce(d.begin(), d.end(), (float) 0, thrust::plus<float>()); 
    /*return thrust::transform_reduce(d.begin(), d.end(),
                                    absolute_value(), (float) 0, thrust::maximum<float>());  
    */
  }

  // ||x||_2^2
  void device_ptr2Norm(thrust::device_ptr<float> x, float* result, int size,
                       cublasStatus_t stat, cublasHandle_t handle)
  {  
    cublasSnrm2(handle, size, thrust::raw_pointer_cast(x), 1, result);
  }

  void device_ptrDot(thrust::device_ptr<float> x, thrust::device_ptr<float> y,
                     float* result, int size,
                     cublasStatus_t stat, cublasHandle_t handle)
  {  
    cublasSdot(handle, size, thrust::raw_pointer_cast(x), 1,
               thrust::raw_pointer_cast(y), 1, result);
  }

  // b = X^T y
  void device_ptrCrossProd(thrust::device_ptr<float> X,
                           thrust::device_ptr<float> y,
                           thrust::device_ptr<float> b,
                           int n, int p,
                           cublasStatus_t stat, cublasHandle_t handle)
  {
    float alpha = 1; float beta = 0;
    cublasSgemv(handle, CUBLAS_OP_T, n, p, &alpha,
                thrust::raw_pointer_cast(X), n,
                thrust::raw_pointer_cast(y), 1,
                &beta, thrust::raw_pointer_cast(b), 1); 
  }

  // b = Ax
  void device_ptrSgemv(thrust::device_ptr<float> A,
                       thrust::device_ptr<float> x,
                       thrust::device_ptr<float> b,
                       int n, int p,
                       cublasStatus_t stat, cublasHandle_t handle)
  {
    float alpha = 1; float beta = 0;
    cublasSgemv(handle, CUBLAS_OP_N, n, p, &alpha,
                thrust::raw_pointer_cast(A), n,
                thrust::raw_pointer_cast(x), 1,
                &beta, thrust::raw_pointer_cast(b), 1);
  }

  // S(x, lambda)
  void device_ptrSoftThreshold(thrust::device_ptr<float> x,
                               thrust::device_ptr<float> dest,
                               float lambda, int size)
  {
    thrust::transform(x, x + size,
                      dest, soft_threshold(lambda));
  }

}

int main() {
  int* n = (int*)malloc(sizeof(int)); n[0] = 100;
  int* p = (int*)malloc(sizeof(int)); p[0] = 10;
  int* num_lambda = (int*)malloc(sizeof(int)); num_lambda[0] = 1;
 
  thrust::host_vector<float> X(n[0]*p[0],1);
  thrust::host_vector<float> y(n[0],1);
  thrust::host_vector<float> beta(p[0] * num_lambda[0],1);

  int* type = (int*)malloc(sizeof(int)); type[0] = 0;
  int* maxIt = (int*)malloc(sizeof(int)); maxIt[0] = 10;
  int* reset = (int*)malloc(sizeof(int)); reset[0] = 5;
  float* lambda = (float*)malloc(sizeof(float) * num_lambda[0]); lambda[0] = 1;
  float* thresh = (float*)malloc(sizeof(float)); thresh[0] = 0.00001;
  float* gamma = (float*)malloc(sizeof(float)); gamma[0] = 0.9;
  float* t = (float*)malloc(sizeof(float)); t[0] = 10;

  activePathSol(thrust::raw_pointer_cast(&X[0]),
                thrust::raw_pointer_cast(&y[0]),
                n, p, lambda, num_lambda,
                type, thrust::raw_pointer_cast(&beta[0]), maxIt, thresh, gamma,
                t, reset);
  
  free(n); free(p); free(num_lambda);
  free(type); free(maxIt); free(reset);
  free(lambda); free(thresh); free(gamma); free(t);
  return 0;
}
