#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <iostream>
#include <time.h>

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
#define min_t 1e-9

typedef struct {
  int n,p,num_lambda;
  float* lambda;
  thrust::device_ptr<float> X, y;

  // Cox stuff
  int *s_person_ind, *e_person_ind, *status, *s_risk_set, *e_risk_set, *nSet;
  //end of Cox stuff

} data;

typedef struct {
    thrust::device_ptr<float> beta, beta_old, theta, theta_old, momentum;
} coef;

typedef struct {
  float nLL;
  float *U_host;
  thrust::device_ptr<float> eta, yhat, residuals, residuals_p, grad, U, diff_beta, diff, ll_temp, norm_multiplier;

  //Cox stuff
  float *eta_host, *cumInvCumRisk_host, *resids_host;
  double *cumRisk_host;
  //end of Cox stuff

} opt;

typedef struct {
  int type, maxIt, reset, backtracking, type_pen, num_groups;
  int *begins, *ends;
  float gamma, t, t_init, thresh, alpha;
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

struct log_one_minus_p
{
  __host__ __device__
  float operator()(const float& x) const { 
    return -log(1 + exp(x));
        }
};

struct inv_logit
{
    __host__ __device__
        float operator()(const float& x) const { 
            return exp(x)/(1+exp(x));
        }
};

struct log_one_minus
{
    __host__ __device__
        float operator()(const float& x) const { 
            return log(1-x);
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
          float, int, int, float, int, int*, int*);
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
float device_ptr2Norm(thrust::device_ptr<float>, int);
float device_ptrLLlogit(thrust::device_ptr<float>,
			thrust::device_ptr<float>, int);
float device_ptrDot(thrust::device_ptr<float>, thrust::device_ptr<float>, int);
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
            float t, int reset, int backtracking, int type_pen, float alpha, int num_groups, int* begins, int* ends)
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
    dmisc->t_init = t;
    dmisc->reset = reset;
    dmisc->backtracking = backtracking;
    dmisc->thresh = thresh;
    dmisc->type_pen = type_pen;
    dmisc->alpha = alpha;
    dmisc->num_groups = num_groups;
    dmisc->begins = begins;
    dmisc->ends = ends;
  }

  void pathSol(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, float* beta,
               cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside pathSol\n");
    int j = 0;
    for (j=0; j < ddata->num_lambda; j++){
    
      dmisc->t = dmisc->t_init;
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
      float nll = calcNegLL(ddata, dcoef, dopt, dmisc, dcoef->beta, j, stat, handle);
      dopt->nLL = nll;
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
    
    device_ptrSgemv(ddata->X, pvector, dopt->eta, ddata->n, ddata->p, stat, handle); // Calculating X\beta or X\theta

    switch (dmisc->type)
      {
      case 0:  //normal
	{
	  dopt->yhat = dopt->eta;
	  //residuals_p = y - yhat
	  thrust::transform(ddata->y, ddata->y + ddata->n,
			    dopt->yhat,
			    dopt->residuals_p,
			    thrust::minus<float>());
	  cudaThreadSynchronize();
	  
	  return 0.5 * device_ptr2Norm(dopt->residuals_p, ddata->n);
	}
	
      case 1: //logit
	{
	  thrust::transform(dopt->eta, dopt->eta + ddata->n,
			    dopt->yhat,
			    inv_logit()); //Calculating yhat
	  cudaThreadSynchronize();
	  thrust::transform(ddata->y, ddata->y + ddata->n,
			    dopt->yhat,
			    dopt->residuals_p,
			    thrust::minus<float>()); //Calculating residuals for later
	  cudaThreadSynchronize();
	  
	  return device_ptrLLlogit(dopt->eta, ddata->y, ddata->n);
	}

      case 2: //Cox
	{ //NEW VARS dopt->eta_host, dopt->cumRisk_host, ll, dopt->cumInvCumRisk_host, dopt->resids_host
	  //copy data over to cpu eta
	  //cumulative sum of exp(eta) based on person_ind and entry_exit for denominators
	  //sum of eta for all deaths - sum log(cumulative sum) at all risk_set_end_ind
	  //calculate residuals for later!
	  int i; float ll;
	  thrust::copy(dopt->eta, dopt->eta + ddata->n, dopt->eta_host);

	  // Calculating cumulative risk for each risk set
	  memset(dopt->cumRisk_host, 0, ddata->nSet[0]*sizeof(double));
	  for(i = 0; i < ddata->n; i++){
	    dopt->cumRisk_host[ddata->s_risk_set[i]] = dopt->cumRisk_host[ddata->s_risk_set[i]] +  (double)(exp(dopt->eta_host[ddata->s_person_ind[i]]));
	    if(ddata->e_risk_set[i] < ddata->nSet[0] - 1){
	      dopt->cumRisk_host[ddata->e_risk_set[i] + 1] = dopt->cumRisk_host[ddata->e_risk_set[i] + 1] - (double)(exp(dopt->eta_host[ddata->e_person_ind[i]]));
	    }
	  }

	  for(i = 1; i < ddata->nSet[0]; i++){
	    dopt->cumRisk_host[i] = dopt->cumRisk_host[i] + dopt->cumRisk_host[i-1];
	  }

	  // Calculating (neg) log likelihood from that
	  ll = 0;
	  for(i = 0; i < ddata->n; i++){
	    if(ddata->status[ddata->e_person_ind[i]] == 1){
	      ll = ll - (float)(dopt->eta_host[ddata->e_person_ind[i]]) + (float)log(dopt->cumRisk_host[ddata->e_risk_set[i]]);
	    }
	  }

	  // Calculating residuals (bit of a pain in the butt)
	  dopt->cumInvCumRisk_host[0] = 1/dopt->cumRisk_host[0];
	  for(i = 1; i < ddata->nSet[0]; i++){
	    dopt->cumInvCumRisk_host[i] = dopt->cumInvCumRisk_host[i-1] + 1/dopt->cumRisk_host[i];
	  }

	  memset(dopt->resids_host, 0, ddata->n*sizeof(float));
	  for(i = 0; i < ddata->n; i++){
	    dopt->resids_host[ddata->e_person_ind[i]] = dopt->resids_host[ddata->e_person_ind[i]] + dopt->cumInvCumRisk_host[ddata->e_risk_set[i]] * exp(dopt->eta_host[ddata->e_person_ind[i]]);
	    if(ddata->s_risk_set[i] > 0){
	      dopt->resids_host[ddata->s_person_ind[i]] = dopt->resids_host[ddata->s_person_ind[i]] - dopt->cumInvCumRisk_host[ddata->s_risk_set[i] - 1] * exp(dopt->eta_host[ddata->s_person_ind[i]]);
	    }
	  }

	  for(i = 0; i < ddata->n; i++){
	    dopt->resids_host[i] = ddata->status[i] - dopt->resids_host[i];
	  }
	  //copying residuals back over
	  thrust::copy(dopt->resids_host, dopt->resids_host + ddata->n, dopt->residuals_p);

	  return(ll);
	}


	

      default:  //default to normal
      { 
        return 0.5 * device_ptr2Norm(dopt->residuals_p, ddata->n);
      }
    }
  }

  /*
    yhat is already computed as eta in the previous calcNegLL call
    residuals are computed as residuals_p in the previous calcNegLL call
  */
  void computeFit(data* ddata, coef* dcoef, opt* dopt, misc* dmisc,
                  cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside computeFit\n");

        //grad = X^T residuals

        device_ptrCrossProd(ddata->X, dopt->residuals_p, dopt->grad,
                            ddata->n, ddata->p, stat, handle);
 
  }

  void gradStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside gradStep\n");

        //U = t * grad + beta
       thrust::transform(dopt->grad, dopt->grad + ddata->p, 
                          dcoef->beta,
                          dopt->U,
                          dmisc->t * thrust::placeholders::_1 + thrust::placeholders::_2);

        cudaThreadSynchronize();
        proxCalc(ddata, dcoef, dopt, dmisc, j, stat, handle);

  }

  void proxCalc(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside proxCalc\n");
     switch (dmisc->type_pen)
      {
      case 0:  //lasso
	{
	  device_ptrSoftThreshold(dopt->U, dcoef->theta, ddata->lambda[j] * dmisc->t, ddata->p);
	  break;
	}
      case 1: //holder_p
	{
	  device_ptrSoftThreshold(dopt->U, dopt->U, ddata->lambda[j] * dmisc->t * dmisc->alpha, ddata->p);

	  // Thrust transform square
	  // Thrust reduce_by_key sum
	  // Thrust transform sqrtroot
	  // Thrust transform (1 - (1-alpha)*t*lambda/norms)_+
	  // Thrst Scatter
	  // Thrust multiply


	  int i;
	  float norms, norm_multiplier;
	  for(i = 0; i < dmisc->num_groups; i++){
	    norms = sqrt(thrust::transform_reduce(dopt->U + dmisc->begins[i], dopt->U + dmisc->ends[i]+1,
						  square(), (float) 0, thrust::plus<float>()));                  //Calculate norm_multiplier
	    if(norms == 0)norm_multiplier = 0;
	    else if(norms > (1 - dmisc->alpha) * ddata->lambda[j] * dmisc->t) 
	      norm_multiplier = 1 - (1 - dmisc->alpha) * ddata->lambda[j] * dmisc->t/norms;
	    else norm_multiplier = 0;

	    thrust::fill(dopt->norm_multiplier + dmisc->begins[i], dopt->norm_multiplier + dmisc->ends[i]+1, norm_multiplier);
	  }
	 
	  thrust::transform(dopt->norm_multiplier, dopt->norm_multiplier + ddata->p, dopt->U, dcoef->theta, thrust::multiplies<float>());	   
	  break;
	}
      case 2: 	  // ON CPU! U_host, grp_norms
	{
	  device_ptrSoftThreshold(dopt->U, dopt->U, ddata->lambda[j] * dmisc->t * dmisc->alpha, ddata->p);
	  int i,k;
	  float norm_multiplier, grp_norms;
	  thrust::copy (dopt->U, dopt->U + ddata->p, dopt->U_host);
	  
	  for(i = 0; i < dmisc->num_groups; i++){
	    grp_norms = 0;
	    for(k = dmisc->begins[i]; k <= dmisc->ends[i]; k++){
	      grp_norms = grp_norms + dopt->U_host[k] * dopt->U_host[k];
	    }
	    grp_norms = sqrt(grp_norms);
	    if(grp_norms == 0) norm_multiplier = 0;
	    else if(grp_norms > (1 - dmisc->alpha) * ddata->lambda[j] * dmisc->t) 
	      norm_multiplier = 1 - (1 - dmisc->alpha) * ddata->lambda[j] * dmisc->t / grp_norms;
	    else norm_multiplier = 0;
	    
	    for(k = dmisc->begins[i]; k <= dmisc->ends[i]; k++){
	      dopt->U_host[k] = dopt->U_host[k] * norm_multiplier;
	    }
	  }
	  thrust::copy(dopt->U_host, dopt->U_host + ddata->p, dcoef->theta);
	  break;
	}
      }
  }


  int checkStep(data* ddata, coef* dcoef, opt* dopt, misc* dmisc, int j,
                cublasStatus_t stat, cublasHandle_t handle)
  {
    if (DEBUG) printf("Inside checkStep\n");

    float nLL = calcNegLL(ddata, dcoef, dopt, dmisc, dcoef->theta, j, stat, handle);
    if (DEBUG) printf("nLL with theta %f\n", nLL);
    if (DEBUG) printf("nLL with beta %f\n", dopt->nLL);
    
    //diff = beta-theta
    thrust::transform(dcoef->beta, dcoef->beta + ddata->p,
		      dcoef->theta,
		      dopt->diff,
		      thrust::minus<float>());
    cudaThreadSynchronize();
    //iprod is the dot product of diff and grad
    float iprod = device_ptrDot(dopt->diff, dopt->grad, ddata->p);
    float sumSquareDiff = device_ptr2Norm(dopt->diff, ddata->p);
    
    int check = (int)(nLL <= (dopt->nLL + iprod + (sumSquareDiff / (2 * dmisc->t))));
    if(dmisc->t <= min_t){ check = 1;}

    if (check == 0) dmisc->t = dmisc->t * dmisc->gamma;
 
    if(dmisc->backtracking == 1){ return check; }
    else{return 1;}
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
    //freeing cox stuff
    free(dopt->cumRisk_host); free(dopt->cumInvCumRisk_host); free(dopt->resids_host); free(dopt->eta_host); 

    free(dopt->U_host);  free(ddata); free(dcoef); free(dopt); free(dmisc);
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
    float device_ptr2Norm(thrust::device_ptr<float> x, int size)
  {  
    return thrust::transform_reduce(x, x+size,
                                    square(), (float) 0, thrust::plus<float>());
  }

  float device_ptrLLlogit(thrust::device_ptr<float> x, thrust::device_ptr<float> y, int size)
  {  
    return -(thrust::transform_reduce(x, x + size, log_one_minus_p(), (float) 0, thrust::plus<float>()) + thrust::inner_product(x, x + size, y, (float) 0));
  }


  //result = <x,y>
  float device_ptrDot(thrust::device_ptr<float> x, thrust::device_ptr<float> y,
                      int size)
  {  
    return thrust::inner_product(x, x+size, y, (float) 0);
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
                     float* t, int* reset, int* backtracking, int* type_pen, float* alpha,
		     int* num_groups, int* begins, int* ends, 

		     //Cox Model Stuff
		     int* s_person_ind, int* e_person_ind, int* status,
		     int* s_risk_set, int* e_risk_set, int* nSet
		     //end of Cox stuff

		     )
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
    thrust::device_vector<float> dbeta(p[0], 0);

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
    thrust::device_vector<float> dresiduals_p(n[0]);
    thrust::device_vector<float> dgrad(p[0]);
    thrust::device_vector<float> dU(p[0]);
    thrust::device_vector<float> ddiff(p[0]);
    thrust::device_vector<float> dll_temp(n[0]);
    thrust::device_vector<float> dnorm_multiplier(p[0]);
    float* U_host = (float*)malloc(sizeof(float) * p[0]);

    dopt->eta = deta.data();
    dopt->yhat = dyhat.data();
    dopt->residuals = dresiduals.data();
    dopt->residuals_p = dresiduals_p.data();
    dopt->grad = dgrad.data();
    dopt->U = dU.data();
    dopt->diff = ddiff.data();
    dopt->ll_temp = dll_temp.data();                    
    dopt->norm_multiplier = dnorm_multiplier.data();
    dopt->U_host = U_host;

    //setting up cox stuff
    ddata->s_person_ind = s_person_ind;
    ddata->e_person_ind = e_person_ind;
    ddata->status = status;
    ddata->s_risk_set = s_risk_set;
    ddata->e_risk_set = e_risk_set;
    ddata->nSet = nSet;
    float* eta_host = (float*)malloc(sizeof(float) * n[0]);
    float* resids_host = (float*)malloc(sizeof(float) * n[0]);
    double* cumRisk_host = (double*)malloc(sizeof(double) * nSet[0]);
    float* cumInvCumRisk_host = (float*)malloc(sizeof(float) * nSet[0]);
    dopt->eta_host = eta_host;
    dopt->resids_host = resids_host;
    dopt->cumRisk_host = cumRisk_host;
    dopt->cumInvCumRisk_host = cumInvCumRisk_host;
    //done with cox


    //allocate pointers
    init(ddata, dcoef, dopt, dmisc,
         X, y, *n, *p, lambda, *num_lambda,
         *type, beta, *maxIt, *thresh, *gamma,
         *t, *reset, *backtracking, *type_pen, *alpha, *num_groups, begins, ends);
   
    //Set cublas variables
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    //solve
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

// Stuff for group lasso

  void set_inds(int* index, int* p, int* begins, int* ends)
  { 
    int i, group;
    begins[0] = 0;
    group = 0;
    for(i = 1; i < p[0]; i++)
      {
	if(index[i] != index[i-1])
	  {
	    ends[group] = i-1;
	    group++;
	    begins[group] = i;
	  }
      }
    ends[group] = p[0] - 1;
  }


}

int main() {
  int* n = (int*)malloc(sizeof(int)); n[0] = 100;
  int* p = (int*)malloc(sizeof(int)); p[0] = 10;
  int* num_lambda = (int*)malloc(sizeof(int)); num_lambda[0] = 2;
  int* backtracking = (int*)malloc(sizeof(int)); backtracking[0] = 1;
  int i = 0;
 
  thrust::host_vector<float> X(n[0]*p[0]);
  thrust::host_vector<float> y(n[0]);
  thrust::host_vector<float> beta(p[0] * num_lambda[0]);
  thrust::sequence(X.begin(), X.end());
  thrust::sequence(y.begin(), y.end());
  
  int* type = (int*)malloc(sizeof(int)); type[0] = 0;
  int* maxIt = (int*)malloc(sizeof(int)); maxIt[0] = 1500;
  int* reset = (int*)malloc(sizeof(int)); reset[0] = 30;
  float* lambda = (float*)malloc(sizeof(float) * num_lambda[0]);
  for(i = 0; i < num_lambda[0]; i++) lambda[i] = i + 1;
  float* thresh = (float*)malloc(sizeof(float)); thresh[0] = 0.00001;
  float* gamma = (float*)malloc(sizeof(float)); gamma[0] = 0.9;
  float* t = (float*)malloc(sizeof(float)); t[0] = 5;
  int *type_pen =(int*)malloc(sizeof(int)); type_pen[0] = 0;
  float *alpha =(float*)malloc(sizeof(float)); alpha[0] = 0;
  int *num_groups =(int*)malloc(sizeof(int)); num_groups[0] = 1;
  int *begins =(int*)malloc(sizeof(int)); begins[0] = 0;
  int *ends =(int*)malloc(sizeof(int)); ends[0] = 0;
  
  int* s_person_ind = (int*)malloc(sizeof(int)); s_person_ind[0] = 0;
  int* e_person_ind = (int*)malloc(sizeof(int)); e_person_ind[0] = 0;
  int* status = (int*)malloc(sizeof(int)); status[0] = 0;
  int* s_risk_set = (int*)malloc(sizeof(int)); s_risk_set[0] = 0;
  int* e_risk_set = (int*)malloc(sizeof(int)); e_risk_set[0] = 0;
  int* nSet = (int*)malloc(sizeof(int)); nSet[0] = 0;

  // STUFF

  activePathSol(thrust::raw_pointer_cast(&X[0]),
                thrust::raw_pointer_cast(&y[0]),
                n, p, lambda, num_lambda,
                type, thrust::raw_pointer_cast(&beta[0]), maxIt, thresh, gamma,
                t, reset, backtracking, type_pen, alpha, num_groups, begins,
		ends, s_person_ind, e_person_ind, status, s_risk_set, e_risk_set, nSet);
  if (DEBUG) { for(i = 0; i < beta.size(); i++) printf("beta[%i]: %f\n", i, beta[i]); }
  free(n); free(p); free(num_lambda);
  free(type); free(maxIt); free(reset);
  free(lambda); free(thresh); free(gamma); free(t);
  return 0;
}

