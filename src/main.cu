#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <cuda.h>
#define index(i,j,ld) (((j)*(ld))+(i))

/*
Copies parts of matrix gpu_X into sub_X
*/
__global__ void copySubmatrix(float *gpu_X, float *sub_X, int *gpu_indices,
                              int length_ind, int n, int p)
{
  int k = threadIdx.x + blockDim.x*blockIdx.x;
  if (k < n*length_ind) {
    int j = (k - 1) / n;
    int i = k - n * j;    
    sub_X[j * n + i] = gpu_X[gpu_indices[j] * n + i];
  }
}

/*
Copies parts of a vector gpu_beta into gpu_Abeta
*/
__global__ void copySubBeta(float *gpu_beta, float *gpu_Abeta, int *gpu_indices,
                            int length_ind)
{
  int k = threadIdx.x + blockDim.x*blockIdx.x;
  if (k < length_ind) {    
    gpu_Abeta[k] = gpu_beta[gpu_indices[k]];
  }
}

/*
Copies parts of a vector gpu_Abeta into gpu_beta
*/
__global__ void copyunSubBeta(float *gpu_beta, float *gpu_Abeta, int *gpu_indices,
                              int length_ind)
  
{
  int k = threadIdx.x + blockDim.x*blockIdx.x;
  if (k < length_ind) {    
    gpu_beta[gpu_indices[k]] = gpu_Abeta[k];
  }
}

/*
Finds components of the grad that have absolute value > lambda
saves into gpu_isActive
*/
__global__ void checkKKT(float *gpu_grad, int *gpu_isActive, float lambda, int p)

{
  int k = threadIdx.x + blockDim.x*blockIdx.x;
  if (k < p){
    if ((gpu_grad[k] < -lambda) || (gpu_grad[k] > lambda)) gpu_isActive[k] = 1;
  }
}

//Extract ind-th element of gpu_vector
//This probably should not be a kernel itself
__global__ void getKernel (float *gpu_vector, int ind, float *gpu_val)

{
  gpu_val[0] = gpu_vector[ind];
}

//Probably should not be a kernel itself
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
 
  //copies part of gpu_X into sub_X
  void subMatrix(float *gpu_X, float *sub_X, int *gpu_indices, int length_ind, int n, int p){
    int block_size = 256;
    int n_blocks = n*length_ind/block_size + ((n*length_ind)%block_size == 0 ? 0:1);
    
    copySubmatrix <<< block_size, n_blocks >>> (gpu_X, sub_X, gpu_indices, length_ind, n, p);
  }
  
  void subBeta(float *gpu_beta, float *gpu_Abeta, int *gpu_indices, int length_ind){
    int block_size = 256;
    int n_blocks = length_ind/block_size + ((length_ind)%block_size == 0 ? 0:1);
    
    copySubBeta <<< block_size, n_blocks >>> (gpu_beta, gpu_Abeta, gpu_indices, length_ind);
  }

  void unsubBeta(float *gpu_beta, float *gpu_Abeta, int *gpu_indices, int length_ind){
    int block_size = 256;
    int n_blocks = length_ind/block_size + ((length_ind)%block_size == 0 ? 0:1);
    
    copyunSubBeta <<< block_size, n_blocks >>> (gpu_beta, gpu_Abeta, gpu_indices, length_ind);
  }

  void softThreshold(float *gpu_beta, float lambda, float step, int p){
    int block_size = 256;
    int n_blocks = p/block_size + ((p)%block_size == 0 ? 0:1);
    
    softKernel <<< block_size, n_blocks >>> (gpu_beta, lambda*step, p);
  }

  //transfers gpu_vector[ind] into returnPtr
  void getIndVal(float *gpu_vector, int ind, float *returnPtr){
    int block_size = 1;
    int n_blocks = 1;
    float *gpu_val;
    cudaMalloc((void**) &gpu_val, sizeof(float));
    getKernel <<< block_size, n_blocks >>> (gpu_vector, ind, gpu_val);
    cudaMemcpy(returnPtr, gpu_val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(gpu_val);
  }

  void checkStep(float *gpu_X, float *gpu_resid, float *gpu_grad, int* gpu_indices,
                 int* indices, float lambda, int *cont, int *gpu_isActive, int *isActive,
                 int *numActive, int *gpu_numActive, int *n, int *p){
    /* 
       Calculating new grad
       computes X^T (residuals) and saves it into gpu_grad
    */
    cublasSgemv('t', n[0], p[0], 1, gpu_X, n[0], gpu_resid, 1, 0, gpu_grad, 1);

    /* Checking if KKT holds */
    int block_size = 256;
    int n_blocks = p[0]/block_size + ((p[0])%block_size == 0 ? 0:1);
    checkKKT <<< block_size, n_blocks >>> (gpu_grad, gpu_isActive, lambda, p[0]);
 
    int oldNumActive = numActive[0];
    numActive[0] = 0;

    cudaMemcpy(isActive, gpu_isActive, sizeof(int)*p[0], cudaMemcpyDeviceToHost);

    int counter = 0;
    for (int i=0; i<p[0];i++){
      if (isActive[i] != 0){
	      indices[counter] = i;
	      counter++;
      }
    }
    numActive[0] = counter;

    cont[0] = 0;
    if (numActive[0] > oldNumActive) cont[0] = 1;
    cudaMemcpy(gpu_numActive, numActive, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_indices, indices, sizeof(int)*p[0],cudaMemcpyHostToDevice);
  }

  void gradStep(float *gpu_X, float *gpu_y, float *gpu_resid, float *gpu_fit,
                float *gpu_beta, float *gpu_oldBeta, float *gpu_grad, float *gpu_diff,
                float lambda, float *thresh, int *maxIt, float *step_size, float *beta,
                int *n, int *p, float *diff, float *step){
 
    float oldLL = 0;
    float newLL = 0;
    float dot_val = 0;
    int max_move_ind = 0;
    float *max_move;
    max_move = (float*)malloc(sizeof(float));
    max_move[0] = 0;

    /* Copying beta to oldBeta for backtracking */
    cublasScopy(p[0], gpu_beta, 1, gpu_oldBeta, 1);
    
    /* Calculating the new fit 
       Computes gpu_X (gpu_beta) and store into gpu_fit 
    */
    cublasSgemv('n', n[0], p[0], 1, gpu_X, n[0], gpu_beta, 1, 0, gpu_fit, 1);

    /* Calculating new residuals */
    cublasScopy(n[0], gpu_y, 1, gpu_resid, 1);  // Copying y to resid
    cublasSaxpy(n[0], -1, gpu_fit, 1, gpu_resid, 1);  // Subtracting fit from y (which is stored in resid)

     /* Calculating oldLL based on resid */

    oldLL = cublasSnrm2(n[0], gpu_resid, 1);
  
    /* Calculating new grad */
    cublasSgemv('t', n[0], p[0], 1, gpu_X, n[0], gpu_resid, 1, 0, gpu_grad, 1);

     /* Step beta in the proper direction
        Computes step_size (gpu_grad) and stores in gpu_beta 
     */
    cublasSaxpy(p[0], step_size[0], gpu_grad, 1, gpu_beta, 1);

    /* Soft-threshholding beta by lambda */
    
    softThreshold(gpu_beta, lambda, step_size[0], p[0]);
    
    /* Step size optimization */
    // Calculating RHS
    /* Calculating difference between beta and oldBeta */
    
    cublasScopy(p[0], gpu_beta, 1, gpu_diff, 1);
    cublasSaxpy(p[0], -1, gpu_oldBeta, 1, gpu_diff, 1);
    
    /* calculating the dot product between grad and diff */
    
    dot_val = cublasSdot(p[0], gpu_diff, 1, gpu_grad, 1);
    
    /* Calculating length of move */
    
    *step = cublasSnrm2(p[0], gpu_diff, 1);
    max_move_ind = cublasIsamax(p[0], gpu_diff, 1); /// Problem???
    
    getIndVal(gpu_diff, (max_move_ind-1), max_move);
    max_move[0] = max_move[0] * max_move[0];

    /* Terrible way to do this! Don't need to copy the whole vector! */
    //   cublasGetVector(p[0], sizeof(float), gpu_diff, 1, diff, 1);
    //max_move = diff[max_move_ind-1]*diff[max_move_ind-1];
    
    // Calculating LHS
    
    cublasSgemv('n', n[0], p[0], 1, gpu_X, n[0], gpu_beta, 1, 0, gpu_fit,1);
    cublasScopy(n[0], gpu_y, 1, gpu_resid, 1);
    cublasSaxpy(n[0], -1, gpu_fit, 1, gpu_resid, 1);
    newLL = cublasSnrm2(n[0], gpu_resid, 1);
    
    if(newLL*newLL/2 > oldLL*oldLL/2 - dot_val + step[0]*step[0]/(2*step_size[0])){
      cublasScopy(p[0], gpu_oldBeta, 1, gpu_beta, 1);
      step_size[0] = step_size[0] * 0.8;
      step[0] = 100000;
    }
    free(max_move);
  }

  void singleSol(float *gpu_X, float *gpu_y, float *gpu_resid, float *gpu_fit,
                 float *gpu_beta, float *gpu_oldBeta, float *gpu_grad, float *gpu_diff,
                 float lambda, float *thresh, int *maxIt, float *step_size_set, float *beta,
                 int *n, int *p, float *diff, int* gpu_isActive, int* isActive, int* numActive,
                 int* gpu_numActive, int* gpu_indices, int *indices,float* gpu_AX,
                 float* gpu_Abeta, float* gpu_AoldBeta, float* gpu_Agrad, float* gpu_Adiff,
                 float* Abeta, float* Adiff) {
    int count = 0;
    int cont = 1;
    int inner_cont = 1; // inner loop variable (for active set)
    float step = 0;
    float init_step = step_size_set[0];

    int act_p = numActive[0];

    checkStep(gpu_X, gpu_resid, gpu_grad, gpu_indices, indices, lambda, &cont,
              gpu_isActive, isActive, numActive, gpu_numActive, n, p);
   
    while (cont == 1){
      inner_cont = 1;
      /* Defining all the new active variables */
    
      subBeta(gpu_beta, gpu_Abeta, gpu_indices, numActive[0]);
   
      subMatrix(gpu_X, gpu_AX, gpu_indices, numActive[0], n[0], p[0]);    

      while (inner_cont == 1){    
        
        act_p = numActive[0];
        gradStep(gpu_AX, gpu_y, gpu_resid, gpu_fit, gpu_Abeta, gpu_AoldBeta,
                 gpu_Agrad, gpu_Adiff, lambda, thresh, maxIt, step_size_set, Abeta,
                 n, &act_p, Adiff, &step);

        /* Checking if stop criteria are satisfied */
        count++;
        if (count > maxIt[0] || step < thresh[0]) inner_cont = 0; break;
      }
    

      unsubBeta(gpu_beta, gpu_Abeta, gpu_indices, numActive[0]);

      checkStep(gpu_X, gpu_resid, gpu_grad, gpu_indices, indices, lambda,
                &cont, gpu_isActive, isActive, numActive, gpu_numActive, n, p);
      
    }
    step_size_set[0] = init_step;
  }
  
 


/*
Entry point for R
X is a matrix (represented as a 1d array) that is n by p
y is a vector that is n by 1
*/
  void activePathSol(float* X, float* y, int* n, int* p, int* maxIt, float* thresh,
                     float* step_size, float* lambda, float* beta, int* num_lambda)
  { 
    int number_of_devices;
    cudaGetDeviceCount(&number_of_devices);
    cudaSetDevice(0);

    int i,j;
    cublasStatus status;

    cublasInit();

    /* ALLOCATING HOST MEMORY */
   
    float *grad = (float*)malloc(p[0]*sizeof(float));
    float *oldBeta = (float*)malloc(p[0]*sizeof(float));
    float *workingBeta = (float*)malloc(p[0]*sizeof(float));
    float *fit = (float*)malloc(n[0]*sizeof(float));
    float *resid = (float*)malloc(n[0]*sizeof(float));
    float *diff = (float*)malloc(p[0]*sizeof(float));
    int *isActive = (int*)malloc(p[0]*sizeof(int));
    int *numActive = (int*)malloc(sizeof(int));
    int *indices = (int*)malloc(p[0]*sizeof(float)); // Ever active index

    /* INITIALIZING ARRAY VALUES */
    
    for (i=0;i<n[0];i++){
      resid[i] = y[i];
      fit[i] = 0;
    }
    for (i=0;i<p[0];i++){
      grad[i] = 0;
      oldBeta[i] = 0;
      isActive[i] = 0;
      indices[i] = -1;
    }
    numActive[0] = 0;

    /* INITIALIZING POINTERS FOR THE GPU VERSIONS OF VARIABLES */

    float* gpu_X; float* gpu_y; float* gpu_workingBeta; float* gpu_oldBeta; float* gpu_fit;
    float* gpu_resid; float* gpu_grad; float* gpu_diff;
    int* gpu_isActive; int* gpu_numActive; int* gpu_indices;

    /* ALLOCATING MEMORY ON THE GPU */

    status=cublasAlloc(n[0]*p[0],sizeof(float),(void**)&gpu_X);
    status=cublasAlloc(n[0],sizeof(float),(void**)&gpu_y);
    status=cublasAlloc(n[0],sizeof(float),(void**)&gpu_resid);
    status=cublasAlloc(n[0],sizeof(float),(void**)&gpu_fit);
    status=cublasAlloc(p[0],sizeof(float),(void**)&gpu_workingBeta);
    status=cublasAlloc(p[0],sizeof(float),(void**)&gpu_oldBeta);
    status=cublasAlloc(p[0],sizeof(float),(void**)&gpu_grad);
    status=cublasAlloc(p[0],sizeof(float),(void**)&gpu_diff);
    status=cublasAlloc(p[0],sizeof(int),(void**)&gpu_isActive);
    status=cublasAlloc(p[0], sizeof(int),(void**)&gpu_indices);
    cudaMalloc((void**) &gpu_numActive, sizeof(int));

    /* Defining submatrix/activeset stuff */

    float *Abeta = (float*)malloc(p[0]*sizeof(float));
    float *Adiff = (float*)malloc(p[0]*sizeof(float));
   
    float *gpu_AX; float *gpu_Abeta; float *gpu_AoldBeta; float *gpu_Agrad; float *gpu_Adiff;
    cublasAlloc(n[0]*p[0],sizeof(float),(void**)&gpu_AX);
    cublasAlloc(p[0], sizeof(int),(void**)&gpu_Abeta);
    cublasAlloc(p[0], sizeof(int),(void**)&gpu_AoldBeta);
    cublasAlloc(p[0], sizeof(int),(void**)&gpu_Agrad);
    cublasAlloc(p[0], sizeof(int),(void**)&gpu_Adiff);

    /* MOVING THE MATRICES OVER TO GPU MEMORY */

    status=cublasSetMatrix(n[0],p[0],sizeof(float),X,n[0],gpu_X,n[0]);
    status=cublasSetVector(n[0],sizeof(float),y,1,gpu_y,1);
    status=cublasSetVector(n[0],sizeof(float),resid,1,gpu_resid,1);
    status=cublasSetVector(n[0],sizeof(float),fit,1,gpu_fit,1);
    status=cublasSetVector(p[0],sizeof(float),oldBeta,1,gpu_workingBeta,1);
    status=cublasSetVector(p[0],sizeof(float),oldBeta,1,gpu_oldBeta,1);
    status=cublasSetVector(p[0],sizeof(float),grad,1,gpu_grad,1);
    status=cublasSetVector(p[0],sizeof(int),isActive,1,gpu_isActive,1);
    status=cublasSetVector(p[0],sizeof(int),indices,1,gpu_indices,1);
    cudaMemcpy(gpu_numActive, numActive, sizeof(int), cudaMemcpyHostToDevice);

    /* RUNNING A LOOP TO SOVLE FOR EACH LAMBDA */

    for(j=0; j < num_lambda[0]; j++){

      //solve for just one lambda
      singleSol(gpu_X, gpu_y, gpu_resid, gpu_fit, gpu_workingBeta, gpu_oldBeta, gpu_grad,
                gpu_diff, lambda[j], thresh, maxIt, step_size, workingBeta, n, p, diff,
                gpu_isActive, isActive, numActive, gpu_numActive, gpu_indices, indices,
                gpu_AX, gpu_Abeta, gpu_AoldBeta, gpu_Agrad, gpu_Adiff, Abeta, Adiff);

      //Extract the p elements of the beta vector
      cublasGetVector(p[0], sizeof(float), gpu_workingBeta, 1, workingBeta, 1);

      //reset numActive and gpu_numActive to 0 after solving for the lambda
      numActive[0] = 0;
      cudaMemcpy(gpu_numActive, numActive, sizeof(int), cudaMemcpyHostToDevice);

      //workingBeta is a vector for one particular lambda
      //Place it inside larger matrix beta
      for(i=0; i < p[0]; i++){
        beta[j*p[0]+i] = workingBeta[i];
      }
    }
    
    /* FREEING UP MEMORY */

    free ( grad ); free( fit ); free( resid ); free( oldBeta ); free( workingBeta ); free( diff ); free ( numActive ); free( Abeta ); free( Adiff ); free( indices ); free( isActive );
    status = cublasFree(gpu_X);
    status = cublasFree(gpu_y);
    status = cublasFree(gpu_grad);
    status = cublasFree(gpu_workingBeta);
    status = cublasFree(gpu_oldBeta);
    status = cublasFree(gpu_resid);
    status = cublasFree(gpu_fit);
    status = cublasFree(gpu_diff);
    status = cublasFree(gpu_isActive);
    status = cublasFree(gpu_indices);
    cudaFree(gpu_numActive);
    cublasFree(gpu_AX);
    cublasFree(gpu_Agrad);
    cublasFree(gpu_Abeta);
    cublasFree(gpu_AoldBeta);
    cublasFree(gpu_Adiff);
    
   /* Shutdown */
    status = cublasShutdown();
  }
}
