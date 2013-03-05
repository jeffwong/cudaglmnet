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

#define N 5

typedef struct {
  int n, p;
  thrust::device_ptr<float> X;
} data;
 
extern "C" {

    thrust::device_ptr<float> makeDeviceVector(float* x, int size) {
      thrust::host_vector<float> h(x, x + size);
      thrust::device_vector<float> d = h;
      return d.data();
    }

    thrust::device_ptr<float> makeEmptyDeviceVector(int size) {
      thrust::host_vector<float> h(size);
      thrust::device_vector<float> d = h;
      return d.data();
    }

    void foo(float* x, int* size) {
        printf("size %i\n", *size);
        data* mydata = (data*)malloc(sizeof(data));
        thrust::host_vector<float> h (N, 10);
        thrust::device_vector<float> d = h;
        mydata->X = makeDeviceVector(x, *size);
        free(mydata);
    }
}

int main() {
  int size = 5;
  float* x = (float*)malloc(sizeof(float) * size);
  foo(x, &size);
  free(x);
  return 0;
}
