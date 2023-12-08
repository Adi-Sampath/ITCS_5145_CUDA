#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
//
float polynomial (float x, float* poly, int degree) {
  float out = 0.;
  float xtothepowerof = 1.;
  for (int i=0; i<=degree; ++i) {
    out += xtothepowerof*poly[i];
    xtothepowerof *= x;
  }
  return out;
}

__global__ void polynomial_expansion_kernel (float* poly, int degree, int n, float* array) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<n) {
    float x = array[i];
    float out = 0.;
    float exp_x = 1.;
    for (int i=0; i<=degree; ++i) {
      out += exp_x*poly[i];
      exp_x *= x;
    }
    array[i] = out;
  }
}

void polynomial_expansion (float* poly, int degree, int n, float* array) {

  float* d_poly;
  float* d_array;

  cudaMalloc(&d_poly, (degree+1)*sizeof(float));
  cudaMalloc(&d_array, n*sizeof(float));

  cudaMemcpy(d_poly, poly, (degree+1)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_array, array, n*sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  polynomial_expansion_kernel<<<numBlocks, blockSize>>>(d_poly, degree, n, d_array);

  cudaMemcpy(array, d_array, n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_poly);
  cudaFree(d_array);
}


int main (int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr<<"usage: "<<argv[0]<<" n degree"<<std::endl;
    return -1;
  }

  int n = atoi(argv[1]); //TODO: atoi is an unsafe function
  int degree = atoi(argv[2]);
  int nbiter = 1;

  float* array = new float[n];
  float* poly = new float[degree+1];
  for (int i=0; i<n; ++i)
    array[i] = 1.;

  for (int i=0; i<degree+1; ++i)
    poly[i] = 1.;

  
  std::chrono::time_point<std::chrono::system_clock> begin, end;
  begin = std::chrono::system_clock::now();
  
  for (int iter = 0; iter<nbiter; ++iter)
    polynomial_expansion (poly, degree, n, array);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> totaltime = (end-begin)/nbiter;

  std::cerr<<array[0]<<std::endl;
  std::cout<<n<<" "<<degree<<" "<<totaltime.count()<<std::endl;

  delete[] array;
  delete[] poly;

  return 0;
}
