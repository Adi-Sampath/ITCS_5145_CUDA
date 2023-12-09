#include <iostream>
#include <chrono>
#include <cuda_runtime.h>



__global__ void polynomial_expansion(float* poly, int degree, int n, float* array) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        float tmp = array[i];
        float result = 0.;
        for (int j = 0; j < degree+1; j++) {
            result += poly[j] * tmp;
            tmp *= array[i];
        }
        array[i] = result;
    }
}



int main (int argc, char* argv[]) {
  //TODO: add usage
  
  if (argc < 3) {
    std::cerr<<"usage: "<<argv[0]<<" n degree "<<std::endl;
    return -1;
  }

  int n = atoi(argv[1]); //TODO: atoi is an unsafe function
  int degree = atoi(argv[2]);
  int block_size = 256;
  int nbiter = 1;

  float* array = new float[n];
  float* poly = new float[degree+1];
  for (int i=0; i<n; ++i)
    array[i] = 1.;

  for (int i=0; i<degree+1; ++i)
    poly[i] = 1.;

  
  std::chrono::time_point<std::chrono::system_clock> begin, end;
  begin = std::chrono::system_clock::now();

  
  // Code Add Here
  float *d_array, *d_poly;
  cudaMalloc(&d_array, n*sizeof(float)); 

  cudaMalloc(&d_poly, (degree+1)*sizeof(float));
  cudaMemcpy(d_array, array, n*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_poly, poly, (degree+1)*sizeof(float), cudaMemcpyHostToDevice);

  polynomial_expansion<<<(n+block_size-1)/block_size, block_size>>>(d_poly, degree, n, d_array);
  cudaMemcpy(array, d_array, n*sizeof(float), cudaMemcpyDeviceToHost);

  //for (int iter = 0; iter<nbiter; ++iter)
  //  polynomial_expansion (poly, degree, n, array);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> totaltime = (end-begin)/nbiter;

  {
    bool correct = true;
    int ind;
    for (int i=0; i< n; ++i) {
      if (fabs(array[i]-(degree+1))>0.01) {
        correct = false;
        ind = i;
      }
    }
    if (!correct)
      std::cerr<<"Result is incorrect. In particular array["<<ind<<"] should be "<<degree+1<<" not "<< array[ind]<<std::endl;
  }
  

  std::cerr<<array[0]<<std::endl;
  std::cout<<n<<" "<<degree<<" "<<totaltime.count()<<std::endl;

  //make sure you clean up everything!

  delete[] array;
  delete[] poly;

  return 0;
}
