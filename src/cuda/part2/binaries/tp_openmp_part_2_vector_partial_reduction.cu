/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <fstream>

#include <cmath>

#define MAX_THREADS 1024

#define CHECK_CUDA_ERROR(x) checkCudaError(x, __FILE__, __LINE__)

inline void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s:%d: %s)", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void matrixKernel(float * result, float * x, float * y, float * A, int N, int M) {
  int iN = blockIdx.x;
  int iM = threadIdx.x;
  __shared__ float sum[MAX_THREADS];
  sum[threadIdx.x] = A[iN*M+iM]*x[iM];
  __syncthreads();
  // reduce
  for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (threadIdx.x < s)
        {
            sum[threadIdx.x] += sum[threadIdx.x + s];
        }
        __syncthreads();
    }

  if (threadIdx.x == 0) {
    y[iN] *= sum[0];
    atomicAdd(&result[0], y[iN]);
  }
}

void checkSizes(int &N, int &M, long &S);

int main(int argc, char *argv[])
{
  int N = -1;        // number of rows 2^12
  int M = -1;        // number of columns 2^10
  long S = -1;       // total size 2^22

  // Read command line arguments.
  for (int i = 0; i < argc; i++)
  {
    if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-Rows") == 0))
    {
      N = pow(2, atoi(argv[++i]));
      printf("  User N is %d\n", N);
    }
    else if ((strcmp(argv[i], "-M") == 0) || (strcmp(argv[i], "-Columns") == 0))
    {
      M = pow(2, atof(argv[++i]));
      printf("  User M is %d\n", M);
    }
    else if ((strcmp(argv[i], "-S") == 0) || (strcmp(argv[i], "-Size") == 0))
    {
      S = pow(2, atof(argv[++i]));
      printf("  User S is %ld\n", S);
    }
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
    {
      printf("  y^T*A*x Options:\n");
      printf("  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n");
      printf("  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n");
      printf("  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n");
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  // Check sizes.
  checkSizes(N, M, S);

  // Allocate x,y,A
  auto y = new float[N];
  for (int i = 0; i < N; i++)
  {
    y[i] = 1;
  }
  auto x = new float[M];
  for (int i = 0; i < M; i++)
  {
    x[i] = 1;
  }
  auto A = new float[S];
  for (int i = 0; i < S; i++)
  {
    A[i] = 1;
  }

  float * result;
  result = new float[1];
  result[0] = 0;

  // Allocate device memory.
  float *d_x, *d_y, *d_A;
  CHECK_CUDA_ERROR(cudaMalloc(&d_x, M * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_y, N * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_A, S * sizeof(float)));
  float *d_result;
  CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(float)));

  // Copy data to device.
  CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, S * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_result, result, sizeof(float), cudaMemcpyHostToDevice));

  // Timer products.
  struct timeval begin, end;

  gettimeofday(&begin, NULL);

  // For each line i
  // Multiply the i lines with the vector x
  // Sum the results of the previous step into a single variable
  // Multiply the result of the previous step with the i value of vector y
  // Sum the results of the previous step into a single variable (result)
  matrixKernel<<<N, M>>>(d_result, d_x, d_y, d_A, N, M);
  cudaDeviceSynchronize();

  CHECK_CUDA_ERROR(cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

  printf("  Computed result for %d x %d is %lf\n", N, M, result[0]);

  const float solution = (float)N * (float)M;

  if (result[0] != solution)
  {
    printf("  Error: result( %lf ) != solution( %lf )\n", result[0], solution);
  }

  gettimeofday(&end, NULL);

  // Calculate time.
  // float time = timer.seconds();
  float time = 1.0 * (end.tv_sec - begin.tv_sec) +
                1.0e-6 * (end.tv_usec - begin.tv_usec);

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // float Gbytes = 1.0e-9 * float( sizeof(float) * ( 2 * M * N + N ) );
  float Gbytes = 1.0e-9 * float(sizeof(float) * (M + M * N + N));

  // Print results (problem size, time and bandwidth in GB/s).
  printf("  N( %d ) M( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
         N, M, Gbytes * 1000, time, Gbytes / time);

  std::free(A);
  std::free(y);
  std::free(x);

  // Free device memory.
  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_y));
  CHECK_CUDA_ERROR(cudaFree(d_x));
  CHECK_CUDA_ERROR(cudaFree(d_result));

  std::ofstream myfile;
  myfile.open("../vector_Stats.csv", std::ios_base::app);
  myfile << "Partial_reduction," << N << "," << M << "," << time << "," << (Gbytes * 1000) << "," << (Gbytes / time) << std::endl;
  myfile.close();
  return 0;
}

void checkSizes(int &N, int &M, long &S)
{
  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of N and M.
  if (S == -1 && (N == -1 || M == -1))
  {
    S = pow(2, 22);
    if (S < N)
      S = N;
    if (S < M)
      S = M;
  }

  // If S is undefined and both N and M are defined, set S = N * M.
  if (S == -1)
    S = N * M;

  // If both N and M are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if (N == -1 && M == -1)
  {
    if (S > 1024)
    {
      M = 1024;
    }
    else
    {
      M = S;
    }
  }

  // If only M is undefined, set it.
  if (M == -1)
    M = S / N;

  // If N is undefined, set it.
  if (N == -1)
    N = S / M;

  printf("  Total size S = %ld N = %d M = %d\n", S, N, M);

  // Check sizes.
  if ((S < 0) || (N < 0) || (M < 0))
  {
    printf("  Sizes must be greater than 0.\n");
    exit(1);
  }

  if ((N * M) != S)
  {
    printf("  N * M != S\n");
    exit(1);
  }
}
