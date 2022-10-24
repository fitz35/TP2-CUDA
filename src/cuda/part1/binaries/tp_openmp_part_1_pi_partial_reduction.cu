/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <fstream>

#define CHECK_CUDA_ERROR(x) checkCudaError(x, __FILE__, __LINE__)

static long num_steps = 100000000;
static int num_cores = 1;
static int num_threads_per_bloc = 1;
double step;

__global__ void pi_kernel(float *sum, float step, int steps_per_thread, int num_steps)
{
    float x;
    int i;

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float prox_partial_sum[1024];
    prox_partial_sum[threadIdx.x] = 0;
    for (i = id * steps_per_thread; i < (id + 1) * steps_per_thread && i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        prox_partial_sum[threadIdx.x] += 4.0 / (1.0 + x * x);
    }

    // printf("prox_partial_sum[%d] = %f \n", threadIdx.x, prox_partial_sum[threadIdx.x]);
    __syncthreads();
    // reduxion
    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (threadIdx.x < s)
        {
            prox_partial_sum[threadIdx.x] += prox_partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    // add to global sum
    if (threadIdx.x == 0)
    {
        atomicAdd(sum, prox_partial_sum[0]);
    }
}

inline void checkCudaError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s:%d: %s) \n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

int main(int argc, char **argv)
{

    // Read command line arguments.
    for (int i = 0; i < argc; i++)
    {
        if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-num_steps") == 0))
        {
            num_steps = atol(argv[++i]);
            printf("  User num_steps is %ld\n", num_steps);
        }

        if ((strcmp(argv[i], "-T") == 0) || (strcmp(argv[i], "-num_thread") == 0))
        {
            num_threads_per_bloc = atol(argv[++i]);
            printf("  User num_threads is %d\n", num_threads_per_bloc);
        }

        if ((strcmp(argv[i], "-C") == 0) || (strcmp(argv[i], "-num_cores") == 0))
        {
            num_cores = atol(argv[++i]);
            printf("  User num_cores is %d\n", num_cores);
        }
        else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
        {
            printf("  Pi Options:\n");
            printf("  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n");
            printf("  -help (-h):            print this message\n\n");
            exit(1);
        }
    }
    double pi;
    float *sum = (float *)malloc(sizeof(float));
    sum[0] = 0;

    float step = 1.0 / (double)num_steps;
    int all_threads = num_threads_per_bloc * num_cores;
    int steps_per_thread = floor(num_steps / (all_threads));
    printf("steps_per_thread = %d \n", steps_per_thread);
    printf("step = %f \n", step);

    // Timer products.
    struct timeval begin, end;

    // Allocate memory on the device
    //  result
    float *d_sum;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sum, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sum, sum, sizeof(float), cudaMemcpyHostToDevice));

    gettimeofday(&begin, NULL);
    // Launch kernel
    pi_kernel<<<num_cores, num_threads_per_bloc>>>(d_sum, step, steps_per_thread, num_steps);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    pi = step * sum[0];

    // Calculate time.
    double time = 1.0 * (end.tv_sec - begin.tv_sec) +
                  1.0e-6 * (end.tv_usec - begin.tv_usec);

    free(sum);
    cudaFree(d_sum);

    printf("\n pi with %ld steps is %lf in %lf seconds\n ", num_steps, pi, time);
    std::ofstream myfile;
    myfile.open("../pi_Stats.csv", std::ios_base::app);
    myfile << "Partial_Reduction," << num_steps << "," << num_cores << "," << num_threads_per_bloc << "," << time << "," << pi << std::endl;
    myfile.close();
    return 0;
}
