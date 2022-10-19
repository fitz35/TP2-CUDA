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

static long num_steps = 100000000;
static int num_cores = 1;
double step;

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

    int core;
    double x, pi, sum = 0.0;
    double *partial_sum = new double[num_cores];
    for(int i = 0; i < num_cores; i++)
    {
        partial_sum[i] = 0.0;
    }
    int num_steps_per_core = num_steps / num_cores;
    int extra_steps = num_steps % num_cores;
    step = 1.0 / (double)num_steps;

    // Timer products.
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

#pragma omp parallel for private(x) num_threads(num_cores)
    for (core = 1; core <= num_cores; core++)
    {
        if (core == num_cores)
        {
            for (int i = num_steps_per_core*(core-1); i <= num_steps_per_core*core + extra_steps; i++)
            {
                x = (i - 0.5) * step;
                partial_sum[core - 1] = partial_sum[core - 1] + 4.0 / (1.0 + x * x);
            }
        }
        else
        {
            for (int i = num_steps_per_core*(core-1); i <= num_steps_per_core*core; i++)
            {
               x = (i - 0.5) * step;
                partial_sum[core - 1] = partial_sum[core - 1] + 4.0 / (1.0 + x * x);
            }
        }
    }
    for (core = 1; core <= num_cores; core++)
    {
        sum = sum + partial_sum[core - 1];
    }

    pi = step * sum;

    gettimeofday(&end, NULL);

    // Calculate time.
    double time = 1.0 * (end.tv_sec - begin.tv_sec) +
                  1.0e-6 * (end.tv_usec - begin.tv_usec);

    printf("\n pi with %ld steps is %lf in %lf seconds\n ", num_steps, pi, time);
    std::ofstream myfile;
    myfile.open("../pi_Stats.csv", std::ios_base::app);
    myfile << "Split_array," << num_steps << "," << num_cores << "," << time << "," << pi << std::endl;
    myfile.close();
    return 0;
}
