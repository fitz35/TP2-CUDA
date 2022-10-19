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
static int num_cores=1;
double step;

int main (int argc, char** argv)
{
    
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        }
        if ( ( strcmp( argv[ i ], "-C" ) == 0 ) || ( strcmp( argv[ i ], "-num_cores" ) == 0 ) ) {
            num_cores = atol( argv[ ++i ] );
            printf( "  User num_cores is %d\n", num_cores );
        
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
	  int i;
	  double x, pi, sum = 0.0;
	  
      step = 1.0/(double) num_steps;

      // Timer products.
      struct timeval begin, end;

      gettimeofday( &begin, NULL );

	  #pragma omp parallel for shared(sum) num_threads(num_cores) private(x,i)
	  for (i=1;i<= num_steps; i++){
		  x = (i-0.5)*step;
      #pragma omp atomic
		  sum = sum + 4.0/(1.0+x*x);
      }
	  pi = step * sum;

      
      gettimeofday( &end, NULL );

      // Calculate time.
      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
      printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,time);
       std::ofstream myfile;
      myfile.open ("../pi_Stats.csv", std::ios_base::app);
      myfile << "Atomic,"<< num_steps << "," << num_cores << "," << time << "," << pi<<std::endl ;
      myfile.close();
      return 0;
}
