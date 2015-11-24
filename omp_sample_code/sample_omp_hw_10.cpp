/*
 * =====================================================================================
 *
 *       Filename:  my_hw_10.cpp
 *
 *    Description:  implement an OMP version to compute an integral
 *
 *        Version:  1.0
 *        Created:  11/24/2015 03:44:03 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  DAOHANG
 *   Organization:  
 *
 * =====================================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <ctime>
#include <omp.h>
#include <sys/time.h>

struct MyTimer {
    struct timeval start, stop;
    inline void startTimer() { gettimeofday( &start, NULL ); }
    inline void stopTimer() { gettimeofday( &stop, NULL ); }
    inline double duration() {
        return ( stop.tv_sec + stop.tv_usec * 1.0 / 1000000 ) - ( start.tv_sec + start.tv_usec * 1.0 / 1000000 );
    }
};

#define NUM_THREADS 16

const double h = 0.0001;
const int   n = ( 100 - 0 ) / h;

inline double func( const double x ) {
    return exp( sin(x) ) * cos( x/40.f );
}

double seqInt();

double ompInt();

int main( int argc, char** argv ) {
    MyTimer timer;
    double result = 0.f;
    //  sequential
    timer.startTimer();
    result = seqInt();
    timer.stopTimer();
    printf( "sequential execution : %lf\n", result );
    printf( "sequantial runtime : %lf\n", timer.duration() );
    //  omp
    double start = omp_get_wtime();
    result = ompInt();
    double stop = omp_get_wtime();
    printf( "OMP execution : %lf\n", result );
    printf( "OMP runtime : %lf\n", stop - start );
    return 0;
}

double seqInt() {
    double r = 0.f;
    std::vector<double> disPts( n+1, 0 );
    //  init the dis pts
    for( int i = 0; i <= n; i++ ) {
        disPts[i] = i * 100.0 / (n+1);
        const double coe = ( i==0 || i==n ) ? 17 
            : ( i==1 || i==n-1 ) ? 59
            : ( i==2 || i==n-2 ) ? 43
            : ( i==3 || i==n-3 ) ? 49 : 48;
        r += coe * func(disPts[i]);
    }
    r = r * h / 48.0;
    return r;
}

double ompInt() {
    double r = 0.f;
    std::vector<double> disPts( n+1, 0 );
    std::vector<double> subSums( NUM_THREADS, 0 );
    //  init the dis pts
#pragma omp parallel num_threads(NUM_THREADS)
    {
#pragma omp for schedule (static, NUM_THREADS) 
        for( int i = 0; i <= n; i++ ) {
            disPts[i] = i * 100.0 / (n+1);
            subSums[omp_get_thread_num()] += func(disPts[i]) * ( ( i==0 || i==n ) ? 17.0
                    : ( i==1 || i==n-1 ) ? 59.0
                    : ( i==2 || i==n-2 ) ? 43.0
                    : ( i==3 || i==n-3 ) ? 49.0 : 48.0 );
        }
    }
    r = std::accumulate( subSums.begin(), subSums.end(), 0.f );
    r = r * h / 48.0;
    return r;
}
