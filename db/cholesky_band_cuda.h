#ifndef _CHOLESKY_CUDA_H_
#define _CHOLESKY_CUDA_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>

#define myCudaCheck(ans) { cuda_assert((ans), __FILE__, __func__, __LINE__); }
inline void cuda_assert( cudaError_t status, const char * file, const char * func, const int line ) {
    if( status != cudaSuccess ) {
        fprintf( stderr, "cuda error : %s %s %s %d\n", cudaGetErrorString(status), file, func, line );
    }
}

//  
//  When solving column [colIdx], 
//  To Write: 
//  (1) D[colIdx]   ( in serial )
//  (2) L[(colIdx+1):(colIdx+k), colIdx]

//  Solving D[colIdx] requires:
//  L[ colIdx, (colIdx-k):(colIdx-1) ], D[ 0:(colIdx-1) ]

//  Solving L[r:colIdx] ( colIdx+1 <= r <= colIdx+k ) requires:
//  D[ (colIdx-k):colIdx ]
//  L[ colIdx, (colIdx-k):(colIdx-1) ]
//  L[ r, (colIdx-k):(colIdx-1) ]

//  entries from L : 
//  L [colIdx:(colIdx+k), (colIdx-k):(colIdx-1)]
//  # required entries from L : (k+1)*k

//  entries from D : 
//  D[ 0:(colIdx-1) ]
//  # required entires from D : (colIdx)

__global__ void choleskyColumnSolverKernel( data_t * devA, data_t * devD, data_t * devL, const dim_t colIdx, const dim_t matDim, const dim_t bandWidth );

#endif
