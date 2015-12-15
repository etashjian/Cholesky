#include "db/common.h"
#include "db/cholesky_band_cuda.h"

void cholesky_band_parallel_cuda( 
        const BandMatrix & A, 
        BandMatrix & L, 
        BandMatrix & D ) 
{
    //  init D and L with entries in A
    for( dim_t i = 0; i < A._matDim; i++ ) {
        D.writeEntry( i, i, A.getEntry( i, i ) );
        L.writeEntry( i, i, 1 );
        for( dim_t j = 1; j <= A._lowerBand && (i+j < A._matDim); j++ ) {
            L.writeEntry( i+j, i, A.getEntry( i+j, i ) );
        }
    }

    //  Copy A to device global memory
    const data_t * const hostA = &A._vals[0];
    data_t * hostD = &D._vals[0];
    data_t * hostL = &L._vals[0];
    data_t * devA = NULL;
    data_t * devD = NULL;
    data_t * devL = NULL;

    //  allocate memory in device
    myCudaCheck( cudaMalloc( (void**)&devA, sizeof(data_t) * A.getNumNonZeroEntries() ) );
    myCudaCheck( cudaMalloc( (void**)&devD, sizeof(data_t) * D.getNumNonZeroEntries() ) );
    myCudaCheck( cudaMalloc( (void**)&devL, sizeof(data_t) * L.getNumNonZeroEntries() ) );
    myCudaCheck( cudaMemcpy( devA, hostA, sizeof(data_t) * A.getNumNonZeroEntries(), cudaMemcpyHostToDevice ) );
    myCudaCheck( cudaMemcpy( devD, hostD, sizeof(data_t) * D.getNumNonZeroEntries(), cudaMemcpyHostToDevice ) );
    myCudaCheck( cudaMemcpy( devL, hostL, sizeof(data_t) * L.getNumNonZeroEntries(), cudaMemcpyHostToDevice ) );

    //  solve column by column
    for( dim_t colIdx = 0; colIdx < A._matDim; colIdx ++ ) {
        choleskyColumnSolverKernel<<< 1, L._lowerBand+1, (2*L._lowerBand+1)*sizeof(data_t)>>>( devA, devD, devL, colIdx, L._matDim, L._lowerBand );
    }

    myCudaCheck( cudaMemcpy( hostD, devD, sizeof(data_t) * D.getNumNonZeroEntries(), cudaMemcpyDeviceToHost ) );
    myCudaCheck( cudaMemcpy( hostL, devL, sizeof(data_t) * L.getNumNonZeroEntries(), cudaMemcpyDeviceToHost ) );
        
    cudaDeviceSynchronize();

#ifdef ENABLE_LOG
    std::cout << "cholesky on band matrix finishes... [parallel version (cuda)]\n";
#endif
}

__global__ void choleskyColumnSolverKernel( data_t * devA, data_t * devD, data_t * devL, const dim_t colIdx, const dim_t matDim, const dim_t bandWidth ) 
{
    extern __shared__ data_t temp[];
    data_t * prevD = &temp[0];          //  D[(col-k):(col-1)]
    data_t * currD = &temp[bandWidth];  //  D[col]
    data_t * prevL = &temp[bandWidth+1];//  L[col, (col-k):(col-1)]    

    if( threadIdx.x == 0 ) {
        //  devD -> prevD, currD
        for( dim_t i = colIdx-1; (i >= colIdx-bandWidth) && i>=0; i-- ) {
            prevD[ i - (colIdx-bandWidth) ] = devD[i];
        }
        currD[0] = devD[colIdx];
    } else {
        //  devL -> prevL
        dim_t row = colIdx;
        dim_t col = colIdx-threadIdx.x;
        if( col >= 0 ) {
            prevL[col - (colIdx-bandWidth)] = devL[col*(bandWidth+1) + (row-col)];
        }
    };
    __syncthreads();

    if( colIdx + threadIdx.x >= matDim ) {
        return;
    }
    dim_t col = colIdx;
    dim_t row = colIdx + threadIdx.x;
    data_t currL = 0;
    if( threadIdx.x == 0 ) {
        for( dim_t i = col-1; (i >= col-bandWidth) && i>=0; i-- ) {
            currD[0] -= prevL[i-(col-bandWidth)]*prevL[i-(col-bandWidth)]*prevD[i-(col-bandWidth)];
        }
    } else {
        if( row < matDim ) {
            currL = devL[col*(bandWidth+1) + (row-col)];
            for( dim_t i = col-1; (i >= col-bandWidth+threadIdx.x) && i>=0; i-- ) {
                currL -= prevD[i-(col-bandWidth)]*prevL[i-(col-bandWidth)]*devL[i*(bandWidth+1) + (row-i)];
            }
        }
    }

    __syncthreads();
    if( threadIdx.x == 0 ) {
        devD[colIdx] = currD[0];
    } else {
        if( row < matDim ) {
            devL[col*(bandWidth+1) + (row-col)] = currL / currD[0];
        }
    }
}
