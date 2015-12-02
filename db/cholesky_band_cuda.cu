#include "db/common.h"
#include "db/cholesky_band_cuda.h"

void cholesky_band_parallel_cuda( 
        const BandMatrix & A, 
        BandMatrix & L, 
        BandMatrix & D ) 
{
    CholeskySolver solver( A, L, D );
    solver.choleskyDecomposition();
    L = solver.getL();
    D = solver.getD();
}

void CholeskySolver::choleskyDecomposition() 
{
    for( dim_t colIdx = 0; colIdx < _A._matDim; colIdx ++ ) {
        choleskySolvingColumn( colIdx );
    }
}

void CholeskySolver::choleskySolvingColumn( const dim_t colIdx ) 
{
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
}
