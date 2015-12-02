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
