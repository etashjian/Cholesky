#include <db/eigen_util.h>
#include <db/cholesky_eigen.h>
#include <db/common.h>
#include <db/cholesky_band_cuda.h>
#include <db/timer.h>
#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
    MyTimer _myTimer;
    dim_t dim = 1000;
    dim_t bandwidth = 10;

    // generate random input matrix
    cout << "Generating input... " << flush;
    _myTimer.startTimer();
    srand(time(NULL));
    BandMatrix A = createSymmetricPositiveDefiniteBandMatrix(dim, bandwidth);
    _myTimer.stopTimer();
    std::cout << "Elapsed Time to Create Input SPDB Matrix : " << _myTimer.elapsedInSec() << " second\n";

    // Compute reference case
    cout << "Computing Reference Case... " << flush;
    MatrixXf A_ref = band_to_eigen(A);
    MatrixXf L_ref = MatrixXf::Zero(dim, dim);
    MatrixXf D_ref = MatrixXf::Zero(dim, dim);
    cholesky_eigen_serial(A_ref, L_ref, D_ref);
    cout << "done" << endl;

    // allocate L and D and compute decomp
    cout << "Computing serial decomposition... " << endl;
    BandMatrix L = createEmptyBandMatrix(dim, bandwidth, 0);
    BandMatrix D = createEmptyBandMatrix(dim, 0, 0);

    _myTimer.startTimer();
    cholesky_band_serial_index_handling(A, L, D);
    _myTimer.stopTimer();
    std::cout << "Elapsed Time to Perform Cholesky Decomposition on Input SPDB Matrix : " << _myTimer.elapsedInSec() << " second\n";

    // compare results
    unsigned diff = cmp_matrices(L_ref, L);
    diff += cmp_matrices(D_ref, D);
    if(diff) cout << "FAILED" << endl;
    else cout << "PASSED" << endl;


    _myTimer.startTimer();
    cholesky_band_parallel_cuda(A, L, D);
    _myTimer.stopTimer();
    std::cout << "Elapsed Time to Perform Cholesky Decomposition on Input SPDB Matrix : " << _myTimer.elapsedInSec() << " second. Speedup=" << _myTimer.cummulativeSpeedup() << "x\n";

    diff = cmp_matrices(L_ref, L);
    diff += cmp_matrices(D_ref, D);
    if(diff) cout << "FAILED" << endl;
    else cout << "PASSED" << endl;
}
