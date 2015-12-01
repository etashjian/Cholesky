#include <db/cholesky_band.h>
#include <db/cholesky_eigen.h>
#include <db/band_matrix_type.h>
#include <db/eigen_util.h>
#include <db/common.h>
#include <db/timer.h>
#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
    MyTimer _myTimer;
    dim_t dim = 1000;
    dim_t bandwidth = 1;

    // generate random input matrix
    cout << "Generating input... " << flush;
    _myTimer.startTimer();
    srand(time(NULL));
    BandMatrix A = createSymmetricPositiveDefiniteBandMatrix(dim, bandwidth);
    _myTimer.stopTimer();
    std::cout << "Elapsed Time to Create Input SPDB Matrix : " << _myTimer.elapsedInSec() << " second\n";

    // allocate L and D and compute decomp
    cout << "Computing serial decomposition... " << endl;
    BandMatrix L = createEmptyBandMatrix(dim, bandwidth);
    BandMatrix D = createEmptyBandMatrix(dim, bandwidth);

    _myTimer.startTimer();
    cholesky_band_serial(A, L, D);
    _myTimer.stopTimer();
    std::cout << "Elapsed Time to Perform Cholesky Decomposition on Input SPDB Matrix : " << _myTimer.elapsedInSec() << " second\n";

    _myTimer.startTimer();
    cholesky_band_serial_index_handling(A, L, D);
    _myTimer.stopTimer();
    std::cout << "Elapsed Time to Perform Cholesky Decomposition on Input SPDB Matrix : " << _myTimer.elapsedInSec() << " second\n";
}
