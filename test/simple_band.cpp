#include <db/eigen_util.h>
#include <db/cholesky_eigen.h>
#include <db/common.h>
#include <db/cholesky_band_cuda.h>
#include <db/timer.h>
#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

int main( int argc, char ** argv )
{
    MyTimer _myTimer;
    dim_t dim = atoi( argv[1] );
    dim_t bandwidth = atoi( argv[2] );

    // generate random input matrix
    cout << "Generating input band matrix : dim=" << dim << " band=" << bandwidth << endl;
    _myTimer.startTimer();
    srand(time(NULL));
    BandMatrix A = createSymmetricPositiveDefiniteBandMatrix(dim, bandwidth);
    _myTimer.stopTimer();
    std::cout << "Elapsed Time to Create Input SPDB Matrix : " << _myTimer.elapsedInSec() << " second\n";

    // Compute reference case
//    cout << "Computing Reference Case... " << flush;
//    MatrixXf A_ref = band_to_eigen(A);
//    MatrixXf L_ref = MatrixXf::Zero(dim, dim);
//    MatrixXf D_ref = MatrixXf::Zero(dim, dim);
//    cholesky_eigen_serial(A_ref, L_ref, D_ref);
//    cout << "done" << endl;

    // allocate L and D and compute decomp
//    cout << "Computing serial decomposition... " << endl;
    BandMatrix L_ref = createEmptyBandMatrix(dim, bandwidth, 0);
    BandMatrix D_ref = createEmptyBandMatrix(dim, 0, 0);
    BandMatrix L = createEmptyBandMatrix(dim, bandwidth, 0);
    BandMatrix D = createEmptyBandMatrix(dim, 0, 0);

    //  TEST!!! (REFERENCE)
    _myTimer.startTimer();
    cholesky_band_serial(A, L_ref, D_ref);
    _myTimer.stopTimer();
    printf( "Elapsed Time to Perform Cholesky Decomposition on Input SPDB Matrix : %f seconds\n",
            _myTimer.elapsedInSec() );
    // compare results
//    diff = cmp_matrices(L_ref, L);
//    diff += cmp_matrices(D_ref, D);
//    if(diff) cout << "FAILED" << endl;
//    else cout << "PASSED" << endl;

    //  TEST!!!
    _myTimer.startTimer();
    cholesky_band_serial_index_handling(A, L, D);
    _myTimer.stopTimer();
    printf( "Elapsed Time to Perform Cholesky Decomposition on Input SPDB Matrix : %f seconds\n",
            _myTimer.elapsedInSec() );
    // compare results
//    diff = cmp_matrices(L_ref, L);
//    diff += cmp_matrices(D_ref, D);
    if(checkBandMatrixEqual( L_ref, L ) && checkBandMatrixEqual( D_ref, D ) ) cout << "PASSED" << endl;
    else cout << "FAILED" << endl;

    //  TEST!!!
    L = createEmptyBandMatrix(dim, bandwidth, 0);
    D = createEmptyBandMatrix(dim, 0, 0);
    _myTimer.startTimer();
    cholesky_band_parallel_cuda(A, L, D);
    _myTimer.stopTimer();
    printf( "Elapsed Time to Perform Cholesky Decomposition on Input SPDB Matrix : %f seconds\n",
            _myTimer.elapsedInSec() );
    //  compare results
    if( checkBandMatrixEqual(L_ref, L) == false ) {
        cout << "FAILED" << endl;
//        L_ref.printBandMatrix();
//        L.printBandMatrix();
    } else if( checkBandMatrixEqual( D_ref, D ) == false ) {
        cout << "FAILED" << endl;
//        D_ref.printBandMatrix();
//        D.printBandMatrix();
    } else {
        cout << "PASSED" << endl;
    }

    //  TEST!!!
    L = createEmptyBandMatrix(dim, bandwidth, 0);
    D = createEmptyBandMatrix(dim, 0, 0);
    _myTimer.startTimer();
    cholesky_band_serial_index_handling_omp_v1(A, L, D);
    _myTimer.stopTimer();
    printf( "Elapsed Time to Perform Cholesky Decomposition on Input SPDB Matrix : %f seconds\n",
            _myTimer.elapsedInSec() );
    // compare results
//    diff = cmp_matrices(L_ref, L);
//    diff += cmp_matrices(D_ref, D);
    if(checkBandMatrixEqual( L_ref, L ) && checkBandMatrixEqual( D_ref, D ) ) cout << "PASSED" << endl;
    else cout << "FAILED" << endl;

    //  TEST!!!
    L = createEmptyBandMatrix(dim, bandwidth, 0);
    D = createEmptyBandMatrix(dim, 0, 0);
    _myTimer.startTimer();
    cholesky_band_serial_index_handling_omp_v2(A, L, D);
    _myTimer.stopTimer();
    printf( "Elapsed Time to Perform Cholesky Decomposition on Input SPDB Matrix : %f seconds\n",
            _myTimer.elapsedInSec() );
    // compare results
//    diff = cmp_matrices(L_ref, L);
//    diff += cmp_matrices(D_ref, D);
    if(checkBandMatrixEqual( L_ref, L ) && checkBandMatrixEqual( D_ref, D ) ) cout << "PASSED" << endl;
    else cout << "FAILED" << endl;

    return 0;
}
