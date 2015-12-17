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
  MyTimer timer;

  // print header
  cout << "matrix dimension, matrix bandwidth, runtime serial, runtime omp for,"
       <<  "runtime omp task,runtime cuda,\n";

  // get results for varying matrix dimension
  for(dim_t dim = 1 << 8; dim < 1 << 16; dim <<= 1)
  {
    for(dim_t bandwidth = 1; bandwidth < min(dim, 1 << 10); bandwidth <<= 1)
    {
      cout << dim << "," << bandwidth << ",";

      // create input matrix
      BandMatrix a = createSymmetricPositiveDefiniteBandMatrix(dim, bandwidth);

      // run serial case
      BandMatrix l_ref = createEmptyBandMatrix(dim, bandwidth, 0);
      BandMatrix d_ref = createEmptyBandMatrix(dim, 0, 0);
      timer.startTimer();
      cholesky_band_serial_index_handling(a, l_ref, d_ref);
      timer.stopTimer();
      cout << timer.elapsedInSec() << ",";

      // run open mp for
      BandMatrix l = createEmptyBandMatrix(dim, bandwidth, 0);
      BandMatrix d = createEmptyBandMatrix(dim, 0, 0);
      timer.startTimer();
      cholesky_band_serial_index_handling_omp_v2(a, l, d);
      timer.stopTimer();
      cout << timer.elapsedInSec();
      if(!checkBandMatrixEqual( l_ref, l ) || !checkBandMatrixEqual( d_ref, d ))
        cout << "- failed!" << endl;
      cout << ",";

      // run open mp task
      l = createEmptyBandMatrix(dim, bandwidth, 0);
      d = createEmptyBandMatrix(dim, 0, 0);
      timer.startTimer();
      cholesky_band_serial_index_handling_omp_v3(a, l, d);
      timer.stopTimer();
      cout << timer.elapsedInSec();
      if(!checkBandMatrixEqual( l_ref, l ) || !checkBandMatrixEqual( d_ref, d ))
        cout << "- failed!" << endl;
      cout << ",";

      // run cuda
      l = createEmptyBandMatrix(dim, bandwidth, 0);
      d = createEmptyBandMatrix(dim, 0, 0);
      timer.startTimer();
      cholesky_band_parallel_cuda(a, l, d);
      timer.stopTimer();
      cout << timer.elapsedInSec();
      if(!checkBandMatrixEqual( l_ref, l ) || !checkBandMatrixEqual( d_ref, d ))
        cout << "- failed!" << endl;
      cout << ",\n";
    }
  }
}
