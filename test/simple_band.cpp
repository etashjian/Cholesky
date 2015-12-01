#include <db/cholesky_band.h>
#include <db/band_matrix_type.h>
#include <db/common.h>
#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  dim_t dim = 10;
  dim_t bandwidth = 1;

  // generate random input matrix
  cout << "Generating input... " << flush;
  srand(time(NULL));
  BandMatrix A = createSymmetricPositiveDefiniteBandMatrix(dim, bandwidth);
  cout << "done" << endl;

  // allocate L and D and compute decomp
  cout << "Computing serial decomposition... " << endl;
  BandMatrix L = createEmptyBandMatrix(dim, bandwidth);
  BandMatrix D = createEmptyBandMatrix(dim, bandwidth);
  //cholesky_band_serial(A, L, D);

  /*
  // check results
  cout << "Comparing results... " << flush;
  int differences = cmp_matrices(result, A);
  if(differences == 0)
  {
    cout << "PASSED!" << endl;
  }
  else
  {
    cout << "FAILED! - " << differences << " differences" << endl;
  }
  */
}
