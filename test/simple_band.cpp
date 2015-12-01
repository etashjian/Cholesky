#include <db/cholesky_band.h>
#include <db/cholesky_eigen.h>
#include <db/band_matrix_type.h>
#include <db/eigen_util.h>
#include <db/common.h>
#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  dim_t dim = 2048;
  dim_t bandwidth = 1;

  // generate random input matrix
  cout << "Generating input... " << flush;
  srand(time(NULL));
  BandMatrix A = createSymmetricPositiveDefiniteBandMatrix(dim, bandwidth);
  cout << "done" << endl;

  // allocate L and D and compute decomp
  cout << "Computing serial decomposition... " << flush;
  BandMatrix L = createEmptyBandMatrix(dim, bandwidth);
  BandMatrix D = createEmptyBandMatrix(dim, bandwidth);
  cholesky_band_serial(A, L, D);
  cout << "done" << endl;

  // compute reference case
  cout << "Compute reference case... " << flush;
  MatrixXf A_ref = band_to_eigen(A);
  MatrixXf L_ref = MatrixXf::Zero(dim, dim);
  MatrixXf D_ref = MatrixXf::Zero(dim, dim);
  cholesky_eigen_serial(A_ref, L_ref, D_ref);
  cout << "done" << endl;

  // check results
  cout << "Comparing results... " << flush;
  unsigned differences = cmp_matrices(L_ref, L);
  if(differences == 0)
  {
    cout << "PASSED!" << endl;
  }
  else
  {
    cout << "FAILED! - " << differences << " differences" << endl;
  }
}
