#include <db/matrixgen_eigen.h>
#include <db/cholesky_eigen.h>
#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  unsigned dim = 1024;
  unsigned bandwidth = 3;
  unsigned range = 5;

  // generate random input matrix (THIS PROBABLY ISN'T TOTALL CORRECT YET)
  srand(time(NULL));
  MatrixXf A = create_banded_sym_pos_def_eigen_matrix(dim, bandwidth, range);

  if(dim < 10)
    cout << "Input Matrix:" << endl << A << endl;

  // allocate L and D and compute decomp
  MatrixXf L = MatrixXf::Zero(dim, dim);
  MatrixXf D = MatrixXf::Zero(dim, dim);
  cholesky_eigen_serial(A, L, D);

  // calculate result
  MatrixXf result = L * D * L.transpose();

  // check result
  unsigned num_differences = 0;
  for(int i = 0; i < dim; ++i)
  {
    for(int j = 0; j < dim; ++j)
    {
      if(abs(result(i,j) - A(i,j)) > .1)
        num_differences++;
    }
  }

  if(num_differences == 0)
  {
    cout << "TEST PASSED!" << endl;
  }
  else
  {
    cout << "TEST FAILED!" << endl;
    cout << num_differences << " differences" << endl;
  }
}
