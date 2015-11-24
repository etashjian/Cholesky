#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  unsigned dim = 4;

  // generate random input matrix (THIS PROBABLY ISN'T TOTALL CORRECT YET)
  srand(time(NULL));
  MatrixXd A = MatrixXd::Random(dim, dim);
  A *= 5;
  MatrixXd A_t = A.transpose();
  A += A_t;

  if(dim < 10)
    cout << "Input Matrix:" << endl << A << endl;

  // build L and D
  MatrixXd L = MatrixXd::Zero(dim, dim);
  DiagonalMatrix<double, Dynamic> D(dim);

  // calculate decomposition
  for(int j = 0; j < dim; ++j)
  {
    L(j, j) = 1;
    D.diagonal()[j] = A(j, j);
    for(int k = 0; k < j; ++k)
    {
      D.diagonal()[j] -= D.diagonal()[k] * L(j,k) * L(j,k);
    }

    for(unsigned i = j + 1; i < dim; ++i)
    {
      L(i,j) = A(i,j);
      for(int k = 0; k < j; ++k)
      {
        L(i,j) -= L(i,k) * L(j,k) * D.diagonal()[k];
      }
      L(i,j) /= D.diagonal()[j];
    }
  }

  if(A.isApprox(L * D * L.transpose()))
  {
    cout << "TEST PASSED!" << endl;
  }
  else
  {
    cout << "TEST FAILED!" << endl;
    cout << L * D * L.transpose() << endl;
  }
}
