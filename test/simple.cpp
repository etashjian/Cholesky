#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  unsigned dim = 1024;
  unsigned k1 = 1, k2 = 1; // tridiagonal matrix

  // generate random input matrix (THIS PROBABLY ISN'T TOTALL CORRECT YET)
  srand(time(NULL));
  MatrixXd A = MatrixXd::Random(dim, dim);
  A *= 5;
  // set non banded elements to 0
  for(int i = 0; i < dim; i++)
    for(int j = 0; j < dim; j++)
      if(j < i - k1 || j > i + k2)
        A(i,j) = 0;

  MatrixXd A_t = A.transpose();
  A += A_t;
  MatrixXd A_I = MatrixXd::Identity(dim, dim);
  A_I *= dim;
  A += A_I;

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

  // calculate result
  MatrixXd result = L * D * L.transpose();

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
