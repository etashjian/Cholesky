#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  MatrixXd A(3, 3);
  A << 4, -12, -16, 12, 37, -43, -16, -43, 98;
  cout << "The matrix A is" << endl << A << endl;

  LLT<MatrixXd> lltOfA(A); // performs decomp
  MatrixXd L = lltOfA.matrixL();

  cout << "Solution for Cholesky LL" << endl;
  cout << "Cholesky L is" << endl << L << endl;
  cout << "Cholesky L_T is" << endl << L.transpose() << endl;
  cout << "L * L_T is" << endl << L * L.transpose() << endl << endl;

  // try figuring out decomp ourselves
  L = MatrixXd::Zero(3, 3);
  DiagonalMatrix<double, Dynamic> D(3);

  for(int j = 0; j < 3; ++j)
  {
    L(j, j) = 1;
    D.diagonal()[j] = A(j, j);
    for(int k = 0; k < j; ++k)
    {
      D.diagonal()[j] -= D.diagonal()[k] * L(j,k) * L(j,k);
    }

    for(unsigned i = j + 1; i < 3; ++i)
    {
      L(i,j) = A(i,j);
      for(int k = 0; k < j; ++k)
      {
        L(i,j) -= L(i,k) * L(j,k) * D.diagonal()[k];
      }
      L(i,j) /= D.diagonal()[j];
    }
  }

  cout << "My solution for LDLT" << endl;
  cout << "L is" << endl << L << endl;
  cout << "D is" << endl << D.diagonal() << endl;
  cout << "LDLT is" << endl << L * D * L.transpose() << endl;
}
