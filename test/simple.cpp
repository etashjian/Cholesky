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
}
