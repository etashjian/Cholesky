#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <iostream>

using namespace std;
using namespace Eigen;

#define B_DIM 2

int main()
{
  unsigned dim = 1024;

  // generate random input matrix (THIS PROBABLY ISN'T TOTALL CORRECT YET)
  srand(time(NULL));
  MatrixXd A = MatrixXd::Random(dim, dim);
  A *= 5;
  MatrixXd A_t = A.transpose();
  A += A_t;
  MatrixXd A_I = MatrixXd::Identity(dim, dim);
  A_I *= dim;
  A += A_I;

  if(dim < 10)
    cout << "Input Matrix:" << endl << A << endl;

  // build L and D
  MatrixXd L = MatrixXd::Zero(dim, dim);
  MatrixXd D = MatrixXd::Zero(dim, dim);
  MatrixXd I = MatrixXd::Identity(B_DIM, B_DIM);

  // do decomposition
  for(int j = 0; j < dim; j+=B_DIM)
  {
    // calculate D
    MatrixXd d = A.block<B_DIM, B_DIM>(j, j);
    for(int k = 0; k < j; k += B_DIM)
    {
      d -= L.block<B_DIM, B_DIM>(j, k) *
           D.block<B_DIM, B_DIM>(k, k) *
           L.block<B_DIM, B_DIM>(j, k).transpose();
    }
    D.block<B_DIM, B_DIM>(j,j) = d;

    // calculate L
    L.block<B_DIM,B_DIM>(j,j) = I;
    for(int i = j + B_DIM; i < dim; i+=B_DIM)
    {
      L.block<B_DIM,B_DIM>(i, j) = A.block<B_DIM,B_DIM>(i,j);
      for(int k = 0; k < j; k+=B_DIM)
      {
        L.block<B_DIM,B_DIM>(i,j) -= L.block<B_DIM,B_DIM>(i,k) *
                                     D.block<B_DIM,B_DIM>(k,k) *
                                     L.block<B_DIM,B_DIM>(j,k).transpose();
      }
      L.block<B_DIM,B_DIM>(i,j) *= D.block<B_DIM,B_DIM>(j,j).inverse();
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
