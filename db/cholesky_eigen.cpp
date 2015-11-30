#include "cholesky_eigen.h"

using namespace Eigen;

#define B_DIM 2

////////////////////////////////////////////////////////////////////////////////
void cholesky_eigen_serial(const MatrixXf& A, MatrixXf& L, MatrixXf& D)
{
  assert(A.rows() == L.rows() && L.rows() == D.rows());
  assert(A.cols() == L.cols() && L.cols() == D.cols());

  // calculate decomposition
  for(int j = 0; j < A.rows(); ++j)
  {
    // compute D values
    D.diagonal()[j] = A(j, j);
    for(int k = 0; k < j; ++k)
    {
      D.diagonal()[j] -= D.diagonal()[k] * L(j,k) * L(j,k);
    }

    // compute L values
    L(j, j) = 1;
    for(unsigned i = j + 1; i < A.rows(); ++i)
    {
      L(i,j) = A(i,j);
      for(int k = 0; k < j; ++k)
      {
        L(i,j) -= L(i,k) * L(j,k) * D.diagonal()[k];
      }
      L(i,j) /= D.diagonal()[j];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
void cholesky_eigen_blocked_serial(const MatrixXf& A, MatrixXf& L, MatrixXf& D)
{
  assert(A.rows() == L.rows() && L.rows() == D.rows());
  assert(A.cols() == L.cols() && L.cols() == D.cols());

  MatrixXf I = MatrixXf::Identity(B_DIM, B_DIM);

  // do decomposition
  for(int j = 0; j < A.rows(); j+=B_DIM)
  {
    // calculate D
    MatrixXf d = A.block<B_DIM, B_DIM>(j, j);
    for(int k = 0; k < j; k += B_DIM)
    {
      d -= L.block<B_DIM, B_DIM>(j, k) *
           D.block<B_DIM, B_DIM>(k, k) *
           L.block<B_DIM, B_DIM>(j, k).transpose();
    }
    D.block<B_DIM, B_DIM>(j,j) = d;

    // calculate L
    L.block<B_DIM,B_DIM>(j,j) = I;
    for(int i = j + B_DIM; i < A.rows(); i+=B_DIM)
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
}

////////////////////////////////////////////////////////////////////////////////
void cholesky_eigen_omp_v1(const MatrixXf& A, MatrixXf& L, MatrixXf& D)
{
  assert(A.rows() == L.rows() && L.rows() == D.rows());
  assert(A.cols() == L.cols() && L.cols() == D.cols());

  // calculate decomposition
  for(int j = 0; j < A.rows(); ++j)
  {
    // compute D values
    D.diagonal()[j] = A(j, j);
    for(int k = 0; k < j; ++k)
    {
      D.diagonal()[j] -= D.diagonal()[k] * L(j,k) * L(j,k);
    }

    // compute L values
    L(j, j) = 1;
    #pragma omp parallel for
    for(unsigned i = j + 1; i < A.rows(); ++i)
    {
      L(i,j) = A(i,j);
      for(int k = 0; k < j; ++k)
      {
        L(i,j) -= L(i,k) * L(j,k) * D.diagonal()[k];
      }
      L(i,j) /= D.diagonal()[j];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
