/**
 * \file eigen_util.cpp
 * \brief General functions for working with Eigen matrices
 */

#include "eigen_util.h"

using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
MatrixXf create_sym_pos_def_eigen_matrix(const dim_t mat_dim, const int range)
{
  // generate matrix with random values from 0 - 1
  MatrixXf A = MatrixXf::Random(mat_dim, mat_dim);

  // adjust values to fit range
  A *= 2 * range;
  A.array() -= range;

  // add transpose to make matrix symettirc
  MatrixXf A_t = A.transpose();
  A += A_t;

  // hack to make positive definite
  MatrixXf A_I = MatrixXf::Identity(mat_dim, mat_dim);
  A_I *= mat_dim;
  A += A_I;

  return A;
}

////////////////////////////////////////////////////////////////////////////////
MatrixXf create_banded_sym_pos_def_eigen_matrix(const dim_t mat_dim,
                                                const dim_t bandwidth,
                                                const int range)
{
  // generate matrix with random values from 0 - 1
  MatrixXf A = MatrixXf::Random(mat_dim, mat_dim);

  // adjust values to fit range
  A *= 2 * range;
  A.array() -= range;

  // set non banded elements to 0
  int k1 = bandwidth, k2 = bandwidth;
  for(int i = 0; i < mat_dim; i++)
    for(int j = 0; j < mat_dim; j++)
      if(j < i - k1 || j > i + k2)
        A(i,j) = 0;

  // add transpose to make matrix symettirc
  MatrixXf A_t = A.transpose();
  A += A_t;

  // hack to make positive definite
  MatrixXf A_I = MatrixXf::Identity(mat_dim, mat_dim);
  A_I *= mat_dim;
  A += A_I;

  return A;
}

////////////////////////////////////////////////////////////////////////////////
MatrixXf band_to_eigen(const BandMatrix& mat)
{
  MatrixXf e_mat = MatrixXf::Zero(mat._matDim, mat._matDim);

  // copy over values
  for(dim_t i = 0; i < mat._matDim; ++i)
    for(dim_t j = 0; j < mat._matDim; ++j)
      e_mat(i,j) = mat.getEntry(i,j);

  return e_mat;
}

////////////////////////////////////////////////////////////////////////////////
unsigned cmp_matrices(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B)
{
  assert(A.cols() == B.cols() && A.rows() == B.rows());

  unsigned num_differences = 0;
  for(dim_t i = 0; i < A.rows(); ++i)
  {
    for(dim_t j = 0; j < A.cols(); ++j)
    {
      if(abs(B(i,j) - A(i,j)) > .01)
        num_differences++;
    }
  }

  return num_differences;
}

////////////////////////////////////////////////////////////////////////////////
unsigned cmp_matrices(const Eigen::MatrixXf& A, const BandMatrix& B)
{
  assert(A.cols() == B._matDim && A.rows() == B._matDim);

  unsigned num_differences = 0;
  for(dim_t i = 0; i < A.rows(); ++i)
  {
    for(dim_t j = 0; j < A.cols(); ++j)
    {
      if(abs(B.getEntry(i,j) - A(i,j)) > .01)
        num_differences++;
    }
  }

  return num_differences;
}

////////////////////////////////////////////////////////////////////////////////
