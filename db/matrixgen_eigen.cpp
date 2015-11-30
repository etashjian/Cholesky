#include "matrixgen_eigen.h"

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
