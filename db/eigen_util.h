/**
 * \file eigen_util.h
 * \brief General functions for working with Eigen matrices
 */

#ifndef _MATRIXGEN_EIGEN_H_
#define _MATRIXGEN_EIGEN_H_

#include "db/band_matrix_type.h"
#include "db/common.h"
#include <Eigen/Core>

/**
 * \fn Eigen::MatrixXf create_sym_pos_def_eigen_matrix(const dim_t mat_dim, 
 *                                                     const int range);
 * \brief Creates a square Eigen matrix that is symetric positive definite
 * \param mat_dim Dimension of square matrix to build
 * \param range Range of values to use, from -range to range
 * \return Generated symetric positive definite matrix
 */
Eigen::MatrixXf create_sym_pos_def_eigen_matrix(const dim_t mat_dim, 
                                                const int range);

/**
 * \fn Eigen::MatrixXf create_banded_sym_pos_def_eigen_matrix(
                                                       const dim_t mat_dim, 
 *                                                     const dim_t bandwidth, 
 *                                                     const int range);
 * \brief Creates a square banded Eigen matrix that is symetric positive 
 *        definite
 * \param mat_dim Dimension of square matrix to build
 * \param bandwidth Width of band in matrix
 * \param range Range of values to use, from -range to range
 * \return Generated banded symetric positive definite matrix
 */
Eigen::MatrixXf create_banded_sym_pos_def_eigen_matrix(const dim_t mat_dim, 
                                                       const dim_t bandwidth, 
                                                       const int range);

/**
 * \fn Eigen::MatrixXf band_to_eigen(const BandMatrix& mat);
 * \brief Converts a BandMatrix to Eigen::MatirxXf format
 * \param mat BandMatrix to convert
 * \return Eigen::MatrixXf representing input BandMatrix
 */
Eigen::MatrixXf band_to_eigen(const BandMatrix& mat);

/**
 * \fn unsigned cmp_matrices(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B)
 * \brief Compares to eigen matrices and returns the number of differences
 * \param A First MatrixXf to compare
 * \param B Second MatrixXf to compare
 * \return Number of differences between A and B
 */
unsigned cmp_matrices(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B);

/**
 * \fn unsigned cmp_matrices(const Eigen::MatrixXf& A, const BandMatrixXf& B)
 * \brief Compares to eigen matrices and returns the number of differences
 * \param A MatrixXf to compare
 * \param B BandMatrix to compare
 * \return Number of differences between A and B
 */
unsigned cmp_matrices(const Eigen::MatrixXf& A, const BandMatrix& B);

#endif
