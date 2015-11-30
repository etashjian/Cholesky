/**
 * \file matrixgen_eigen.h
 * \brief General functions for generating random matrices using the Eigen
 *        library
 */

#ifndef _MATRIXGEN_EIGEN_H_
#define _MATRIXGEN_EIGEN_H_

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

#endif
