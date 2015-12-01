/**
 * \file cholesky_eigen.h
 * \brief Functions for computing the cholesky decomp using Eigen library
 */

#ifndef _CHOLESKY_EIGEN_H_
#define _CHOLESKY_EIGEN_H_

#include "db/common.h"
#include <Eigen/Core>
#include <Eigen/LU>

/**
 * \fn void cholesky_eigen_serial(const Eigen::MatrixXf& A, 
 *                                Eigen::MatrixXf& L, 
 *                                Eigen::MatrixXf& D)
 * \brief Computes the Cholesky LDLT decomposition on Eigen matrix A.
 *        Decomposition is computed using a serial algorithm
 * \param A Input matrix to perform decomposition on
 * \param L Output matrix corresponding to L matrix of decomposition
 * \param D Output matrix corresponding to D matrix of decomposition
 */
void cholesky_eigen_serial(const Eigen::MatrixXf& A, 
                           Eigen::MatrixXf& L, 
                           Eigen::MatrixXf& D);

/**
 * \fn void cholesky_eigen_omp_v1(const Eigen::MatrixXf& A, 
 *                                Eigen::MatrixXf& L, 
 *                                Eigen::MatrixXf& D)
 * \brief Computes the Cholesky LDLT decomposition on Eigen matrix A.
 *        Decomposition is computed using ompenmp
 * \param A Input matrix to perform decomposition on
 * \param L Output matrix corresponding to L matrix of decomposition
 * \param D Output matrix corresponding to D matrix of decomposition
 */
void cholesky_eigen_omp_v1(const Eigen::MatrixXf& A, 
                           Eigen::MatrixXf& L, 
                           Eigen::MatrixXf& D);

/**
 * \fn void cholesky_eigen_omp_v1(const Eigen::MatrixXf& A, 
 *                                Eigen::MatrixXf& L, 
 *                                Eigen::MatrixXf& D)
 * \brief Computes the Cholesky LDLT decomposition on Eigen matrix A.
 *        Decomposition is computed using ompenmp
 * \param A Input matrix to perform decomposition on
 * \param L Output matrix corresponding to L matrix of decomposition
 * \param D Output matrix corresponding to D matrix of decomposition
 */
void cholesky_eigen_omp_v2(const Eigen::MatrixXf& A, 
                           Eigen::MatrixXf& L, 
                           Eigen::MatrixXf& D);

/**
 * \fn void cholesky_eigen_blocked_serial(const Eigen::MatrixXf& A, 
 *                                        Eigen::MatrixXf& L, 
 *                                        Eigen::MatrixXf& D)
 * \brief Computes the Cholesky LDLT decomposition on Eigen matrix A.
 *        Decomposition is computed using a serial algorithm. The matrices
 *        are considered as blocked matrices with size 2x2.
 * \param A Input matrix to perform decomposition on
 * \param L Output matrix corresponding to L matrix of decomposition
 * \param D Output matrix corresponding to D matrix of decomposition
 */
void cholesky_eigen_blocked_serial(const Eigen::MatrixXf& A, 
                                   Eigen::MatrixXf& L, 
                                   Eigen::MatrixXf& D);

/**
 * \fn void cholesky_eigen_blocked_serial(const Eigen::MatrixXf& A, 
 *                                        Eigen::MatrixXf& L, 
 *                                        Eigen::MatrixXf& D)
 * \brief Computes the Cholesky LDLT decomposition on Eigen matrix A.
 *        Decomposition is computed using openmp. The matrices
 *        are considered as blocked matrices with size 2x2.
 * \param A Input matrix to perform decomposition on
 * \param L Output matrix corresponding to L matrix of decomposition
 * \param D Output matrix corresponding to D matrix of decomposition
 */
void cholesky_eigen_blocked_omp_v1(const Eigen::MatrixXf& A, 
                                   Eigen::MatrixXf& L, 
                                   Eigen::MatrixXf& D);
#endif
