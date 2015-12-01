#ifndef _COMMON_H_
#define _COMMON_H_

#include "db/band_matrix_type.h"

BandMatrix  createSymmetricPositiveDefiniteBandMatrix ( const dim_t matDim, const dim_t bandWidth );

BandMatrix  createEmptyBandMatrix(const dim_t matDim, const dim_t bandWidth);

/**
 * \fn void cholesky_band_serial(const BandMatrix& A, 
 *                               BandMatrix& L, 
 *                               BandMatrix& D);
 * \brief Computes the Cholesky LDLT decomposition on band matrix A.
 *        Decomposition is computed using a serial algorithm
 * \param A Input matrix to perform decomposition on
 * \param L Output matrix corresponding to L matrix of decomposition
 * \param D Output matrix corresponding to D matrix of decomposition
 */
void cholesky_band_serial(const BandMatrix& A, BandMatrix& L, BandMatrix& D);


#endif
