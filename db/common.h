#ifndef _COMMON_H_
#define _COMMON_H_

#include <algorithm>

#include "db/band_matrix_type.h"

BandMatrix  createSymmetricPositiveDefiniteBandMatrix ( const dim_t matDim, const dim_t bandWidth );

BandMatrix  createEmptyBandMatrix(const dim_t matDim, const dim_t bandWidth);



/*-----------------------------------------------------------------------------
 *  Version 0 : Lazy Man
 *  Version 1 : Handling of Index
 *-----------------------------------------------------------------------------*/

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

void cholesky_band_serial_index_handling( const BandMatrix & A, BandMatrix & L, BandMatrix & D );

#endif
