#ifndef _COMMON_H_
#define _COMMON_H_

#include <algorithm>
#include <cmath>

#include "db/band_matrix_type.h"

#define EPSILON 0.01

BandMatrix  createSymmetricPositiveDefiniteBandMatrix ( const dim_t matDim, const dim_t bandWidth );

//BandMatrix  createEmptyBandMatrix(const dim_t matDim, const dim_t bandWidth);

BandMatrix  createEmptyBandMatrix( const dim_t matDim, const dim_t lowerBand, const dim_t upperBand );

bool    checkBandMatrixEqual( BandMatrix & m1, BandMatrix & m2 );

/*-----------------------------------------------------------------------------
 *  Version 0 : Lazy Man
 *  Version 1 : Handling of Index
 *  Version 2 : OMP
 *  Version 3 : CUDA
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
void cholesky_band_serial(const BandMatrix& A, BandMatrix& L, BandMatrix& D); /* cholesky_band.cpp */

void cholesky_band_serial_index_handling( const BandMatrix & A, BandMatrix & L, BandMatrix & D ); /* cholesky_band.cpp */

void cholesky_band_parallel_omp( const BandMatrix & A, BandMatrix & L, BandMatrix & D );

void cholesky_band_parallel_cuda( const BandMatrix & A, BandMatrix & L, BandMatrix & D ); /* cholesky_band_cuda.cu */

#endif
