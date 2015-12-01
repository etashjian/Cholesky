#ifndef _COMMON_H_
#define _COMMON_H_

#include "db/band_matrix_type.h"

BandMatrix  createSymmetricPositiveDefiniteBandMatrix ( const dim_t matDim, const dim_t bandWidth );

BandMatrix  createEmptyBandMatrix(const dim_t matDim, const dim_t bandWidth);

#endif
