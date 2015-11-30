#include "db/band_matrix_type.h"


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  BandMatrix::getEntry
 *  Description:  
 * =====================================================================================
 */
    data_t

BandMatrix::getEntry ( const dim_t row, const dime col ) const
{
    assert( row >=0 && row < _matDim );
    assert( col >=0 && col < _matDim );

    if( row-col > _lowerBand || row-col < -_upperBand ) {
        return 0;
    } else {
        return _vals[ col * (_upperBand+_lowerBand+1) + (row-col)+_upperBand ];
    }
}		/* -----  end of function BandMatrix::getEntry  ----- */
