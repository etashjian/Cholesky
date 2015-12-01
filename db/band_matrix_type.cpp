#include "db/band_matrix_type.h"

data_t BandMatrix::getEntry ( const dim_t row, const dim_t col ) const
{
    assert( row >=0 && row < _matDim );
    assert( col >=0 && col < _matDim );

    if( row-col > _lowerBand || row-col < -_upperBand ) {
        return 0;
    } else {
        return _vals[ getEntryIdx( row, col ) ];
    }
}		

void    BandMatrix::writeEntry( const dim_t row, const dim_t col, const data_t val ) {
    if( row <0 || row >= _matDim ) return;
    if( col <0 || col >= _matDim ) return;
    if( row-col > _lowerBand || row-col < -_upperBand ) {
        return;
    } else {
//        std::cout << "write : " << row << " " << col << std::endl;
        _vals[ getEntryIdx( row, col ) ] = val;
    }
}
