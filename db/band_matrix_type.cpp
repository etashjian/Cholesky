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

dim_t BandMatrix::getEntryIdx( const dim_t row, const dim_t col ) const {
    assert( row-col <= _lowerBand && row-col >= -_upperBand );
    const dim_t loc = col * (_upperBand+_lowerBand+1) + (row-col) + _upperBand;
    assert( loc >= 0 && loc < (dim_t)_vals.size() );
    return loc;
}

dim_t BandMatrix::getNumNonZeroEntries() const {
    return _matDim * (_upperBand+_lowerBand+1);
}

void    BandMatrix::writeEntry( const dim_t row, const dim_t col, const data_t val ) {
    if( row <0 || row >= _matDim ) return;
    if( col <0 || col >= _matDim ) return;
    if( row-col > _lowerBand || row-col < -_upperBand ) {
        return;
    } else {
        _vals[ getEntryIdx( row, col ) ] = val;
    }
}

void    BandMatrix::printBandMatrix() {
    printf( "Printing Matrix : %d x %d\n", _matDim, _matDim );
    for( dim_t row = 0; row < _matDim; row++ ) {
        for( dim_t col = 0; col < _matDim; col++ ) {
            printf( "%10.1e", getEntry( row, col ) );
        }
        printf( "\n" );
    }
}
