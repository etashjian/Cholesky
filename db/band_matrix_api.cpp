#include "db/common.h"

#include <ctime>
#include <cstdlib>
#include <iostream>

BandMatrix  createSymmetricPositiveDefiniteBandMatrix( const dim_t matDim, const dim_t bandWidth ) {
    assert( bandWidth < matDim );

    /*-----------------------------------------------------------------------------
     *  initialize the band matrix
     *-----------------------------------------------------------------------------*/
    BandMatrix  bandM;
    bandM._matDim = matDim;
    bandM._lowerBand = bandWidth;
    bandM._upperBand = bandWidth;
    bandM._vals.assign( matDim * (bandWidth*2+1), 0 );


    /*-----------------------------------------------------------------------------
     *  generate rand values for matrix entries
     *-----------------------------------------------------------------------------*/
    //std::cout << "dim = " << bandM._matDim << std::endl;
    srand( static_cast<unsigned>(time(0)) );    /* seed */
    for( dim_t i = 0; i < matDim; i++ ) {
        for( dim_t j = 0; j <= bandWidth; j++ ) { /* let diagnoal entires MUCH GREATER than the others */
            const data_t HI = (j==0) ? 20000 : 10;
            const data_t LO = (j==0) ? 10000 : -10;
            const data_t r = LO + static_cast<data_t>(rand()) / (static_cast<data_t>( RAND_MAX/(HI-LO) ) );
            bandM.writeEntry( i, i+j, r );
            if( j>0 )
                bandM.writeEntry( i+j, i, r );
        }
    }

    return bandM;
}

BandMatrix  createEmptyBandMatrix(const dim_t matDim, const dim_t lowerBand, const dim_t upperBand)
{
    /*-----------------------------------------------------------------------------
     *  initialize the band matrix
     *-----------------------------------------------------------------------------*/
    BandMatrix  bandM;
    bandM._matDim = matDim;
    bandM._lowerBand = lowerBand;
    bandM._upperBand = upperBand;
    bandM._vals.assign( matDim * (lowerBand+upperBand+1), 0 );

    return bandM;
}

bool    checkBandMatrixEqual( BandMatrix & m1, BandMatrix & m2 ) {
    if( m1._matDim != m2._matDim || m1._lowerBand != m2._lowerBand || m1._upperBand != m2._upperBand || m1._vals.size() != m2._vals.size() )
        return false;
    for( dim_t i = 0; i < m1.getNumNonZeroEntries(); i++ ) {
        if( std::fabs( m1._vals[i]-m2._vals[i] ) > EPSILON )
            return false;
    }
    return true;
}
