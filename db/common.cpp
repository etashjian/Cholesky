#include "db/common.h"

#include <ctime>
#include <cstdlib>

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
}
