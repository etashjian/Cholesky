#ifndef _BAND_MAT_H_
#define _BAND_MAT_H_

typedef int dim_t;
typedef float data_t;

#include <vector>
#include <cassert>
#include <iostream>

struct BandMatrix {

    /*-----------------------------------------------------------------------------
     *  get the entry with given row (>=0) and col (>=0)
     *-----------------------------------------------------------------------------*/
    data_t              getEntry( const dim_t row, const dim_t col ) const; 
    inline dim_t        getEntryIdx( const dim_t row, const dim_t col ) const {
        assert( row-col <= _lowerBand && row-col >= -_upperBand );
        const dim_t loc = col * (_upperBand+_lowerBand+1) + (row-col) + _upperBand;
        if( loc < 0 || loc >= (dim_t)_vals.size() ) {
            std::cout << _matDim << " " << _lowerBand << " " << _upperBand << " " << loc << " " << row << " " << col << std::endl;
        }
        assert( loc >= 0 && loc < (dim_t)_vals.size() );
        return loc;
    }
    dim_t               getNumNonZeroEntries() const { 
        return _matDim * (_upperBand+_lowerBand+1);
    }
    void                writeEntry( const dim_t row, const dim_t col, const data_t val );

//    private:

    dim_t               _matDim;
    /*-----------------------------------------------------------------------------
     *  _lowerBand : the maximum difference between 
     *                  colId and rowId of non-zero entry below the diagonal
     *  _upperBand : the maximum difference between 
     *                  colId and rowId of non-zero entry above the diagonal
     *-----------------------------------------------------------------------------*/
    dim_t               _lowerBand, _upperBand;

    /*-----------------------------------------------------------------------------
     *  create a "_matDim by (_lowerBand+_upperBand+1)" 2-D array for data storage
     *  for example, _lowerBand =2 and _upperBand =1 and _matDim =10
     *
     *  Original Matrix : 
     *      A00 A01 0   ...                 0
     *      A10 A11 A12 0   ...             0 
     *      A20 A21 A22 A22 0   ...         0
     *      0   A31 A32 A33 A34 0   ...     0
     *      ...
     *      0   ...             A86 A87 A88 A89      
     *      0   ...                 A97 A98 A99      
     *
     *   Data Storage : 
     *      0   A00 A10 A20
     *      A01 A11 A21 A31
     *      ...
     *      A78 A88 A98 0
     *      A89 A99 0   0
     *
     *   Index :
     *      0   1   2   3
     *      4   5   6   7   
     *      ...
     *      36  37  38  39
     *-----------------------------------------------------------------------------*/
    std::vector<data_t> _vals;

};				/* ----------  end of struct BandMatrix  ---------- */

typedef struct BandMatrix BandMatrix;

#endif
