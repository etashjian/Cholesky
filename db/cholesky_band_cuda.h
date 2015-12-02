#ifndef _CHOLESKY_CUDA_H_
#define _CHOLESKY_CUDA_H_

struct CholeskySolver {

    CholeskySolver( const BandMatrix & A, BandMatrix & L, BandMatrix & D ) : _A(A), _L(L), _D(D) {};
    BandMatrix getL() { return _L; }
    BandMatrix getD() { return _D; }
    void choleskyDecomposition();
    
    private:
    void choleskySolvingColumn( const dim_t colIdx );
    BandMatrix _A, _L, _D;
};				/* ----------  end of struct CholeskySolver  ---------- */

typedef struct CholeskySolver CholeskySolver;

#endif
