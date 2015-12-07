/**
 * \file cholesky_band.h
 * \brief Functions for computing the cholesky decomp using band matrices
 */

#include "db/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

////////////////////////////////////////////////////////////////////////////////
void cholesky_band_serial(const BandMatrix& A, BandMatrix& L, BandMatrix& D)
{
  assert(A._matDim == L._matDim && L._matDim == D._matDim);

  // calculate decomposition
  for(dim_t j = 0; j < A._matDim; ++j)
  {
    // compute D values
    data_t value = A.getEntry(j,j);
    for(dim_t k = 0; k < j; ++k)
    {
      value -= D.getEntry(k,k) * L.getEntry(j,k) * L.getEntry(j,k);
    }
    D.writeEntry(j,j,value);

    // compute L values
    L.writeEntry(j, j, 1);
    for(dim_t i = j + 1; i < A._matDim; ++i)
    {
      data_t value = A.getEntry(i,j);
      for(dim_t k = 0; k < j; ++k)
      {
        value -= L.getEntry(i,k) * L.getEntry(j,k) * D.getEntry(k,k);
      }
      value /= D.getEntry(j,j);
      L.writeEntry(i,j,value);
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void cholesky_band_serial_index_handling(const BandMatrix& A, BandMatrix& L, BandMatrix& D)
{
  assert(A._matDim == L._matDim && L._matDim == D._matDim);

  // calculate decomposition
  for(dim_t j = 0; j < A._matDim; ++j)
  {
    // compute D values
    data_t value = A.getEntry(j,j);
//    for(dim_t k = 0; k < j; ++k)              /* CHANGED */
    for(dim_t k = std::max( 0, j-A._lowerBand ); k < j; ++k)
    {
      value -= D.getEntry(k,k) * L.getEntry(j,k) * L.getEntry(j,k);
    }
    D.writeEntry(j,j,value);

    // compute L values
    L.writeEntry(j, j, 1);
    for(dim_t i = j + 1; i < A._matDim; ++i)
    {
      data_t value = A.getEntry(i,j);
//      for(dim_t k = 0; k < j; ++k)            /* CHANGED */
      for(dim_t k = std::max( 0, j-A._lowerBand ); k < j; ++k)
      {
        value -= L.getEntry(i,k) * L.getEntry(j,k) * D.getEntry(k,k);
      }
      value /= D.getEntry(j,j);
      L.writeEntry(i,j,value);
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version (index handling)]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
void cholesky_band_serial_index_handling_omp_v1(const BandMatrix& A, BandMatrix& L, BandMatrix& D)
{
  assert(A._matDim == L._matDim && L._matDim == D._matDim);


  // calculate decomposition
  for(dim_t j = 0; j < A._matDim; ++j)
  {
    // compute D values
    data_t value = A.getEntry(j,j);
    for(dim_t k = std::max( 0, j-A._lowerBand ); k < j; ++k)
    {
      value -= D.getEntry(k,k) * L.getEntry(j,k) * L.getEntry(j,k);
    }
    D.writeEntry(j,j,value);

    // compute L values
    L.writeEntry(j, j, 1);
    #pragma omp parallel for schedule(dynamic)
    for(dim_t i = j + 1; i < A._matDim; ++i)
    {
      data_t value = A.getEntry(i,j);
      for(dim_t k = std::max( 0, j-A._lowerBand ); k < j; ++k)
      {
        value -= L.getEntry(i,k) * L.getEntry(j,k) * D.getEntry(k,k);
      }
      value /= D.getEntry(j,j);
      L.writeEntry(i,j,value);
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version (index handling)]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
void cholesky_band_serial_index_handling_omp_v2(const BandMatrix& A, BandMatrix& L, BandMatrix& D)
{
  assert(A._matDim == L._matDim && L._matDim == D._matDim);
  dim_t step = 10;

  // calculate decomposition
  for(dim_t j = 0; j < A._matDim; ++j)
  {
    // compute D values
    data_t d_value = A.getEntry(j,j);
    for(dim_t k = std::max( 0, j-A._lowerBand ); k < j; ++k)
    {
      d_value -= D.getEntry(k,k) * L.getEntry(j,k) * L.getEntry(j,k);
    }

    D.writeEntry(j,j,d_value);

    // compute L values
    L.writeEntry(j, j, 1);
    #pragma omp parallel for schedule(dynamic)
    for(dim_t i = j + 1; i < A._matDim; i+= step)
    {
      for(dim_t t_i = i; t_i < i + step; ++t_i) {
        data_t l_value = A.getEntry(t_i,j);
        for(dim_t k = std::max( 0, j-A._lowerBand ); k < j; ++k)
        {
          l_value -= L.getEntry(t_i,k) * L.getEntry(j,k) * D.getEntry(k,k);
        }
        l_value /= D.getEntry(j,j);
        L.writeEntry(t_i,j,l_value);
      }
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version (index handling)]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////
