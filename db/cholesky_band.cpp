/**
 * \file cholesky_band.h
 * \brief Functions for computing the cholesky decomp using band matrices
 */

#include "db/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <queue>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
typedef struct Pos_t
{
  dim_t _i;
  dim_t _j;
  dim_t _stride;

  Pos_t(dim_t i, dim_t j, dim_t stride) : _i(i), _j(j), _stride(stride) {}
} Pos;

typedef queue<Pos> PosQueue;

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

  omp_set_nested(1);

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
    for(dim_t i = j + 1; i < A._matDim; ++i)
    {
        data_t l_value = A.getEntry(i,j);
        #pragma omp parallel for reduction(+:l_value)
        for(dim_t k = std::max( 0, j-A._lowerBand ); k < j; ++k)
        {
          l_value -= L.getEntry(i,k) * L.getEntry(j,k) * D.getEntry(k,k);
        }
        l_value /= D.getEntry(j,j);
        L.writeEntry(i,j,l_value);
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version (index handling)]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

void compute_L_entry(const BandMatrix& A, BandMatrix& L, BandMatrix& D, dim_t i, dim_t j);
void compute_D_entry(const BandMatrix& A, BandMatrix& L, BandMatrix& D, dim_t j);


inline void compute_L_entry(const BandMatrix *A, BandMatrix *L, BandMatrix *D, dim_t i, dim_t j)
{

  if(i > j + A->_lowerBand) return;

  //printf("Computing L entry %d,%d on thread %d\n", i, j, omp_get_thread_num());

  if(i == j) {
    L->writeEntry(i, j, 1);
  }
  else
  {
    data_t value = A->getEntry(i,j);
    for(dim_t k = std::max( 0, j-A->_lowerBand ); k < j; ++k)
    {
      value -= L->getEntry(i,k) * L->getEntry(j,k) * D->getEntry(k,k);
    }
    value /= D->getEntry(j,j);
    L->writeEntry(i,j,value);
  }

  //printf("Done Computing L entry %d,%d on thread %d\n", i, j, omp_get_thread_num());
}

inline void compute_D_entry(const BandMatrix *A, BandMatrix *L, BandMatrix *D, dim_t j)
{
  //printf("Computing D entry %d,%d on thread %d\n", j, j, omp_get_thread_num());

  data_t value = A->getEntry(j,j);
  for(dim_t k = std::max( 0, j-A->_lowerBand ); k < j; ++k)
  {
    value -= D->getEntry(k,k) * L->getEntry(j,k) * L->getEntry(j,k);
  }
  D->writeEntry(j,j,value);

  //printf("Done Computing D entry %d,%d on thread %d\n", j, j, omp_get_thread_num());
}

void consumer(const BandMatrix *A, BandMatrix *L, BandMatrix *D, PosQueue *D_q, PosQueue *L_q, unsigned *D_dq, unsigned *L_dq, bool *done)
{
  Pos pos(0,0,0);
  bool valid_L = false, valid_D = false;

//  printf("Thread %d starting\n", omp_get_thread_num());

  while(!*done)
  {
    #pragma omp critical
    {
      if(!D_q->empty())
      {
        pos = D_q->front();
        D_q->pop();
        valid_D = true;
      }
    }

    if(!valid_D)
    {
      #pragma omp critical
      {
        if(!L_q->empty())
        {
          pos = L_q->front();
          L_q->pop();
          valid_L = true;
        }
      }
    }

    if(valid_D)
    {
      compute_D_entry(A, L, D, pos._j);
      #pragma omp critical
      (*D_dq)++;
    }
    else if(valid_L)
    {
      for(dim_t i = pos._i; i < pos._i + pos._stride; ++i)
        if(i < A->_matDim)
          compute_L_entry(A, L, D, i, pos._j);

      #pragma omp critical
      (*L_dq) += pos._stride;
    }

    valid_L = false;
    valid_D = false;
  }

//  printf("Thread %d done!\n", omp_get_thread_num());
}

void cholesky_band_serial_index_handling_omp_v3(const BandMatrix& A, BandMatrix& L, BandMatrix& D)
{
  assert(A._matDim == L._matDim && L._matDim == D._matDim);

  #pragma omp parallel
  {
    #pragma omp single
    {
      int num_threads = omp_get_num_threads();

      // create task queues
      bool done = false;
      PosQueue D_q, L_q;
      dim_t stride = 32;

      unsigned L_dq = 0, D_dq = 0;

      // launch consumer tasks
      for(dim_t i = 0; i < num_threads - 1; ++i)
      {
        #pragma omp task shared(A, L, D, D_q, L_q, D_dq, L_dq, done)
        consumer(&A, &L, &D, &D_q, &L_q, &D_dq, &L_dq, &done);
      }

      unsigned D_count = 0, L_count = 0;
      for(dim_t j = 0; j < A._matDim; ++j)
      {
        // compute D values
        #pragma omp critical
        D_q.push(Pos(j,j, 1)); D_count++;

        bool D_wait = true;
        while(D_wait)
        {
          #pragma omp critical
          D_wait = D_dq < D_count;
        }

        // compute L values
        L.writeEntry(j, j, 1);
        for(dim_t i = j + 1; i < A._matDim; i+= stride)
        {
          #pragma omp critical
          L_q.push(Pos(i,j,stride)); L_count+=stride;
        }

        bool L_wait = true;
        while(L_wait)
        {
          #pragma omp critical
          L_wait = L_dq < L_count;
        }
      }

      done = true;
      #pragma omp taskwait
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version (index handling)]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////
