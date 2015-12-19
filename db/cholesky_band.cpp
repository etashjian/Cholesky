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
/**
 * Struct for passing calculation requests to tasks
 */
typedef struct Pos_t
{
  dim_t _i; // starting i position
  dim_t _j; // starting j position
  dim_t _stride; // number of entries to calculate

  Pos_t(dim_t i, dim_t j, dim_t stride) : _i(i), _j(j), _stride(stride) {}
} Pos;

typedef queue<Pos> PosQueue; // Queues for communicating between threads

////////////////////////////////////////////////////////////////////////////////
/**
 * Basic serial algorithm. Compute all entries of L.
 */
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
/**
 * Functions for computing L and D matrix entiries
 */
void compute_L_entry(const BandMatrix& A, BandMatrix& L, BandMatrix& D, dim_t i, dim_t j);
void compute_D_entry(const BandMatrix& A, BandMatrix& L, BandMatrix& D, dim_t j);


static inline void compute_L_entry(const BandMatrix *A, BandMatrix *L, BandMatrix *D, dim_t i, dim_t j)
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

static inline void compute_D_entry(const BandMatrix *A, BandMatrix *L, BandMatrix *D, dim_t j)
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

////////////////////////////////////////////////////////////////////////////////
/**
 * Serial algorithm, optimized indexing for band matrices
 */
void cholesky_band_serial_index_handling(const BandMatrix& A, BandMatrix& L, BandMatrix& D)
{
  assert(A._matDim == L._matDim && L._matDim == D._matDim);

  // calculate decomposition
  for(dim_t j = 0; j < A._matDim; ++j)
  {
    // compute D values
    compute_D_entry(&A, &L, &D, j);

    // compute L values
    L.writeEntry(j, j, 1);
    for(dim_t i = j + 1; i < A._matDim; ++i)
    {
      compute_L_entry(&A, &L, &D, i, j);
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version (index handling)]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/**
 * First try at openmp. just throw in a for pragma
 */
void cholesky_band_serial_index_handling_omp_v1(const BandMatrix& A, BandMatrix& L, BandMatrix& D)
{
  assert(A._matDim == L._matDim && L._matDim == D._matDim);

  // calculate decomposition
  for(dim_t j = 0; j < A._matDim; ++j)
  {
    // compute D values
    compute_D_entry(&A, &L, &D, j);

    // compute L values
    L.writeEntry(j, j, 1);
    #pragma omp parallel for schedule(dynamic)
    for(dim_t i = j + 1; i <= j + A._lowerBand; ++i)
    {
      compute_L_entry(&A, &L, &D, i, j);
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version (index handling)]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/**
 * Second attemp, a more sophisticated version using for pragma
 */
void cholesky_band_serial_index_handling_omp_v2(const BandMatrix& A, BandMatrix& L, BandMatrix& D)
{
  assert(A._matDim == L._matDim && L._matDim == D._matDim);

  omp_set_nested(1);

  // calculate decomposition
  #pragma omp parallel
  {
    for(dim_t j = 0; j < A._matDim; ++j)
    {
      #pragma omp single
      {
        // compute D values
        compute_D_entry(&A, &L, &D, j);
      }
      #pragma omp barrier

      // compute L values
      L.writeEntry(j, j, 1);
      #pragma omp for schedule(dynamic)
      for(dim_t i = j + 1; i <= j + A._lowerBand; ++i)
      {
        compute_L_entry(&A, &L, &D, i, j);
      }

      #pragma omp barrier
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version (index handling)]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

/**
 * Function for tasks computing L and D values
 */
void consumer(const BandMatrix *A, BandMatrix *L, BandMatrix *D, PosQueue *D_q, PosQueue *L_q, unsigned *D_dq, unsigned *L_dq, unsigned *D_sched, bool *done)
{
  Pos pos(0,0,0);
  bool valid_L = false, valid_D = false;

  //printf("Thread %d starting\n", omp_get_thread_num());

  while(!*done)
  {
    // check for D requests
    #pragma omp critical
    {
      if(!D_q->empty())
      {
        pos = D_q->front();
        D_q->pop();
        valid_D = true;
      }
    }

    // otherwise check for L requests
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

    // compute D values
    if(valid_D)
    {
      compute_D_entry(A, L, D, pos._j);
      #pragma omp critical
      (*D_dq)++;
    }
    // compute L values
    else if(valid_L)
    {
      for(dim_t i = pos._i; i < pos._i + pos._stride; ++i)
        if(i < A->_matDim)
          compute_L_entry(A, L, D, i, pos._j);

      #pragma omp critical
      (*L_dq) += pos._stride;

      if(pos._i == pos._j + 1) {
        #pragma omp critical
        (*D_sched) += 1;
      }
    }

    valid_L = false;
    valid_D = false;
  }

  //printf("Thread %d done!\n", omp_get_thread_num());
}

/**
 * Third attempt, producer/consumer model scheduling
 */
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

      unsigned L_dq = 0, D_dq = 0, D_sched = 0;

      // launch consumer tasks
      for(dim_t i = 0; i < num_threads - 1; ++i)
      {
        #pragma omp task shared(A, L, D, D_q, L_q, D_dq, D_sched, L_dq, done)
        consumer(&A, &L, &D, &D_q, &L_q, &D_dq, &L_dq, &D_sched, &done);
      }

      unsigned D_count = 0, L_count = 0;
      for(dim_t j = 0; j < A._matDim; ++j)
      {
        // compute D values
        #pragma omp critical
        D_q.push(Pos(j,j, 1)); D_count++;

        // wait for L values of previous column to be ready
        bool L_wait = true;
        while(L_wait)
        {
          #pragma omp critical
          L_wait = L_dq < L_count;
        }

        // wait for D value to be ready
        bool D_wait = true;
        while(D_wait)
        {
          #pragma omp critical
          D_wait = D_dq < D_count;
        }

        // compute L values
        L.writeEntry(j, j, 1);
        for(dim_t i = j + 1; i <= j + A._lowerBand; i += stride)
        {
          #pragma omp critical
          L_q.push(Pos(i,j,stride)); L_count+=stride;
        }

        // Wait until enough L values to calculate next D have been computed
        bool D_ready = false;
        while(!D_ready)
        {
          #pragma omp critical
          D_ready = D_sched == D_count;
        }
      }

      // finish up
      done = true;
      #pragma omp taskwait
    }
  }

#ifdef ENABLE_LOG
  std::cout << "cholesky on band matrix finishes... [serial version (index handling)]\n";
#endif

}

////////////////////////////////////////////////////////////////////////////////
