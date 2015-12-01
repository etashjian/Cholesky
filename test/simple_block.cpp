#include <db/eigen_util.h>
#include <db/cholesky_eigen.h>
#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  struct timespec serial_start, serial_end, omp_start, omp_end;
  double time_elapsed;
  unsigned dim = 2048;
  unsigned bandwidth = 3;
  unsigned range = 5;

  // generate random input matrix (THIS PROBABLY ISN'T TOTALL CORRECT YET)
  cout << "Generating input... " << flush;
  srand(time(NULL));
  MatrixXf A = create_banded_sym_pos_def_eigen_matrix(dim, bandwidth, range);
  cout << "done" << endl;

  // allocate L and D and compute decomp
  cout << "Computing serial decomposition... " << flush;
  MatrixXf L = MatrixXf::Zero(dim, dim);
  MatrixXf D = MatrixXf::Zero(dim, dim);
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &serial_start);
  cholesky_eigen_blocked_serial(A, L, D);
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &serial_end);
  time_elapsed = serial_end.tv_sec + serial_end.tv_nsec / 1e9 -
                 serial_start.tv_sec - serial_start.tv_nsec / 1e9;
  time_elapsed *= 1000; // convert to ms
  cout << time_elapsed << endl;

  // calculate result
  cout << "Computing reference result... " << flush;
  MatrixXf result = L * D * L.transpose();
  cout << "done" << endl;

  // check results
  cout << "Comparing results... " << flush;
  int differences = cmp_matrices(result, A);
  if(differences == 0)
  {
    cout << "PASSED!" << endl;
  }
  else
  {
    cout << "FAILED! - " << differences << " differences" << endl;
  }

  // allocate L and D and compute decomp
  cout << "Computing openmp decomposition... " << flush;
  L = MatrixXf::Zero(dim, dim);
  D = MatrixXf::Zero(dim, dim);
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &omp_start);
  cholesky_eigen_blocked_omp_v1(A, L, D);
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &omp_end);
  time_elapsed = omp_end.tv_sec + omp_end.tv_nsec / 1e9 -
                 omp_start.tv_sec - omp_start.tv_nsec / 1e9;
  time_elapsed *= 1000; // convert to ms
  cout << time_elapsed << endl;

  // calculate result
  cout << "Computing reference result... " << flush;
  result = L * D * L.transpose();
  cout << "done" << endl;

  // check results
  cout << "Comparing results... " << flush;
  differences = cmp_matrices(result, A);
  if(differences == 0)
  {
    cout << "PASSED!" << endl;
  }
  else
  {
    cout << "FAILED! - " << differences << " differences" << endl;
  }
}
