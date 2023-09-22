#include "osm.h"
#include <sys/time.h>
#include <iostream>

#define UNROLLING_NUM 10
#define TO_ADD 2
#define SEC_TO_NSEC 1000000000
#define USEC_TO_NSEC 1000
using namespace std;

/* Time measurement function for a simple arithmetic operation.
   returns time in nano-seconds upon success,
   and -1 upon failure.
   */
double osm_operation_time (unsigned int iterations)
{
  if (iterations == 0)
  {
    return -1;
  }
  double var = 0;
  struct timeval begin{}, end{};
  gettimeofday (&begin, nullptr);
  for (unsigned int i = 0; i < iterations; i += UNROLLING_NUM)
  {
    var = TO_ADD + TO_ADD;
    var = TO_ADD + TO_ADD;
    var = TO_ADD + TO_ADD;
    var = TO_ADD + TO_ADD;
    var = TO_ADD + TO_ADD;
    var = TO_ADD + TO_ADD;
    var = TO_ADD + TO_ADD;
    var = TO_ADD + TO_ADD;
    var = TO_ADD + TO_ADD;
    var = TO_ADD + TO_ADD;
  }
  gettimeofday (&end, nullptr);
  double microseconds =
      ((double) end.tv_usec - (double) begin.tv_usec) * USEC_TO_NSEC;
  double seconds = ((double) end.tv_sec - (double) begin.tv_sec) * SEC_TO_NSEC;
  double result = (microseconds + seconds) / iterations;
  (void)var;    // used to avoid warning
  return result;

}

void empty_func ()
{}

/* Time measurement function for an empty function call.
   returns time in nano-seconds upon success,
   and -1 upon failure.
   */
double osm_function_time (unsigned int iterations)
{
  if (iterations == 0)
  {
    return -1;
  }
  struct timeval begin{}, end{};
  gettimeofday (&begin, nullptr);
  for (unsigned int i = 0; i < iterations; i += UNROLLING_NUM)
  {
    empty_func ();
    empty_func ();
    empty_func ();
    empty_func ();
    empty_func ();
    empty_func ();
    empty_func ();
    empty_func ();
    empty_func ();
    empty_func ();
  }
  gettimeofday (&end, nullptr);
  double microseconds =
      ((double) end.tv_usec - (double) begin.tv_usec) * USEC_TO_NSEC;
  double seconds = ((double) end.tv_sec - (double) begin.tv_sec) * SEC_TO_NSEC;
  double result = (microseconds + seconds) / iterations;
  return result;
}

/* Time measurement function for an empty trap into the operating system.
   returns time in nano-seconds upon success,
   and -1 upon failure.
   */
double osm_syscall_time (unsigned int iterations)
{
  if (iterations == 0)
  {
    return -1;
  }
  struct timeval begin{}, end{};
  gettimeofday (&begin, nullptr);
  for (unsigned int i = 0; i < iterations; i += UNROLLING_NUM)
  {
    OSM_NULLSYSCALL;
    OSM_NULLSYSCALL;
    OSM_NULLSYSCALL;
    OSM_NULLSYSCALL;
    OSM_NULLSYSCALL;
    OSM_NULLSYSCALL;
    OSM_NULLSYSCALL;
    OSM_NULLSYSCALL;
    OSM_NULLSYSCALL;
    OSM_NULLSYSCALL;
  }
  gettimeofday (&end, nullptr);
  double microseconds =
      ((double) end.tv_usec - (double) begin.tv_usec) * USEC_TO_NSEC;
  double seconds = ((double) end.tv_sec - (double) begin.tv_sec) * SEC_TO_NSEC;
  double result = (microseconds + seconds) / iterations;
  return result;
}
