#include <stdlib.h>
#include <stdio.h>
#include <asl.h>

#define D_M 1.0
#define D_S 0.5

int main(int argc, char const *argv[])
{
    const int N = 3312;
    const int niter = 100000;
    int i, j;
    asl_random_t rng;

    /* Library Initialization */
    asl_library_initialize();
    /* Generator Preparation */
    asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
    asl_random_distribute_normal(rng, D_M, D_S);

    double *vals, *sum0, *sum;
    vals = (double *)aligned_alloc(128, N * sizeof(double));
    sum0 = (double *)aligned_alloc(128, N * sizeof(double));
    sum = (double *)aligned_alloc(128, N * sizeof(double));
    asl_random_generate_d(rng, N, vals);
    #pragma _NEC novector
    for (j = 0; j < niter; j++)
    {
        ftrace_region_begin("simple");
        sum0[0] = 0;
        for (i = 1; i < N; i++)
        {
            sum0[i] += sum0[i - 1]+vals[i];
        }
        ftrace_region_end("simple");
        double tmp;
        ftrace_region_begin("r1");
        sum[0] = 0;
        tmp=sum[0];
        for (i = 1; i < N; i++)
        {
            sum[i] += tmp+vals[i];
            tmp = sum[i];
        }
        ftrace_region_end("r1");

    }

    /* Generator Finalization */
    asl_random_destroy(rng);
    /* Library Finalization */
    asl_library_finalize();

    return 0;
}
