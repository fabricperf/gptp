#include "gptp.cuh"

#ifdef GPTP_USE_MPI
#include <mpi.h>
#endif

__global__ void example_kernel() {
    gptp_sync_global();
}

static int get_gptp_runs() {
    const char *env = getenv("GPTP_RUNS");
    if (env == nullptr || env[0] == '\0') {
        env = getenv("PTP_RUNS");
    }
    if (env == nullptr || env[0] == '\0') {
        return 100;
    }
    const int value = atoi(env);
    return value > 0 ? value : 100;
}

int main(int argc, char **argv) {
    dim3 grid(16);
    dim3 block(128);
    const int ptp_runs = get_gptp_runs();

#ifdef GPTP_USE_MPI
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    (void) world_rank;
    (void) world_size;

    for (int run = 0; run < ptp_runs; run++) {
        prepareGptpMpi(MPI_COMM_WORLD);
        __CHECK_CUDA(cudaMemcpyToSymbol(gptpRunId, &run, sizeof(int)));
        example_kernel<<<grid, block>>>();
        __CHECK_CUDA(cudaGetLastError());
        __CHECK_CUDA(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        cleanGptp();
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
#else
    for (int run = 0; run < ptp_runs; run++) {
        prepareGptp();
        __CHECK_CUDA(cudaMemcpyToSymbol(gptpRunId, &run, sizeof(int)));
        example_kernel<<<grid, block>>>();
        __CHECK_CUDA(cudaGetLastError());
        __CHECK_CUDA(cudaDeviceSynchronize());
        cleanGptp();
    }

    return 0;
#endif
}
