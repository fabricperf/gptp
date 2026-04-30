# GPTP

Standalone header-only GPU-PTP Clock Synchronization

## Files

- `gptp.cuh`: single-header GPTP device and host runtime.
- `main.cu`: GPTP-driven example that emits `[GPTP]` timing lines.
- `analyze_ptp_metrics.py`: summarizes `[GPTP]` and legacy `[PTP]` logs.

## Build

Requires CUDA. MPI builds also require `mpicxx`.

```bash
make single
make mpi
```

## Run

```bash
GPTP_RUNS=1 ./main
GPTP_RUNS=1 mpirun -n 2 ./main_mpi
python3 analyze_ptp_metrics.py ptp.log
```
