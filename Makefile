CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
MPICXX ?= mpicxx
MPICXX_BIN := $(firstword $(MPICXX))
HAVE_MPICXX := $(shell command -v $(MPICXX_BIN) >/dev/null 2>&1 && echo 1 || echo 0)

NVCUFLAGS ?= -O2
NVCC_GENCODE ?=
NVLDFLAGS ?= -lcuda

MPI_TARGET ?= main_mpi
SINGLE_TARGET ?= main
MPI_SOURCE ?= main_mpi.cu
SINGLE_SOURCE ?= main.cu
CUDA_GENERATED := \
	*.ii \
	*.cudafe1.c \
	*.cudafe1.cpp \
	*.cudafe1.gpu \
	*.cudafe1.stub.c \
	*.fatbin \
	*.fatbin.c \
	*.module_id \
	*.o \
	*.ptx \
	*.cubin \
	*_dlink.fatbin \
	*_dlink.fatbin.c \
	*_dlink.o \
	*_dlink.reg.c \
	*_dlink.sm_*.cubin
STALE_TARGETS := main_original_mpi

ifeq ($(HAVE_MPICXX),1)
DEFAULT_TARGET := $(MPI_TARGET)
else
DEFAULT_TARGET := $(SINGLE_TARGET)
endif

.PHONY: all single mpi clean

all: $(DEFAULT_TARGET)
ifeq ($(HAVE_MPICXX),0)
	@echo "MPI compiler '$(MPICXX_BIN)' not found; built $(SINGLE_TARGET) instead of $(MPI_TARGET)."
endif

$(MPI_TARGET): $(MPI_SOURCE)
ifeq ($(HAVE_MPICXX),1)
	$(NVCC) $(NVCUFLAGS) $(NVCC_GENCODE) -DGPTP_USE_MPI -ccbin $(MPICXX) -o $@ $(MPI_SOURCE) $(NVLDFLAGS)
else
	@echo "MPI compiler '$(MPICXX_BIN)' not found. Install MPI or override MPICXX to build $(MPI_TARGET)." >&2
	@false
endif

single: $(SINGLE_TARGET)

mpi: $(MPI_TARGET)

$(SINGLE_TARGET): $(SINGLE_SOURCE) gptp.cuh
	$(NVCC) $(NVCUFLAGS) $(NVCC_GENCODE) -o $@ $(SINGLE_SOURCE)

clean:
	rm -f $(MPI_TARGET) $(SINGLE_TARGET) $(STALE_TARGETS) $(CUDA_GENERATED)
