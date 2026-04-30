/**
 * Precision Time Protocol (PTP) for GPU
 * Merged Single Header
 * Supported Platform: H100 DGX, GB200 NVL72
 */

#ifndef GPTP_CUH
#define GPTP_CUH

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#ifdef GPTP_USE_MPI
#include <string.h>
#include <cuda.h>
#include <mpi.h>
#endif

///////////////////////////////
// Macros & Constants
///////////////////////////////

#define CPUM (1<<4)
#define GPUM (2<<4)

#define LEADER_SPACES 3
#define FOLLOWER_SPACES 5
#define TIMESTAMP_SPACES 3
#define GPTP_SAMPLES_PER_LEADER 20

constexpr int LEADER_CONN_RESP = 0;
constexpr int LEADER_DELAY_REQ = 1;
constexpr int LEADER_DELAY_READY = 2;

constexpr int FOLLOWER_CONN_REQ = 0;
constexpr int FOLLOWER_SYNC = 1;
constexpr int FOLLOWER_FOLLOW_UP = 2;
constexpr int FOLLOWER_DELAY_RESP = 3;
constexpr int FOLLOWER_DELAY_GO = 4;

constexpr uint8_t SYNC = 0x1u;
constexpr uint8_t DELAY_REQ = 0x2u; 
constexpr uint8_t CONN_REQ = 0x5u; 
constexpr uint8_t CONN_RESP = 0x6u;
constexpr uint8_t FOLLOW_UP = 0x7u; 
constexpr uint8_t DELAY_RESP = 0x8u; 
constexpr uint8_t REPORT_RES = 0x3u; 
constexpr uint8_t BCAST_RES = 0x4u; 

constexpr int MAX_NUM_TIMES = 256; 
constexpr int MAX_CHANNEL = 16; 

///////////////////////////////
// Structures
///////////////////////////////

// For illustration only, use ulong2 with shift in production
typedef struct __attribute__((packed)) {
    uint8_t tsmt;       /* transportSpecific | messageType */
    uint8_t sequenceId; // sequenceIdx of message
    uint16_t gptpHostId;    // host ranks
    uint16_t gptpDeviceId;  // device ranks
    uint16_t coreId;    /* @note shall be blockIdx.x at the moment */
    uint64_t time;      // clock as u64
} gptp_message; 

///////////////////////////////
// Global Variables (Device)
///////////////////////////////

// Configuration
__constant__ int gptpHostId; 
__constant__ int gptpDeviceId; 
__constant__ int gptpNumDevices; 
__constant__ int gptpRunId;

// PTP Buffers
// sync 1st block on different GPUs
__constant__ uint64_t gptpGlobalFollowerBuffers; 
__constant__ uint64_t gptpGlobalLeaderBuffers; 
__constant__ uint64_t gptpGlobalResultBuffer; 

// Synchronization & Latency
__device__ volatile int gptpGlobalBarrier; 
__device__ volatile int gptpGlobalBarrierSense; 
__constant__ uint64_t gptpLatencyGlobal; 

///////////////////////////////
// Helper Functions
///////////////////////////////

#define __CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

__device__ __forceinline__ uint64_t __clock64() {
    uint64_t clock_value; 
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(clock_value));
    return clock_value;
}

__device__ __forceinline__ ulong2 __ld128(const ulong2* ptr) {
    ulong2 val; 
    asm volatile (
        "{\n\t"
        "    .reg .u64 p_global;\n\t"
        "    cvta.to.global.u64 p_global, %2;\n\t" 
        "    ld.global.volatile.v2.u64 {%0, %1}, [p_global];\n\t"
        "}"
        : "=l"(val.x), "=l"(val.y)  
        : "l"(ptr)                  
        : "memory"                                                  
    );
    return val; 
}

__device__ __forceinline__ void __st128(ulong2* ptr, ulong2 val) {
    asm volatile (
        "{\n\t"
        "    .reg .u64 p_global;\n\t"
        "    cvta.to.global.u64 p_global, %2;\n\t"
        "    st.global.volatile.v2.u64 [p_global], {%0, %1};\n\t"
        "}"
        :                                                   
        : "l"(val.x), "l"(val.y), "l"(ptr)                                                  
        : "memory"                                                  
    );
}

__device__ __forceinline__ void __syncgrid() {
    int num_blocks = gridDim.x;

    if (threadIdx.x == 0) {
        int sense = gptpGlobalBarrierSense;
        int ticket = atomicAdd((int *)&gptpGlobalBarrier, 1);

        if (ticket == num_blocks - 1) {
            gptpGlobalBarrier = 0;
            __threadfence();
            atomicAdd((int *)&gptpGlobalBarrierSense, 1);
        } else {
            while (gptpGlobalBarrierSense == sense) {
            }
        }
    }

    // Synchronize all threads within the block
    __syncthreads();
}        

__device__ __forceinline__ void memory_fence() {
    asm volatile("fence.sc.sys;" ::: "memory");
}

__device__ __forceinline__ void memory_fence_gpu() {
    asm volatile("fence.sc.gpu;" ::: "memory");
}

///////////////////////////////
// Message Packing
///////////////////////////////

__device__ __forceinline__ ulong2 prepare_message(uint8_t messageType, uint8_t sequenceId, uint64_t time) {
    uint8_t tmst = (GPUM & 0x0F) | ((messageType & 0x0F) << 4);
    ulong2 message; 
    message.x = (uint64_t)tmst |
                ((uint64_t)sequenceId  << 8) |
                ((uint64_t)gptpHostId      << 16) |
                ((uint64_t)gptpDeviceId    << 32) |
                ((uint64_t)blockIdx.x  << 48);
    message.y = time; 
    return message; 
}

__device__ __forceinline__ uint8_t get_messageType(ulong2 message) {
    return (uint8_t)((message.x >> 4) & 0x0F);
}
        
__device__ __forceinline__ uint16_t get_sequenceId(ulong2 message) {
    return (uint16_t)((message.x >> 8) & 0xFF);
}

__device__ __forceinline__ uint16_t get_deviceId(ulong2 message) {
    return (uint16_t)((message.x >> 32) & 0xFFFF);
}

__device__ __forceinline__ uint16_t get_coreId(ulong2 message) {
    return (uint16_t)((message.x >> 48) & 0xFFFF);
}

__device__ __forceinline__ uint64_t get_time(ulong2 message) {
    return message.y;
}

///////////////////////////////
// PTP Protocol Logic
///////////////////////////////

__device__ __forceinline__ void lead(ulong2* leaderBuff, ulong2* followerBuff) {
    //////////////////////////////
    // CONN 1 - leader -> follower
    //////////////////////////////

    // 1. Leader send a CONN_REQ message to corresponding follower
    ulong2 connreq_msg = prepare_message(CONN_REQ, 0, __clock64()); 
    __st128(&followerBuff[FOLLOWER_CONN_REQ], connreq_msg); 
    memory_fence(); // message sent

    // 2. Leader receivs a CONN_RESP message from corresponding follower
    ulong2 connresp_msg; 
    do { 
        connresp_msg = __ld128(&leaderBuff[LEADER_CONN_RESP]); 
    } while (get_messageType(connresp_msg) != CONN_RESP);

    uint64_t start = __clock64(); 

    //////////////////////////////
    // PTP 1 - SYNC & FOLLOW_UP
    //////////////////////////////

    // 1. Leader sends SYNC, then timestamps after a GPU-scope fence.
    ulong2 sync_msg = prepare_message(SYNC, 0, __clock64()); 
    __st128(&followerBuff[FOLLOWER_SYNC], sync_msg);
    memory_fence_gpu();
    uint64_t T1 = __clock64(); 

    // 2. Leader issues FOLLOW_UP with the same send timestamp.
    memory_fence(); // message sent
    ulong2 followup_msg = prepare_message(FOLLOW_UP, 1, T1); 
    __st128(&followerBuff[FOLLOWER_FOLLOW_UP], followup_msg);
    // Ablation backup: original system-scope send fence.
    // memory_fence(); // message sent
    memory_fence_gpu(); // message sent

    //////////////////////////////
    // CONN 2 - follower -> leader (disabled for ablation)
    //////////////////////////////

    // Ablation backup: original follower-initiated handshake before DELAY_REQ.
    // The follower sends ready first; the leader responds only when it is
    // entering receive.
    // ulong2 delay_ready_msg;
    // do {
    //     delay_ready_msg = __ld128(&leaderBuff[LEADER_DELAY_READY]);
    // } while (get_messageType(delay_ready_msg) != CONN_REQ);
    //
    // uint64_t Tgo = __clock64();
    // ulong2 delay_go_msg = prepare_message(REPORT_RES, 4, Tgo);
    // __st128(&followerBuff[FOLLOWER_DELAY_GO], delay_go_msg);
    // memory_fence_gpu(); // follower may now send DELAY_REQ

    //////////////////////////////
    // PTP 2 - DELAY_REQ & RESP
    //////////////////////////////
    // 4. Leader records receive time of DELAY_REQ 
    ulong2 delayreq_msg; 
    uint64_t T2p; 
    do { 
        delayreq_msg = __ld128(&leaderBuff[LEADER_DELAY_REQ]); 
        T2p = __clock64(); 
    } while (get_messageType(delayreq_msg) != DELAY_REQ);
            
    // 5. Leader send the receive time as DELAY_RESP
    ulong2 delayresp_msg = prepare_message(DELAY_RESP, 3, T2p); 
    __st128(&followerBuff[FOLLOWER_DELAY_RESP], delayresp_msg);
    memory_fence(); // message sent

    // clean buffer before go
    ulong2 zero = make_ulong2(0u, 0u);
    __st128(&leaderBuff[LEADER_CONN_RESP], zero);
    __st128(&leaderBuff[LEADER_DELAY_REQ], zero);
    __st128(&leaderBuff[LEADER_DELAY_READY], zero);
}

__device__ __forceinline__ long2 follow(ulong2* leaderBuff, ulong2* followerBuff) {
    /////////////////////
    // Connection Phase 
    /////////////////////

    // 1. Receiver receis a CONNREQ message from the leader
    ulong2 connreq_msg; 
    do { 
        connreq_msg = __ld128(&followerBuff[FOLLOWER_CONN_REQ]); 
    } while (get_messageType(connreq_msg) != CONN_REQ);

    // 2. Receiver sends a CONN_RESP message to the leader
    ulong2 connresp_msg = prepare_message(CONN_RESP, 0, __clock64()); 
    __st128(&leaderBuff[LEADER_CONN_RESP], connresp_msg); 
    memory_fence(); // message sent

    uint64_t start = __clock64(); 

    /////////////////////
    // PTP Phase 
    /////////////////////

    // 1. Follower listen for the SYNC message and tag T1' (T1p)
    ulong2 sync_msg; 
    uint64_t T1p; 
    do {
        sync_msg = __ld128(&followerBuff[FOLLOWER_SYNC]); 
        T1p = __clock64(); 
    } while (get_messageType(sync_msg) != SYNC); // message received

    // 2. Follower get the accurate T1 from FOLLOW_UP
    ulong2 followup_msg; 
    do {
        followup_msg = __ld128(&followerBuff[FOLLOWER_FOLLOW_UP]); 
    } while (get_messageType(followup_msg) != FOLLOW_UP); // message received
    uint64_t T1 = get_time(followup_msg); 

    // Ablation backup: original CONN 2 follower-side ready/go handshake.
    // ulong2 delay_ready_msg = prepare_message(CONN_REQ, 4, __clock64());
    // __st128(&leaderBuff[LEADER_DELAY_READY], delay_ready_msg);
    // memory_fence(); // delay phase ready
    //
    // ulong2 delay_go_msg;
    // do {
    //     delay_go_msg = __ld128(&followerBuff[FOLLOWER_DELAY_GO]);
    // } while (get_messageType(delay_go_msg) != REPORT_RES);

    // 3. Follower sends DELAY_REQ, then timestamps after a GPU-scope fence.
    ulong2 delayreq_msg = prepare_message(DELAY_REQ, 0, __clock64()); 
    __st128(&leaderBuff[LEADER_DELAY_REQ], delayreq_msg);
    memory_fence_gpu();
    uint64_t T2 = __clock64(); 
    memory_fence(); // message sent
    
    // 4. Follower get the recv time T2' (T2p) from DELAY_RESP
    ulong2 delayresp_msg;
    do {
        delayresp_msg = __ld128(&followerBuff[FOLLOWER_DELAY_RESP]); 
    } while (get_messageType(delayresp_msg) != DELAY_RESP); // message received
    uint64_t T2p = get_time(delayresp_msg); 

    // 5. Follower calculate latency and offset by formula
    long master_to_follower = (long)T1p - (long)T1;
    long follower_to_master = (long)T2p - (long)T2;
    long offset = (master_to_follower - follower_to_master) / 2; 
    long latency = (master_to_follower + follower_to_master) / 2;

    // clear the followerBuff
    ulong2 zero = make_ulong2(0u, 0u);
    __st128(&followerBuff[FOLLOWER_CONN_REQ], zero);
    __st128(&followerBuff[FOLLOWER_SYNC], zero);
    __st128(&followerBuff[FOLLOWER_FOLLOW_UP], zero);
    __st128(&followerBuff[FOLLOWER_DELAY_RESP], zero);
    __st128(&followerBuff[FOLLOWER_DELAY_GO], zero);

    // printf("[COST] %lu\n", __clock64() - start);

    return make_long2(offset, latency); 
}

///////////////////////////////
// Device API
///////////////////////////////

__device__ void gptp_sync(int seqId, int leaderId) {
    // sync across the devices
    /* Now the layout changes for followerBuff to discontinuous */
    ulong2 **leaderBuffs = reinterpret_cast<ulong2 **>(gptpGlobalLeaderBuffers); 
    ulong2 **followerBuffs = reinterpret_cast<ulong2 **>(gptpGlobalFollowerBuffers); 

    if (gptpDeviceId == leaderId) {
        if  (threadIdx.x == 0 && blockIdx.x != leaderId && blockIdx.x < gptpNumDevices) {
            const int connIdx = blockIdx.x; 
            ulong2 *leaderBuff = leaderBuffs[gptpDeviceId];
            ulong2 *followerBuff = followerBuffs[connIdx]; 
            lead(&leaderBuff[(seqId * gptpNumDevices + connIdx) * LEADER_SPACES], &followerBuff[seqId * FOLLOWER_SPACES]); 
            // printf("[LEAD] %d -> %d, %p, %p\n", leaderId, connIdx, &leaderBuff[(seqId * gptpNumDevices + connIdx) * LEADER_SPACES], &followerBuff[seqId * FOLLOWER_SPACES]);
        }
    } else {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            const int connIdx = gptpDeviceId; 
            ulong2 *leaderBuff = leaderBuffs[leaderId];
            ulong2 *followerBuff = followerBuffs[connIdx]; 
            long2 res = follow(&leaderBuff[(seqId * gptpNumDevices + connIdx) * LEADER_SPACES], &followerBuff[seqId * FOLLOWER_SPACES]); 
            // printf("[FOLO] %d -> %d, %p, %p\n", leaderId, connIdx, &leaderBuff[(seqId * gptpNumDevices + connIdx) * LEADER_SPACES], &followerBuff[seqId * FOLLOWER_SPACES]);
            printf("[GPTP] run %d %d -> %d offset %ld latency %ld\n", gptpRunId, leaderId, gptpDeviceId, res.x, res.y);
        }
    }
    __syncthreads(); 
} 

__device__ void gptp_sync_global() {
    // Iterates through all devices acting as leader
    for (int leaderId = 0; leaderId < gptpNumDevices; leaderId++) {
        for (int sample = 0; sample < GPTP_SAMPLES_PER_LEADER; sample++) {
            gptp_sync(GPTP_SAMPLES_PER_LEADER * leaderId + sample, leaderId);
        }
        __syncgrid(); 
    }
}

/////////////////////////
// Host API
/////////////////////////

// Static to allow inclusion in header
static int runIdx = 0; 

#ifdef GPTP_USE_MPI
static bool gptpMpiPrepared = false;
static bool gptpMpiReportedConfig = false;
static int gptpMpiOriginalDev = 0;
static int gptpMpiDevice = 0;
static int gptpMpiRank = -1;
static int gptpMpiWorldSize = 0;
static ulong2 *gptpMpiLocalLeaderBuff = nullptr;
static ulong2 *gptpMpiLocalFollowerBuff = nullptr;
static ulong2 **gptpMpiLeaderBuffs = nullptr;
static ulong2 **gptpMpiFollowerBuffs = nullptr;
static ulong2 **gptpMpiDeviceLeaderTable = nullptr;
static ulong2 **gptpMpiDeviceFollowerTable = nullptr;
static CUmemGenericAllocationHandle *gptpMpiLeaderHandles = nullptr;
static CUmemGenericAllocationHandle *gptpMpiFollowerHandles = nullptr;
static size_t gptpMpiLeaderAllocSize = 0;
static size_t gptpMpiFollowerAllocSize = 0;

static __host__ size_t gptpAlignUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

static __host__ void gptpCheckCu(CUresult err, MPI_Comm comm, const char *call,
                                       const char *file, int line) {
    if (err == CUDA_SUCCESS) {
        return;
    }

    const char *err_name = nullptr;
    const char *err_str = nullptr;
    cuGetErrorName(err, &err_name);
    cuGetErrorString(err, &err_str);
    fprintf(stderr, "CUDA Driver Error: %s (%s) from %s at %s line %d\n",
            err_name != nullptr ? err_name : "unknown",
            err_str != nullptr ? err_str : "unknown", call, file, line);
    MPI_Abort(comm, 1);
}

#define __CHECK_CU_MPI(call, comm) \
    gptpCheckCu((call), (comm), #call, __FILE__, __LINE__)

static __host__ void gptpCheckVmmSupport(CUdevice cu_dev, int world_rank,
                                               MPI_Comm comm) {
    int vmm_supported = 0;
    int fabric_supported = 0;
    __CHECK_CU_MPI(cuDeviceGetAttribute(&vmm_supported,
                                        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                        cu_dev), comm);
    __CHECK_CU_MPI(cuDeviceGetAttribute(&fabric_supported,
                                        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                                        cu_dev), comm);
    if (!vmm_supported || !fabric_supported) {
        fprintf(stderr,
                "GPTP MPI rank %d requires CUDA VMM and fabric handle support "
                "(vmm=%d fabric=%d)\n",
                world_rank, vmm_supported, fabric_supported);
        MPI_Abort(comm, 1);
    }
}

static __host__ void gptpMpiCreateFabricBuffer(size_t requested_size, CUdevice cu_dev,
                                                     MPI_Comm comm, ulong2 **ptr,
                                                     size_t *mapped_size,
                                                     CUmemGenericAllocationHandle *handle,
                                                     CUmemFabricHandle *fabric_handle) {
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cu_dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    size_t granularity = 0;
    __CHECK_CU_MPI(cuMemGetAllocationGranularity(&granularity, &prop,
                                                 CU_MEM_ALLOC_GRANULARITY_MINIMUM), comm);
    *mapped_size = gptpAlignUp(requested_size, granularity);

    __CHECK_CU_MPI(cuMemCreate(handle, *mapped_size, &prop, 0), comm);

    CUdeviceptr addr = 0;
    __CHECK_CU_MPI(cuMemAddressReserve(&addr, *mapped_size, 0, 0, 0), comm);
    __CHECK_CU_MPI(cuMemMap(addr, *mapped_size, 0, *handle, 0), comm);

    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    __CHECK_CU_MPI(cuMemSetAccess(addr, *mapped_size, &access, 1), comm);

    __CHECK_CU_MPI(cuMemExportToShareableHandle(fabric_handle, *handle,
                                                CU_MEM_HANDLE_TYPE_FABRIC, 0), comm);

    *ptr = reinterpret_cast<ulong2 *>(addr);
    __CHECK_CUDA(cudaMemset(*ptr, 0, *mapped_size));
}

static __host__ void gptpMpiImportFabricBuffer(const CUmemFabricHandle *fabric_handle,
                                                     size_t mapped_size, CUdevice cu_dev,
                                                     MPI_Comm comm, ulong2 **ptr,
                                                     CUmemGenericAllocationHandle *handle) {
    __CHECK_CU_MPI(cuMemImportFromShareableHandle(handle,
                                                  const_cast<CUmemFabricHandle *>(fabric_handle),
                                                  CU_MEM_HANDLE_TYPE_FABRIC), comm);

    CUdeviceptr addr = 0;
    __CHECK_CU_MPI(cuMemAddressReserve(&addr, mapped_size, 0, 0, 0), comm);
    __CHECK_CU_MPI(cuMemMap(addr, mapped_size, 0, *handle, 0), comm);

    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    __CHECK_CU_MPI(cuMemSetAccess(addr, mapped_size, &access, 1), comm);

    *ptr = reinterpret_cast<ulong2 *>(addr);
}

static __host__ void gptpMpiUnmapFabricBuffer(ulong2 *ptr, size_t mapped_size,
                                                    MPI_Comm comm) {
    if (ptr == nullptr) {
        return;
    }

    CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(ptr);
    __CHECK_CU_MPI(cuMemUnmap(addr, mapped_size), comm);
    __CHECK_CU_MPI(cuMemAddressFree(addr, mapped_size), comm);
}

__host__ void prepareGptpMpi(MPI_Comm comm) {
    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    if (world_size < 1 || world_size > MAX_CHANNEL) {
        fprintf(stderr, "GPTP MPI requires 1..%d ranks, got %d\n", MAX_CHANNEL, world_size);
        MPI_Abort(comm, 1);
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int local_device_count = 0;
    __CHECK_CUDA(cudaGetDeviceCount(&local_device_count));
    if (local_device_count <= 0) {
        fprintf(stderr, "GPTP MPI rank %d sees no CUDA devices\n", world_rank);
        MPI_Abort(comm, 1);
    }

    __CHECK_CUDA(cudaGetDevice(&gptpMpiOriginalDev));

    gptpMpiRank = world_rank;
    gptpMpiWorldSize = world_size;
    gptpMpiDevice = local_rank % local_device_count;
    __CHECK_CUDA(cudaSetDevice(gptpMpiDevice));

    __CHECK_CU_MPI(cuInit(0), comm);
    CUdevice cu_dev;
    __CHECK_CU_MPI(cuDeviceGet(&cu_dev, gptpMpiDevice), comm);
    gptpCheckVmmSupport(cu_dev, world_rank, comm);

    gptpMpiLeaderHandles = reinterpret_cast<CUmemGenericAllocationHandle*>(
        malloc(world_size * sizeof(CUmemGenericAllocationHandle)));
    gptpMpiFollowerHandles = reinterpret_cast<CUmemGenericAllocationHandle*>(
        malloc(world_size * sizeof(CUmemGenericAllocationHandle)));
    if (gptpMpiLeaderHandles == nullptr || gptpMpiFollowerHandles == nullptr) {
        fprintf(stderr, "GPTP MPI rank %d failed to allocate VMM handle arrays\n", world_rank);
        MPI_Abort(comm, 1);
    }
    memset(gptpMpiLeaderHandles, 0, world_size * sizeof(CUmemGenericAllocationHandle));
    memset(gptpMpiFollowerHandles, 0, world_size * sizeof(CUmemGenericAllocationHandle));

    CUmemFabricHandle localLeaderHandle;
    CUmemFabricHandle localFollowerHandle;
    const size_t sequence_slots = GPTP_SAMPLES_PER_LEADER * world_size;
    const size_t leader_requested_size =
        sequence_slots * world_size * LEADER_SPACES * sizeof(ulong2);
    const size_t follower_requested_size =
        sequence_slots * FOLLOWER_SPACES * sizeof(ulong2);
    gptpMpiCreateFabricBuffer(leader_requested_size, cu_dev, comm,
                                    &gptpMpiLocalLeaderBuff,
                                    &gptpMpiLeaderAllocSize,
                                    &gptpMpiLeaderHandles[world_rank],
                                    &localLeaderHandle);
    gptpMpiCreateFabricBuffer(follower_requested_size, cu_dev, comm,
                                    &gptpMpiLocalFollowerBuff,
                                    &gptpMpiFollowerAllocSize,
                                    &gptpMpiFollowerHandles[world_rank],
                                    &localFollowerHandle);

    CUmemFabricHandle *leaderHandles =
        reinterpret_cast<CUmemFabricHandle*>(malloc(world_size * sizeof(CUmemFabricHandle)));
    CUmemFabricHandle *followerHandles =
        reinterpret_cast<CUmemFabricHandle*>(malloc(world_size * sizeof(CUmemFabricHandle)));
    if (leaderHandles == nullptr || followerHandles == nullptr) {
        fprintf(stderr, "GPTP MPI rank %d failed to allocate fabric handle arrays\n", world_rank);
        MPI_Abort(comm, 1);
    }

    MPI_Allgather(&localLeaderHandle, sizeof(CUmemFabricHandle), MPI_BYTE,
                  leaderHandles, sizeof(CUmemFabricHandle), MPI_BYTE, comm);
    MPI_Allgather(&localFollowerHandle, sizeof(CUmemFabricHandle), MPI_BYTE,
                  followerHandles, sizeof(CUmemFabricHandle), MPI_BYTE, comm);

    gptpMpiLeaderBuffs =
        reinterpret_cast<ulong2**>(malloc(world_size * sizeof(ulong2 *)));
    gptpMpiFollowerBuffs =
        reinterpret_cast<ulong2**>(malloc(world_size * sizeof(ulong2 *)));
    if (gptpMpiLeaderBuffs == nullptr || gptpMpiFollowerBuffs == nullptr) {
        fprintf(stderr, "GPTP MPI rank %d failed to allocate pointer tables\n", world_rank);
        MPI_Abort(comm, 1);
    }

    for (int i = 0; i < world_size; ++i) {
        if (i == world_rank) {
            gptpMpiLeaderBuffs[i] = gptpMpiLocalLeaderBuff;
            gptpMpiFollowerBuffs[i] = gptpMpiLocalFollowerBuff;
        } else {
            gptpMpiImportFabricBuffer(&leaderHandles[i], gptpMpiLeaderAllocSize,
                                            cu_dev, comm, &gptpMpiLeaderBuffs[i],
                                            &gptpMpiLeaderHandles[i]);
            gptpMpiImportFabricBuffer(&followerHandles[i], gptpMpiFollowerAllocSize,
                                            cu_dev, comm, &gptpMpiFollowerBuffs[i],
                                            &gptpMpiFollowerHandles[i]);
        }
    }

    free(leaderHandles);
    free(followerHandles);

    const int zero = 0;
    __CHECK_CUDA(cudaMemcpyToSymbol(gptpDeviceId, &world_rank, sizeof(int)));
    __CHECK_CUDA(cudaMemcpyToSymbol(gptpNumDevices, &world_size, sizeof(int)));
    __CHECK_CUDA(cudaMemcpyToSymbol(gptpGlobalBarrier, &zero, sizeof(int)));
    __CHECK_CUDA(cudaMemcpyToSymbol(gptpGlobalBarrierSense, &zero, sizeof(int)));

    __CHECK_CUDA(cudaMalloc(&gptpMpiDeviceFollowerTable, world_size * sizeof(ulong2 *)));
    __CHECK_CUDA(cudaMemcpy(gptpMpiDeviceFollowerTable,
                            static_cast<const void*>(gptpMpiFollowerBuffs),
                            world_size * sizeof(ulong2 *), cudaMemcpyHostToDevice));
    __CHECK_CUDA(cudaMalloc(&gptpMpiDeviceLeaderTable, world_size * sizeof(ulong2 *)));
    __CHECK_CUDA(cudaMemcpy(gptpMpiDeviceLeaderTable,
                            static_cast<const void*>(gptpMpiLeaderBuffs),
                            world_size * sizeof(ulong2 *), cudaMemcpyHostToDevice));

    __CHECK_CUDA(cudaMemcpyToSymbol(gptpGlobalFollowerBuffers, &gptpMpiDeviceFollowerTable,
                                    sizeof(ulong2 *)));
    __CHECK_CUDA(cudaMemcpyToSymbol(gptpGlobalLeaderBuffers, &gptpMpiDeviceLeaderTable,
                                    sizeof(ulong2 *)));

    gptpMpiPrepared = true;
    MPI_Barrier(comm);

    if (!gptpMpiReportedConfig) {
        char processor_name[MPI_MAX_PROCESSOR_NAME] = {0};
        int processor_name_len = 0;
        MPI_Get_processor_name(processor_name, &processor_name_len);

        int *world_local_ranks = nullptr;
        int *world_local_device_counts = nullptr;
        int *world_devices = nullptr;
        char *world_processor_names = nullptr;
        if (world_rank == 0) {
            world_local_ranks = reinterpret_cast<int*>(malloc(world_size * sizeof(int)));
            world_local_device_counts = reinterpret_cast<int*>(malloc(world_size * sizeof(int)));
            world_devices = reinterpret_cast<int*>(malloc(world_size * sizeof(int)));
            world_processor_names =
                reinterpret_cast<char*>(malloc(world_size * MPI_MAX_PROCESSOR_NAME));
            if (world_local_ranks == nullptr || world_local_device_counts == nullptr ||
                world_devices == nullptr || world_processor_names == nullptr) {
                fprintf(stderr, "GPTP MPI rank %d failed to allocate topology log buffers\n",
                        world_rank);
                MPI_Abort(comm, 1);
            }
        }

        MPI_Gather(&local_rank, 1, MPI_INT, world_local_ranks, 1, MPI_INT, 0, comm);
        MPI_Gather(&local_device_count, 1, MPI_INT, world_local_device_counts, 1, MPI_INT, 0,
                   comm);
        MPI_Gather(&gptpMpiDevice, 1, MPI_INT, world_devices, 1, MPI_INT, 0, comm);
        MPI_Gather(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, world_processor_names,
                   MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, comm);

        if (world_rank == 0) {
            printf("[GPTP] MPI mode: %d rank(s), one GPU per rank via CUDA VMM fabric handles\n",
                   world_size);
            printf("[GPTP] handle exchange: MPI_Allgather of CUmemFabricHandle values, "
                   "then cuMemImportFromShareableHandle on each rank\n");
            for (int i = 0; i < world_size; ++i) {
                const char *rank_host = &world_processor_names[i * MPI_MAX_PROCESSOR_NAME];
                printf("[GPTP] rank %d host=%s local_rank=%d cuda_device=%d "
                       "host_visible_gpus=%d\n",
                       i, rank_host, world_local_ranks[i], world_devices[i],
                       world_local_device_counts[i]);
            }

            free(world_local_ranks);
            free(world_local_device_counts);
            free(world_devices);
            free(world_processor_names);
        }
        fflush(stdout);
        gptpMpiReportedConfig = true;
    }
}
#endif

__host__ void prepareGptp() {
    int device_count, originalDev; 
    __CHECK_CUDA(cudaGetDeviceCount(&device_count));

    // 0. Record the GPU before to set it afterwards -> avoid wired effects 
    __CHECK_CUDA(cudaGetDevice(&originalDev));

    // 1. Allocate the workspaces on each device
    ulong2 **leaderBuffs = reinterpret_cast<ulong2**>(malloc(device_count * sizeof(ulong2 *))); 
    ulong2 **followerBuffs = reinterpret_cast<ulong2**>(malloc(device_count * sizeof(ulong2 *))); 

    for (int i = 0; i < device_count; ++i) {
        __CHECK_CUDA(cudaSetDevice(i));
        
        // PTP part
        const size_t sequence_slots = GPTP_SAMPLES_PER_LEADER * device_count;
        __CHECK_CUDA(cudaMalloc(&leaderBuffs[i], sequence_slots * device_count * LEADER_SPACES * sizeof(ulong2)));
        __CHECK_CUDA(cudaMemset(leaderBuffs[i], 0, sequence_slots * device_count * LEADER_SPACES * sizeof(ulong2))); 
        __CHECK_CUDA(cudaMalloc(&followerBuffs[i], sequence_slots * FOLLOWER_SPACES * sizeof(ulong2))); 
        __CHECK_CUDA(cudaMemset(followerBuffs[i], 0, sequence_slots * FOLLOWER_SPACES * sizeof(ulong2))); 
    }

    const int zero = 0; 

    // 2. Update the Device Symbol on each devices 
    for (int i = 0; i < device_count; ++i) { 
        __CHECK_CUDA(cudaSetDevice(i)); 
        
        // 1. gptpDeviceId and gptpNumDevices 
        __CHECK_CUDA(cudaMemcpyToSymbol(gptpDeviceId, &i, sizeof(int)));
        __CHECK_CUDA(cudaMemcpyToSymbol(gptpNumDevices, &device_count, sizeof(int))); 
        __CHECK_CUDA(cudaMemcpyToSymbol(gptpGlobalBarrier, &zero, sizeof(int)));
        __CHECK_CUDA(cudaMemcpyToSymbol(gptpGlobalBarrierSense, &zero, sizeof(int)));
        
        // 2. Allocate Data in GLOBAL Memory (Heap)
        ulong2 *d_followerBuffs = nullptr, *d_leaderBuffs = nullptr; 
        __CHECK_CUDA(cudaMalloc(&d_followerBuffs, device_count * sizeof(ulong2 *)));
        __CHECK_CUDA(cudaMemcpy(d_followerBuffs, static_cast<const void*>(followerBuffs), 
                device_count * sizeof(ulong2 *), cudaMemcpyHostToDevice));
        __CHECK_CUDA(cudaMalloc(&d_leaderBuffs, device_count * sizeof(ulong2 *)));
        __CHECK_CUDA(cudaMemcpy(d_leaderBuffs, static_cast<const void*>(leaderBuffs), 
                device_count * sizeof(ulong2 *), cudaMemcpyHostToDevice));
        
        // 3. Update the __device__ Symbol
        __CHECK_CUDA(cudaMemcpyToSymbol(gptpGlobalFollowerBuffers, &d_followerBuffs, sizeof(ulong2 *)));
        __CHECK_CUDA(cudaMemcpyToSymbol(gptpGlobalLeaderBuffers, &d_leaderBuffs, sizeof(ulong2 *)));
    }

    cudaError_t err_ = cudaSuccess; 
    // 3. Enable the P2P Access over NvLink/UALink
    for (int i = 0; i < device_count; ++i) { 
        for (int j = 0; j < device_count; ++j) {
            if (i == j) { continue; }
            
            // i -> j
            __CHECK_CUDA(cudaSetDevice(i)); 
            err_ = cudaDeviceEnablePeerAccess(j, 0); 
            if (err_ == cudaErrorPeerAccessAlreadyEnabled) { 
                cudaGetLastError(); // reset error to cudaSuccess
            } else if (err_ != cudaSuccess) { 
                fprintf(stderr, "CUDA Error: %s at %s line %d\n", cudaGetErrorString(err_), __FILE__, __LINE__); 
                exit(1); 
            } 
            
            // j -> i
            __CHECK_CUDA(cudaSetDevice(j)); 
            err_ = cudaDeviceEnablePeerAccess(i, 0); 
            if (err_ == cudaErrorPeerAccessAlreadyEnabled) { 
                cudaGetLastError(); // reset error to cudaSuccess
            } else if (err_ != cudaSuccess) { 
                fprintf(stderr, "CUDA Error: %s at %s line %d\n", cudaGetErrorString(err_), __FILE__, __LINE__); 
                exit(1); 
            } 
        }
    }

    // reset device to previous case
    __CHECK_CUDA(cudaSetDevice(originalDev)); 
    
    // Free host side pointer arrays
    free(leaderBuffs);
    free(followerBuffs);
} 

__host__ void cleanGptp() {
#ifdef GPTP_USE_MPI
    if (gptpMpiPrepared) {
        __CHECK_CUDA(cudaSetDevice(gptpMpiDevice));
        __CHECK_CUDA(cudaDeviceSynchronize());

        if (gptpMpiDeviceLeaderTable != nullptr) {
            __CHECK_CUDA(cudaFree(gptpMpiDeviceLeaderTable));
            gptpMpiDeviceLeaderTable = nullptr;
        }
        if (gptpMpiDeviceFollowerTable != nullptr) {
            __CHECK_CUDA(cudaFree(gptpMpiDeviceFollowerTable));
            gptpMpiDeviceFollowerTable = nullptr;
        }

        if (gptpMpiLeaderBuffs != nullptr) {
            for (int i = 0; i < gptpMpiWorldSize; ++i) {
                gptpMpiUnmapFabricBuffer(gptpMpiLeaderBuffs[i],
                                               gptpMpiLeaderAllocSize,
                                               MPI_COMM_WORLD);
            }
            free(gptpMpiLeaderBuffs);
            gptpMpiLeaderBuffs = nullptr;
        }
        if (gptpMpiFollowerBuffs != nullptr) {
            for (int i = 0; i < gptpMpiWorldSize; ++i) {
                gptpMpiUnmapFabricBuffer(gptpMpiFollowerBuffs[i],
                                               gptpMpiFollowerAllocSize,
                                               MPI_COMM_WORLD);
            }
            free(gptpMpiFollowerBuffs);
            gptpMpiFollowerBuffs = nullptr;
        }

        if (gptpMpiLeaderHandles != nullptr) {
            for (int i = 0; i < gptpMpiWorldSize; ++i) {
                __CHECK_CU_MPI(cuMemRelease(gptpMpiLeaderHandles[i]), MPI_COMM_WORLD);
            }
            free(gptpMpiLeaderHandles);
            gptpMpiLeaderHandles = nullptr;
        }
        if (gptpMpiFollowerHandles != nullptr) {
            for (int i = 0; i < gptpMpiWorldSize; ++i) {
                __CHECK_CU_MPI(cuMemRelease(gptpMpiFollowerHandles[i]), MPI_COMM_WORLD);
            }
            free(gptpMpiFollowerHandles);
            gptpMpiFollowerHandles = nullptr;
        }

        gptpMpiLocalLeaderBuff = nullptr;
        gptpMpiLocalFollowerBuff = nullptr;
        __CHECK_CUDA(cudaSetDevice(gptpMpiOriginalDev));
        gptpMpiPrepared = false;
        runIdx += 1;
        return;
    }
#endif

    int device_count, originalDev; 
    __CHECK_CUDA(cudaGetDevice(&originalDev));
    __CHECK_CUDA(cudaGetDeviceCount(&device_count));

    // wait for kernel finish
    for (int devId = 0; devId < device_count; devId++) {
        __CHECK_CUDA(cudaSetDevice(devId)); 
        __CHECK_CUDA(cudaDeviceSynchronize()); 
    }

    __CHECK_CUDA(cudaSetDevice(originalDev));
    runIdx += 1; 
}

#endif // GPTP_CUH
