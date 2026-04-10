#include "gpu-umstream.hpp"
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
#include <memory>
using namespace Baseliner::Hardware;
using namespace Baseliner;

__global__ void scale(double *A, double *B, size_t N) {
  size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = B[i] * 1.3;
  }
}

__global__ void triad(double *A, double *B, double *C, size_t N) {
  size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = B[i] + C[i] * 1.3;
  }
}
template <>
void GpuUmstream<HipBackend>::setup(std::shared_ptr<HipBackend::stream_t> stream) {
  size_t nb_bytes = this->get_work_size() * ONE_MB;
  m_item_count = this->get_work_size() * (ONE_MB / sizeof(double));

  CHECK_HIP(hipMallocManaged(&A, nb_bytes));
  CHECK_HIP(hipMallocManaged(&B, nb_bytes));
  CHECK_HIP(hipMallocManaged(&C, nb_bytes));

  hipDeviceProp_t prop;
  int deviceId;
  CHECK_HIP(hipGetDevice(&deviceId));
  CHECK_HIP(hipGetDeviceProperties(&prop, deviceId));
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, triad, blockSize, 0));
  m_block_count = smCount * maxActiveBlocks;
  if (m_prefetch) {
    CHECK_HIP(hipMemPrefetchAsync(A, nb_bytes, deviceId, *stream));
    CHECK_HIP(hipMemPrefetchAsync(B, nb_bytes, deviceId, *stream));
    CHECK_HIP(hipMemPrefetchAsync(C, nb_bytes, deviceId, *stream));
  }
}
template <>
void GpuUmstream<HipBackend>::run_workload(std::shared_ptr<HipBackend::stream_t> stream) {
  triad<<<m_block_count, blockSize, 0, *stream>>>(A, B, C, m_item_count);
}

template <>
auto GpuUmstream<HipBackend>::number_of_bytes() -> std::optional<size_t> {
  return m_item_count * sizeof(double) * 3; // 3reads;
};
template <>
void GpuUmstream<HipBackend>::teardown(std::shared_ptr<typename HipBackend::stream_t> stream) {
  CHECK_HIP(hipFree(A));
  CHECK_HIP(hipFree(B));
  CHECK_HIP(hipFree(C));
}
using HipUmstream = GpuUmstream<Hardware::HipBackend>;
BASELINER_REGISTER_WORKLOAD_NAME(HipUmstream, HipUmstream().name());