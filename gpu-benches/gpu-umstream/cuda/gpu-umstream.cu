#include "gpu-umstream.hpp"
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>

using namespace Baseliner;
using namespace Baseliner::Hardware;

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
void GpuUmstream<CudaBackend>::setup(std::shared_ptr<CudaBackend::stream_t> stream) {
  size_t nb_bytes = this->get_work_size() * ONE_MB;
  m_item_count = this->get_work_size() * (ONE_MB / sizeof(double));
  CHECK_CUDA(cudaMallocManaged(&A, nb_bytes));
  CHECK_CUDA(cudaMallocManaged(&B, nb_bytes));
  CHECK_CUDA(cudaMallocManaged(&C, nb_bytes));

  cudaDeviceProp prop;
  int deviceId;
  CHECK_CUDA(cudaGetDevice(&deviceId));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, triad, blockSize, 0));
  m_block_count = smCount * maxActiveBlocks;
  if (m_prefetch) {
    CHECK_CUDA(cudaMemPrefetchAsync(A, nb_bytes, deviceId, *stream));
    CHECK_CUDA(cudaMemPrefetchAsync(B, nb_bytes, deviceId, *stream));
    CHECK_CUDA(cudaMemPrefetchAsync(C, nb_bytes, deviceId, *stream));
  }
}
template <>
void GpuUmstream<CudaBackend>::run_workload(std::shared_ptr<CudaBackend::stream_t> stream) {
  triad<<<m_block_count, blockSize, 0, *stream>>>(A, B, C, m_item_count);
}
template <>
auto GpuUmstream<CudaBackend>::number_of_bytes() -> std::optional<size_t> {
  return m_item_count * sizeof(double) * 3; // 3reads;
};
template <>
void GpuUmstream<CudaBackend>::teardown(std::shared_ptr<typename CudaBackend::stream_t> stream) {
  CHECK_CUDA(cudaFree(A));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(C));
}
using CudaUmstream = GpuUmstream<CudaBackend>;
BASELINER_REGISTER_WORKLOAD_NAME(CudaUmstream, CudaUmstream().name());