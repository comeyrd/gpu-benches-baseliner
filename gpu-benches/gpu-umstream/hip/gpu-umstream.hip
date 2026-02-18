#include "hip/hip_runtime.h"
#include "baseliner/backend/hip/HipBackend.hpp"
#include "gpu-umstream.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Case.hpp>
#include <baseliner/Suite.hpp>

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
void GpuUmstream<Device::HipBackend>::setup(std::shared_ptr<Device::HipBackend::stream_t> stream) {
  size_t nb_bytes = buffer_size_kb * KBTOB * sizeof(double);
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
  if(m_prefetch){
    CHECK_HIP(hipMemPrefetchAsync(A, nb_bytes, deviceId, *stream));
    CHECK_HIP(hipMemPrefetchAsync(B, nb_bytes, deviceId, *stream));
    CHECK_HIP(hipMemPrefetchAsync(C, nb_bytes, deviceId, *stream));
  }
}
template <>
void GpuUmstream<Device::HipBackend>::run_case(std::shared_ptr<HipBackend::stream_t> stream) {
  triad<<<m_block_count, blockSize, 0, *stream>>>(A, B, C, buffer_size_kb * KBTOB);
}
template <>
void GpuUmstream<Device::HipBackend>::setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) {
  size_t buffer_size_bytes = buffer_size_kb * KBTOB;
  size_t total_bytes = buffer_size_bytes * sizeof(double) * 3; // 3 reads
  engine->register_metric<Baseliner::Stats::ByteNumbers>(total_bytes);
}
template <>
void GpuUmstream<Device::HipBackend>::update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) {
  size_t buffer_size_bytes = buffer_size_kb * KBTOB;
  size_t total_bytes = buffer_size_bytes * sizeof(double) * 3; // 3 reads
  engine->update_values<Baseliner::Stats::ByteNumbers>(total_bytes);
}
template <>
void GpuUmstream<Device::HipBackend>::teardown(std::shared_ptr<typename HipBackend::stream_t> stream) {
  CHECK_HIP(hipFree(A));
  CHECK_HIP(hipFree(B));
  CHECK_HIP(hipFree(C));
}

namespace {
  static auto umstreamBench = Baseliner::HipBenchmark()
                                  .set_case<GpuUmstream<Device::HipBackend>>()
                                  .set_stopping_criterion<Baseliner::StoppingCriterion>(10, 1)
                                  .set_block(false)
                                  .set_flush_l2(true)
                                  .add_stat<Baseliner::Stats::Median>()
                                  .add_stat<Baseliner::Stats::MedianItemTroughput>();

  Baseliner::Axe axe2 = {
      "gpu-umstream",
      "bufferSizeKB",
      {"1024", "2048", "4096", "8192", "16384", "32768", "65536", "131072", "262144", "524288", "1048576", "2097152"}};

  Baseliner::SingleAxeSuite suite2(std::make_shared<Baseliner::HipBenchmark>(std::move(umstreamBench)),
                                   std::move(axe2));
  BASELINER_REGISTER_TASK(&suite2);

} // namespace
