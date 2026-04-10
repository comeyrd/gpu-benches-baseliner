#include "gpu-memcpy.hpp"
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
#include <cstring>
using namespace Baseliner;
template <>
void GpuMemcpy<Hardware::CudaBackend>::setup(std::shared_ptr<Hardware::CudaBackend::stream_t> stream) {
  item_count = this->get_work_size() * (ONE_MB / sizeof(char));
  size_t bytes_size = item_count * sizeof(char);
  CHECK_CUDA(cudaMallocAsync(&device_buffer, bytes_size, *stream));
  if (m_pin_memory) {
    CHECK_CUDA(cudaMallocHost(&host_buffer, bytes_size));
  } else {
    host_buffer = new char[bytes_size];
  }
}
template <>
void GpuMemcpy<Hardware::CudaBackend>::reset_workload(std::shared_ptr<Hardware::CudaBackend::stream_t> /*stream*/) {
  memset(host_buffer, 0, item_count * sizeof(char));
}
template <>
void GpuMemcpy<Hardware::CudaBackend>::run_workload(std::shared_ptr<Hardware::CudaBackend::stream_t> stream) {
  CHECK_CUDA(cudaMemcpyAsync(device_buffer, host_buffer, item_count * sizeof(char), cudaMemcpyDefault, *stream));
}

template <>
void GpuMemcpy<Hardware::CudaBackend>::teardown(std::shared_ptr<Hardware::CudaBackend::stream_t> /*stream*/) {
  CHECK_CUDA(cudaFree(device_buffer));
  if (m_pin_memory) {
    CHECK_CUDA(cudaFreeHost(host_buffer));
  } else {
    delete host_buffer;
  }
}
template <>
auto GpuMemcpy<Hardware::CudaBackend>::number_of_bytes() -> std::optional<size_t> {
  return item_count * sizeof(char);
};
using CudaMemcpy = GpuMemcpy<Hardware::CudaBackend>;
BASELINER_REGISTER_WORKLOAD_NAME(CudaMemcpy, CudaMemcpy().name());
