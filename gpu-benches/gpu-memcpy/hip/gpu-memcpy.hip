#include "gpu-memcpy.hpp"
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
#include <cstring>
using namespace Baseliner;
template <>
void GpuMemcpy<Hardware::HipBackend>::setup(std::shared_ptr<Hardware::HipBackend::stream_t> stream) {
  item_count = this->get_work_size() * (ONE_MB / sizeof(char));
  size_t bytes_size = item_count * sizeof(char);
  CHECK_HIP(hipMallocAsync(&device_buffer, bytes_size, *stream));
  if (m_pin_memory) {
    CHECK_HIP(hipHostMalloc(&host_buffer, bytes_size));
  } else {
    host_buffer = new char[bytes_size];
  }
}
template <>
void GpuMemcpy<Hardware::HipBackend>::reset_workload(std::shared_ptr<Hardware::HipBackend::stream_t> /*stream*/) {
  memset(host_buffer, 0, item_count * sizeof(char));
}
template <>
void GpuMemcpy<Hardware::HipBackend>::run_workload(std::shared_ptr<Hardware::HipBackend::stream_t> stream) {
  CHECK_HIP(hipMemcpyAsync(device_buffer, host_buffer, item_count * sizeof(char), hipMemcpyDefault, *stream));
}

template <>
void GpuMemcpy<Hardware::HipBackend>::teardown(std::shared_ptr<Hardware::HipBackend::stream_t> /*stream*/) {
  CHECK_HIP(hipFree(device_buffer));
  if (m_pin_memory) {
    CHECK_HIP(hipHostFree(host_buffer));
  } else {
    delete host_buffer;
  }
}
template <>
auto GpuMemcpy<Hardware::HipBackend>::number_of_bytes() -> std::optional<size_t> {
  return item_count * sizeof(char);
};

using HipMemcpy = GpuMemcpy<Hardware::HipBackend>;
BASELINER_REGISTER_WORKLOAD_NAME(HipMemcpy, HipMemcpy().name());
