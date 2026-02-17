#include "gpu-memcpy.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/backend/hip/HipBackend.hpp>
#include <baseliner/Case.hpp>
#include <baseliner/Suite.hpp>
#include <cstring>
using namespace Baseliner;
template<>
void GpuMemcpy<Device::HipBackend>::setup(std::shared_ptr<Device::HipBackend::stream_t> stream) {
  CHECK_HIP(hipMallocAsync(&device_buffer, buffer_size_kb * KBTOB, *stream));
  CHECK_HIP(hipHostMalloc(&host_buffer, buffer_size_kb * KBTOB));
}
template<>
void GpuMemcpy<Device::HipBackend>::reset_case(std::shared_ptr<Device::HipBackend::stream_t> /*stream*/) {
  memset(host_buffer, 0, buffer_size_kb * KBTOB);
}
template<>
void GpuMemcpy<Device::HipBackend>::run_case(std::shared_ptr<Device::HipBackend::stream_t> stream) {
  CHECK_HIP(hipMemcpyAsync(device_buffer, host_buffer, buffer_size_kb * KBTOB, hipMemcpyDefault, *stream));
}

template<>
void GpuMemcpy<Device::HipBackend>::teardown(std::shared_ptr<Device::HipBackend::stream_t> /*stream*/) {
  CHECK_HIP(hipFree(device_buffer));
  CHECK_HIP(hipHostFree(host_buffer));
}
template<>
void GpuMemcpy<Device::HipBackend>::setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) {
  engine->register_metric<Baseliner::Stats::ByteNumbers>(buffer_size_kb * KBTOB);
};
template<>
void GpuMemcpy<Device::HipBackend>::update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) {
  engine->update_values<Baseliner::Stats::ByteNumbers>(buffer_size_kb * KBTOB);
}

template<>
auto GpuMemcpy<Device::HipBackend>::name()->std::string{
  return "hip-memcpy";
}

static auto memcpyBench = Baseliner::HipBenchmark()
                              .set_case<GpuMemcpy<Device::HipBackend>>()
                              .set_stopping_criterion<Baseliner::StoppingCriterion>(2, 1)
                              .set_block(false)
                              .add_stat<Baseliner::Stats::Median>()
                              .add_stat<Baseliner::Stats::MedianItemTroughput>();

Baseliner::Axe axe = {"gpu-memcpy",
                      "bufferSizeKB",
                      {"1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096", "8192",
                       "16384", "32768", "65536", "131072", "262144"}};

Baseliner::SingleAxeSuite suite(std::make_shared<Baseliner::HipBenchmark>(std::move(memcpyBench)), std::move(axe));
BASELINER_REGISTER_TASK(&suite);
