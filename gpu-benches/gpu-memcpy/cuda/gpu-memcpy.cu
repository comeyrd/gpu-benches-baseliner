#include "cudaMemcpy.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Case.hpp>
#include <baseliner/Suite.hpp>

void GpuMemcpy::setup(std::shared_ptr<CudaBackend::stream_t> stream) {
  CHECK_CUDA(cudaMallocAsync(&device_buffer, buffer_size_kb * KBTOB, *stream));
  CHECK_CUDA(cudaMallocHost(&host_buffer, buffer_size_kb * KBTOB));
}
void GpuMemcpy::reset_case(std::shared_ptr<CudaBackend::stream_t> /*stream*/) {
  memset(host_buffer, 0, buffer_size_kb * KBTOB);
}
void GpuMemcpy::run_case(std::shared_ptr<CudaBackend::stream_t> stream) {
  CHECK_CUDA(cudaMemcpyAsync(device_buffer, host_buffer, buffer_size_kb * KBTOB, cudaMemcpyDefault, *stream));
}

void GpuMemcpy::setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) {
  engine->register_metric<Baseliner::Stats::ByteNumbers>(buffer_size_kb * KBTOB);
};
void GpuMemcpy::update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) {
  engine->update_values<Baseliner::Stats::ByteNumbers>(buffer_size_kb * KBTOB);
}

void GpuMemcpy::teardown(std::shared_ptr<CudaBackend::stream_t> /*stream*/) {
  CHECK_CUDA(cudaFree(device_buffer));
  CHECK_CUDA(cudaFreeHost(host_buffer));
}

static auto memcpyBench = Baseliner::CudaBenchmark()
                              .set_case<GpuMemcpy>()
                              .set_stopping_criterion<Baseliner::StoppingCriterion>(100, 1)
                              .set_block(false)
                              .add_stat<Baseliner::Stats::Median>()
                              .add_stat<Baseliner::Stats::MedianItemTroughput>();

Baseliner::Axe axe = {"cudaMemcpy",
                      "bufferSizeKB",
                      {"1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096", "8192",
                       "16384", "32768", "65536", "131072", "262144"}};

Baseliner::SingleAxeSuite suite(std::make_shared<Baseliner::CudaBenchmark>(std::move(memcpyBench)), std::move(axe));
BASELINER_REGISTER_TASK(&suite);
