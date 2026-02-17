#ifndef GPU_UMSTREAM_HPP
#define GPU_UMSTREAM_HPP
#include <baseliner/Case.hpp>
#include <cstddef>
#include <string>
using namespace Baseliner::Device;
constexpr size_t DEFAULT_BUFFER_SIZE_KB = (size_t)1024;
constexpr size_t KBTOB = 1024;

template <typename Backend>
class GpuUmstream : public Baseliner::ICase<Backend> {
public:
  auto name() -> std::string override {
    return "cudaMemcpy";
  };
  void setup(std::shared_ptr<typename Backend::stream_t> stream) override;
  void reset_case(std::shared_ptr<typename Backend::stream_t> stream) override {};
  void setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override;
  void update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override;
  void run_case(std::shared_ptr<typename Backend::stream_t> stream) override;
  void teardown(std::shared_ptr<typename Backend::stream_t> stream) override;
  auto validate_case() -> bool override {
    return true;
  };

  void register_options() override {
    this->add_option("gpu-umstream", "bufferSizeKB", "the size of the buffer in bytes", buffer_size_kb);
    this->add_option("gpu-umstream", "blocksize", "the block size", blockSize);
  }

private:
  int m_block_count;
  int blockSize = 256;
  double *A, *B, *C;
  size_t buffer_size_kb = DEFAULT_BUFFER_SIZE_KB;
};
#endif // CUDA_GPU_MEMCPY
