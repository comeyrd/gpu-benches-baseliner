#ifndef GPU_UMSTREAM_HPP
#define GPU_UMSTREAM_HPP
#include <baseliner/core/Workload.hpp>
#include <cstddef>
#include <memory>
#include <string>
using namespace Baseliner::Hardware;
constexpr size_t ONE_MB = (size_t)1024 * 1024; // DEFAULT_BUFFER_SIZE = 1MB

template <typename BackendT>
class GpuUmstream : public Baseliner::IWorkload<BackendT> {
public:
  auto name() -> std::string override {
    return "gpu-umstream";
  };
  void setup(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void reset_workload(std::shared_ptr<typename BackendT::stream_t> stream) override {};
  void run_workload(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto number_of_bytes() -> std::optional<size_t> override;
  void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto validate_workload() -> bool override {
    return true;
  };

  void register_options() override {
    Baseliner::IWorkload<BackendT>::register_options();
    this->add_option("gpu-umstream", "blocksize", "the block size", blockSize);
    this->add_option("gpu-umstream", "prefetch", "Is the memory pre-fetched ?", m_prefetch);
  }

private:
  bool m_prefetch = true;
  int m_block_count;
  int blockSize = 256;
  double *A, *B, *C;
  size_t m_item_count = ONE_MB / sizeof(double);
};
#endif // GPU_UMSTREAM_HPP
