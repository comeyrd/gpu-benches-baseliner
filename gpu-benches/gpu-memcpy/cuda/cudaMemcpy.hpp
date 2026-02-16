#ifndef CUDA_GPU_MEMCPY
#define CUDA_GPU_MEMCPY
#include "baseliner/backend/cuda/CudaBackend.hpp"
#include <baseliner/Case.hpp>
#include <cstddef>
#include <string>
using namespace Baseliner::Device;
constexpr size_t DEFAULT_BUFFER_SIZE_KB = (size_t)2 * 1024 * 1024;
constexpr size_t KBTOB = 1024;
class GpuMemcpy : public Baseliner::ICase<CudaBackend> {

public:
  auto name() -> std::string override {
    return "cudaMemcpy";
  };
  void setup(std::shared_ptr<typename CudaBackend::stream_t> stream) override;
  void reset_case(std::shared_ptr<typename CudaBackend::stream_t> stream) override;
  void run_case(std::shared_ptr<typename CudaBackend::stream_t> stream) override;
  void teardown(std::shared_ptr<typename CudaBackend::stream_t> stream) override;
  auto validate_case() -> bool override {
    return true;
  };

  void register_options() override {
    add_option("cudaMemcpy", "bufferSizeKB", "the size of the buffer in bytes", buffer_size_kb);
  }

private:
  char *host_buffer;
  char *device_buffer;
  size_t buffer_size_kb = DEFAULT_BUFFER_SIZE_KB;
};
#endif // CUDA_GPU_MEMCPY
