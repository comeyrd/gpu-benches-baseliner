#ifndef GPU_MEMCPY
#define GPU_MEMCPY
#include <baseliner/Case.hpp>
#include <cstddef>
#include <memory>
#include <string>
constexpr size_t DEFAULT_BUFFER_SIZE_KB = (size_t)2 * 1024 * 1024;
constexpr size_t KBTOB = 1024;

template <typename Backend>
class GpuMemcpy : public Baseliner::ICase<Backend> {

public:
  auto name() -> std::string override ;
  void setup(std::shared_ptr<typename Backend::stream_t> stream) override;
  void reset_case(std::shared_ptr<typename Backend::stream_t> stream) override;
  void run_case(std::shared_ptr<typename Backend::stream_t> stream) override;
  void teardown(std::shared_ptr<typename Backend::stream_t> stream) override;
  void setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override;
  void update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override;
  auto validate_case() -> bool override {
    return true;
  };

  void register_options() override {
    this->add_option("gpu-memcpy", "bufferSizeKB", "the size of the buffer in bytes", buffer_size_kb);
  }

private:
  char *host_buffer;
  char *device_buffer;
  size_t buffer_size_kb = DEFAULT_BUFFER_SIZE_KB;
};
#endif // GPU_MEMCPY
