#ifndef GPU_MEMCPY
#define GPU_MEMCPY
#include <baseliner/core/Workload.hpp>

#include <cstddef>
#include <memory>
#include <string>
constexpr size_t ONE_MB = (size_t)1024 * 1024;

template <typename BackendT>
class GpuMemcpy : public Baseliner::IWorkload<BackendT> {

public:
  auto name() -> std::string override {
    return "gpu-memcpy";
  };
  void setup(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void reset_workload(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void run_workload(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto number_of_bytes() -> std::optional<size_t> override;
  void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto validate_workload() -> bool override {
    return true;
  };
  void register_options() override {
    Baseliner::IWorkload<BackendT>::register_options();
    this->add_option("gpu-memcpy", "pin-memory", "Is the host memory pinned ? ", m_pin_memory);
  }

private:
  char *host_buffer;
  char *device_buffer;
  bool m_pin_memory = true;
  size_t item_count = ONE_MB / sizeof(char);
};
#endif // GPU_MEMCPY
