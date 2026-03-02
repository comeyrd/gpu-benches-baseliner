#ifndef GPU_MEMCPY
#define GPU_MEMCPY
#include "baseliner/Axe.hpp"
#include "baseliner/ConfigFile.hpp"
#include "baseliner/Options.hpp"
#include <baseliner/managers/RegisteringMacros.hpp>

#include <baseliner/Case.hpp>
#include <cstddef>
#include <memory>
#include <string>
constexpr size_t ONE_MB = (size_t)1024 * 1024;

template <typename BackendT>
class GpuMemcpy : public Baseliner::ICase<BackendT> {

public:
  auto name() -> std::string override {
    return "gpu-memcpy";
  };
  void setup(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void reset_case(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void run_case(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto number_of_bytes() -> std::optional<size_t> override;
  void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto validate_case() -> bool override {
    return true;
  };
  void register_options() override {
    Baseliner::ICase<BackendT>::register_options();
    this->add_option("gpu-memcpy", "pin-memory", "Is the host memory pinned ? ", m_pin_memory);
  }

private:
  char *host_buffer;
  char *device_buffer;
  bool m_pin_memory = true;
  size_t item_count = ONE_MB / sizeof(char);
};
#endif // GPU_MEMCPY
