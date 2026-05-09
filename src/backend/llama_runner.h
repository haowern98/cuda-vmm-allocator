#ifndef VMM_PROJECT_BACKEND_LLAMA_RUNNER_H_
#define VMM_PROJECT_BACKEND_LLAMA_RUNNER_H_

#include <string>

// Forward declarations for llama.cpp types.
struct llama_model;
struct llama_context;

namespace vmm_project {

class LlamaRunner {
 public:
  bool Load(const std::string& model_path);
  bool RunOnce(const std::string& prompt);
  void Unload();

 private:
  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_BACKEND_LLAMA_RUNNER_H_
