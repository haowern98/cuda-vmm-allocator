#ifndef VMM_PROJECT_BACKEND_ONNX_RUNNER_H_
#define VMM_PROJECT_BACKEND_ONNX_RUNNER_H_

#include <string>

namespace vmm_project {

class OnnxRunner {
 public:
  bool Load(const std::string& model_path);
  bool RunOnce(const std::string& _unused);
  void Unload();

 private:
  void* env_ = nullptr;
  void* session_ = nullptr;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_BACKEND_ONNX_RUNNER_H_
