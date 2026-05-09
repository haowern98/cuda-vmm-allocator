#include "backend/onnx_runner.h"

#include <cstdio>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace vmm_project {

bool OnnxRunner::Load(const std::string& model_path) {
  auto* env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                           "vmm-onnx-runner");
  env_ = env;

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_DISABLE_ALL);

  // Enable CUDA provider if available.
  try {
    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    opts.AppendExecutionProvider_CUDA(cuda_opts);
  } catch (...) {
    std::printf("  CUDA provider not available, using CPU fallback\n");
  }

  auto* session = new Ort::Session(*env,
                                    std::wstring(model_path.begin(),
                                                  model_path.end())
                                        .c_str(),
                                    opts);
  session_ = session;

  std::printf("  Model loaded. Inputs: %zu, Outputs: %zu\n",
              session->GetInputCount(), session->GetOutputCount());

  return true;
}

bool OnnxRunner::RunOnce(const std::string& _unused) {
  auto* session = static_cast<Ort::Session*>(session_);
  auto* env = static_cast<Ort::Env*>(env_);

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo mem_info_cpu(
      "Cpu", OrtDeviceAllocator, /*device_id=*/0,
      OrtMemTypeDefault);

  // Build input tensors and collect input names.
  std::vector<std::string> input_name_strings;
  std::vector<const char*> input_names;
  std::vector<Ort::Value> inputs;
  inputs.reserve(session->GetInputCount());

  for (std::size_t i = 0; i < session->GetInputCount(); ++i) {
    auto name = session->GetInputNameAllocated(i, allocator);
    input_name_strings.push_back(name.get());
    input_names.push_back(input_name_strings.back().c_str());

    auto type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    auto elem_type = tensor_info.GetElementType();

    // Replace dynamic dimensions with 1.
    std::vector<int64_t> concrete_shape;
    for (auto d : shape) {
      concrete_shape.push_back(d < 0 ? 1 : d);
    }

    std::size_t count = 1;
    for (auto d : concrete_shape) count *= d;

    Ort::Value val{nullptr};
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::vector<float> data(count, 0.0f);
      val = Ort::Value::CreateTensor<float>(
          mem_info_cpu, data.data(), count,
          concrete_shape.data(), concrete_shape.size());
    } else if (elem_type ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      std::vector<int64_t> data(count, 0);
      val = Ort::Value::CreateTensor<int64_t>(
          mem_info_cpu, data.data(), count,
          concrete_shape.data(), concrete_shape.size());
    } else {
      std::printf("  Skipping input %s (unsupported type)\n",
                  name.get());
      continue;
    }
    inputs.push_back(std::move(val));
  }

  // Collect output names.
  std::vector<std::string> output_name_strings;
  std::vector<const char*> output_names;
  std::vector<Ort::Value> outputs;
  for (std::size_t i = 0; i < session->GetOutputCount(); ++i) {
    auto name = session->GetOutputNameAllocated(i, allocator);
    output_name_strings.push_back(name.get());
    output_names.push_back(output_name_strings.back().c_str());
  }

  try {
    outputs = session->Run(
        Ort::RunOptions{nullptr}, input_names.data(),
        inputs.data(), inputs.size(), output_names.data(),
        output_names.size());
  } catch (const Ort::Exception& e) {
    std::printf("  Run failed: %s\n", e.what());
    return false;
  }

  std::printf("  Inference ran successfully (%zu outputs)\n",
              outputs.size());
  return true;
}

void OnnxRunner::Unload() {
  delete static_cast<Ort::Session*>(session_);
  session_ = nullptr;
  delete static_cast<Ort::Env*>(env_);
  env_ = nullptr;
}

}  // namespace vmm_project
