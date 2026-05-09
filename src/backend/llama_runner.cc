#include "backend/llama_runner.h"

#include <cstdio>
#include <vector>

#include <llama.h>

namespace vmm_project {

bool LlamaRunner::Load(const std::string& model_path) {
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = -1;

  model_ = llama_model_load_from_file(model_path.c_str(),
                                       model_params);
  if (!model_) {
    std::printf("llama_model_load_from_file failed\n");
    return false;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 512;

  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_) {
    std::printf("llama_init_from_model failed\n");
    llama_model_free(model_);
    model_ = nullptr;
    return false;
  }

  return true;
}

bool LlamaRunner::RunOnce(const std::string& prompt) {
  const llama_vocab* vocab = llama_model_get_vocab(model_);

  // Tokenize the prompt.
  int n_tokens = llama_tokenize(vocab, prompt.c_str(),
                                 static_cast<int>(prompt.size()),
                                 nullptr, 0, true, true);
  if (n_tokens < 0) {
    std::printf("tokenize count failed\n");
    return false;
  }

  std::vector<llama_token> tokens(n_tokens);
  int actual = llama_tokenize(vocab, prompt.c_str(),
                               static_cast<int>(prompt.size()),
                               tokens.data(), n_tokens, true, true);
  if (actual < 0) {
    std::printf("tokenize failed\n");
    return false;
  }
  tokens.resize(actual);

  // Build batch for prefill.
  llama_batch batch = llama_batch_init(
      static_cast<int32_t>(tokens.size()), /*embd=*/0,
      /*n_seq_max=*/1);
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = (i == batch.n_tokens - 1);
  }

  if (llama_decode(ctx_, batch) != 0) {
    std::printf("llama_decode prefill failed\n");
    llama_batch_free(batch);
    return false;
  }
  llama_batch_free(batch);

  // Sample the last token (greedy).
  const float* logits = llama_get_logits(ctx_);
  int n_vocab = llama_vocab_n_tokens(vocab);
  llama_token max_token = 0;
  float max_logit = logits[0];
  for (int i = 1; i < n_vocab; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
      max_token = i;
    }
  }

  char buf[256] = {};
  int len = llama_token_to_piece(vocab, max_token, buf,
                                  sizeof(buf), 0, true);
  if (len < 0) {
    buf[0] = '?';
    buf[1] = '\0';
  }

  std::printf("  Next token: %d (\"%s\", logit=%.2f)\n",
              max_token, buf, max_logit);
  return true;
}

void LlamaRunner::Unload() {
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

}  // namespace vmm_project
