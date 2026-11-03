/*
 * llama3_inference.cpp
 * 
 * Standalone C++ Inference Engine for Llama 3.2 3B.
 * No external heavy libraries required (Pure C++17 + standard math).
 * 
 * Features implemented:
 * - RMSNorm (Root Mean Square Layer Normalization)
 * - RoPE (Rotary Position Embeddings) with theta = 500000.0f
 * - Grouped-Query Attention (GQA) with KV-Cache (24 Query Heads, 8 KV Heads)
 * - SwiGLU Feed-Forward Network (Gated SiLU activations)
 * - Residual Connections (skip connections)
 * - Auto-regressive Generation Loop (Greedy Search / Argmax)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>
#include <chrono>
#include <algorithm>

// ----------------------------------------------------------------------------
// Model Hyperparameters
// ----------------------------------------------------------------------------
struct Config {
    int dim;              // Hidden dimension (Llama 3.2 3B = 3072)
    int intermediate_dim; // FFN hidden dimension (Llama 3.2 3B = 8192)
    int n_layers;         // Number of transformer layers (28)
    int n_heads;          // Number of query attention heads (24)
    int n_kv_heads;       // Number of key/value attention heads (8 for GQA)
    int vocab_size;       // Vocabulary size (128256)
    int seq_len;          // Maximum context sequence length (e.g. 2048)
};

// ----------------------------------------------------------------------------
// Weights Pointers
// ----------------------------------------------------------------------------
struct TransformerWeights {
    // Token embeddings: [vocab_size, dim]
    float* token_embedding_table;

    // Layer weights (28 layers)
    float* rms_att_weight; // [n_layers, dim]
    float* wq;             // [n_layers, n_heads * head_dim, dim] -> [28, 3072, 3072]
    float* wk;             // [n_layers, n_kv_heads * head_dim, dim] -> [28, 1024, 3072]
    float* wv;             // [n_layers, n_kv_heads * head_dim, dim] -> [28, 1024, 3072]
    float* wo;             // [n_layers, dim, n_heads * head_dim] -> [28, 3072, 3072]

    // Post-Attention / FFN Normalization
    float* rms_ffn_weight; // [n_layers, dim]
    float* w_gate;         // [n_layers, intermediate_dim, dim] -> [28, 8192, 3072]
    float* w_up;           // [n_layers, intermediate_dim, dim] -> [28, 8192, 3072]
    float* w_down;         // [n_layers, dim, intermediate_dim] -> [28, 3072, 8192]

    // Final normalization layer
    float* rms_final_weight; // [dim]

    // Language Model Head (Classifier): [vocab_size, dim]
    float* w_cls;
};

// ----------------------------------------------------------------------------
// RunState: Memory buffers required during forward pass execution
// ----------------------------------------------------------------------------
struct RunState {
    float *x;      // Current activation vector [dim]
    float *xb;     // Buffer for normalized activation [dim]
    float *xb2;    // Second buffer for residual addition [dim]
    float *hb;     // FFN intermediate buffer [intermediate_dim]
    float *hb2;    // FFN intermediate buffer 2 [intermediate_dim]
    float *q;      // Query vector [n_heads * head_dim] = [dim]
    float *k;      // Key vector [n_kv_heads * head_dim]
    float *v;      // Value vector [n_kv_heads * head_dim]
    float *att;    // Attention scores buffer [n_heads, seq_len]
    float *logits; // Output vocabulary probabilities/scores [vocab_size]

    // KV Cache: Past tokens ke keys aur values yahan store hote hain
    // Shape: [n_layers, seq_len, n_kv_heads * head_dim]
    float *key_cache;
    float *value_cache;
};

// ----------------------------------------------------------------------------
// Mathematical Operations / Transformer Building Blocks
// ----------------------------------------------------------------------------

// 1. Root Mean Square Normalization (RMSNorm)
// Formula: out_i = (x_i / sqrt(mean(x^2) + eps)) * weight_i
void rmsnorm(float* out, float* x, float* weight, int size, float eps = 1e-5f) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / std::sqrt((sum_sq / size) + eps);
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * rms * weight[i];
    }
}

// 2. Matrix-Vector Multiplication: out = W * x
// W shape: [d_out, d_in], x shape: [d_in], out shape: [d_out]
void matmul(float* out, float* x, float* w, int d_in, int d_out) {
    #pragma omp parallel for if (d_out > 256)
    for (int i = 0; i < d_out; i++) {
        float val = 0.0f;
        int row_offset = i * d_in;
        for (int j = 0; j < d_in; j++) {
            val += w[row_offset + j] * x[j];
        }
        out[i] = val;
    }
}

// 3. Rotary Position Embeddings (RoPE)
// Llama 3 uses RoPE with theta base = 500,000.0f
void apply_rope(float* vec, int pos, int head_dim, int num_heads, float theta = 500000.0f) {
    for (int h = 0; h < num_heads; h++) {
        float* head_vec = vec + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(theta, (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float v0 = head_vec[i];
            float v1 = head_vec[i + 1];

            // 2D Rotation matrix multiplication
            head_vec[i]     = v0 * cos_val - v1 * sin_val;
            head_vec[i + 1] = v0 * sin_val + v1 * cos_val;
        }
    }
}

// 4. Softmax Function
// Numerically stable softmax: e^(x_i - max) / sum(e^(x_j - max))
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// ----------------------------------------------------------------------------
// Transformer Forward Pass (Single Token at Position pos)
// ----------------------------------------------------------------------------
void forward(Config* p, TransformerWeights* w, RunState* s, int token, int pos) {
    int dim = p->dim;
    int intermediate_dim = p->intermediate_dim;
    int head_dim = dim / p->n_heads; // 3072 / 24 = 128
    int kv_dim = p->n_kv_heads * head_dim; // 8 * 128 = 1024
    int kv_mul = p->n_heads / p->n_kv_heads; // 24 / 8 = 3 (GQA multiplier)

    // Step 1: Token Embedding Lookup
    // Input token index se embedding vector copy karo
    float* content_row = &(w->token_embedding_table[token * dim]);
    std::memcpy(s->x, content_row, dim * sizeof(float));

    // Step 2: Transformer Layers Loop
    for (int l = 0; l < p->n_layers; l++) {

        // --- Attention Pre-Normalization (RMSNorm) ---
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);

        // --- Q, K, V Projections ---
        // Q: [dim] -> [dim]
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        // K: [dim] -> [kv_dim]
        matmul(s->k, s->xb, w->wk + l * kv_dim * dim, dim, kv_dim);
        // V: [dim] -> [kv_dim]
        matmul(s->v, s->xb, w->wv + l * kv_dim * dim, dim, kv_dim);

        // --- Rotary Position Embeddings (RoPE) ---
        apply_rope(s->q, pos, head_dim, p->n_heads);
        apply_rope(s->k, pos, head_dim, p->n_kv_heads);

        // --- Store Key aur Value to KV Cache ---
        // Cache location: layer 'l', position 'pos'
        int cache_offset = l * p->seq_len * kv_dim + pos * kv_dim;
        std::memcpy(s->key_cache + cache_offset, s->k, kv_dim * sizeof(float));
        std::memcpy(s->value_cache + cache_offset, s->v, kv_dim * sizeof(float));

        // --- Grouped-Query Attention (GQA) ---
        // Har Query Head ke liye attention score calculate karo
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float* q_head = s->q + h * head_dim;
            float* att_head = s->att + h * p->seq_len;

            // GQA: Query head 'h' key/value head 'h / kv_mul' ko share karta hai
            int kv_h = h / kv_mul;

            // Past tokens (0 to pos) ke saath dot product (Attention scores)
            for (int t = 0; t <= pos; t++) {
                int key_offset = l * p->seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                float* k_past = s->key_cache + key_offset;

                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q_head[i] * k_past[i];
                }
                score /= std::sqrt((float)head_dim);
                att_head[t] = score;
            }

            // Softmax over attention scores of tokens 0..pos
            softmax(att_head, pos + 1);

            // Weighted sum over Value vectors: xb = sum(score * V)
            float* xb_head = s->xb + h * head_dim;
            std::memset(xb_head, 0, head_dim * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                int val_offset = l * p->seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                float* v_past = s->value_cache + val_offset;
                float a = att_head[t];

                for (int i = 0; i < head_dim; i++) {
                    xb_head[i] += a * v_past[i];
                }
            }
        }

        // --- Attention Output Projection (wo) ---
        // s->xb2 = Wo * s->xb
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // --- Residual Connection 1 ---
        // x = x + Attention_Output
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }

        // --- FFN Pre-Normalization (RMSNorm) ---
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

        // --- SwiGLU Feed-Forward Network ---
        // 1. gate = xb * w_gate
        // 2. up   = xb * w_up
        matmul(s->hb,  s->xb, w->w_gate + l * intermediate_dim * dim, dim, intermediate_dim);
        matmul(s->hb2, s->xb, w->w_up   + l * intermediate_dim * dim, dim, intermediate_dim);

        // 3. SiLU activation on gate: silu(x) = x / (1 + exp(-x))
        // 4. Element-wise multiplication: hb = silu(gate) * up
        for (int i = 0; i < intermediate_dim; i++) {
            float g = s->hb[i];
            float silu_g = g / (1.0f + std::exp(-g));
            s->hb[i] = silu_g * s->hb2[i];
        }

        // 5. Down projection: xb = hb * w_down
        matmul(s->xb, s->hb, w->w_down + l * dim * intermediate_dim, intermediate_dim, dim);

        // --- Residual Connection 2 ---
        // x = x + FFN_Output
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }

    // Step 3: Final Layer Normalization
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);

    // Step 4: LM Head Classifier (Logits generation for all vocab tokens)
    matmul(s->logits, s->x, w->w_cls, dim, p->vocab_size);
}

// ----------------------------------------------------------------------------
// Sampling: Greedy Search (Argmax)
// ----------------------------------------------------------------------------
int sample_argmax(float* logits, int vocab_size) {
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ----------------------------------------------------------------------------
// Model Weights Loader
// ----------------------------------------------------------------------------
void load_weights(const std::string& model_path, Config* config, TransformerWeights* w, std::vector<float>& weights_data) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Model file open nahi ho payi: " << model_path << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Header read karo
    int magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(int));
    if (magic != 0x6c6c6133) {
        std::cerr << "Error: Invalid model magic number!\n";
        std::exit(EXIT_FAILURE);
    }

    file.read(reinterpret_cast<char*>(&config->dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&config->intermediate_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&config->n_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&config->n_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&config->n_kv_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&config->vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&config->seq_len), sizeof(int));

    std::cout << "Model Architecture Loaded:\n";
    std::cout << " - Dim: " << config->dim << "\n";
    std::cout << " - Intermediate Dim: " << config->intermediate_dim << "\n";
    std::cout << " - Layers: " << config->n_layers << "\n";
    std::cout << " - Heads: " << config->n_heads << " (KV: " << config->n_kv_heads << ")\n";
    std::cout << " - Vocab Size: " << config->vocab_size << "\n";

    // File remaining size calculate karo
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t data_size = file_size - (8 * sizeof(int));
    size_t num_floats = data_size / sizeof(float);
    file.seekg(8 * sizeof(int), std::ios::beg);

    std::cout << "Loading " << (file_size / (1024 * 1024)) << " MB of weights into RAM...\n";
    weights_data.resize(num_floats);
    file.read(reinterpret_cast<char*>(weights_data.data()), data_size);
    file.close();

    // Pointers assign karein
    float* ptr = weights_data.data();

    w->token_embedding_table = ptr;
    ptr += (size_t)config->vocab_size * config->dim;

    int head_dim = config->dim / config->n_heads;
    int kv_dim = config->n_kv_heads * head_dim;

    w->rms_att_weight = ptr;
    ptr += (size_t)config->n_layers * config->dim;

    w->wq = ptr;
    ptr += (size_t)config->n_layers * config->dim * config->dim;

    w->wk = ptr;
    ptr += (size_t)config->n_layers * kv_dim * config->dim;

    w->wv = ptr;
    ptr += (size_t)config->n_layers * kv_dim * config->dim;

    w->wo = ptr;
    ptr += (size_t)config->n_layers * config->dim * config->dim;

    w->rms_ffn_weight = ptr;
    ptr += (size_t)config->n_layers * config->dim;

    w->w_gate = ptr;
    ptr += (size_t)config->n_layers * config->intermediate_dim * config->dim;

    w->w_up = ptr;
    ptr += (size_t)config->n_layers * config->intermediate_dim * config->dim;

    w->w_down = ptr;
    ptr += (size_t)config->n_layers * config->dim * config->intermediate_dim;

    w->rms_final_weight = ptr;
    ptr += config->dim;

    w->w_cls = ptr;
    std::cout << "[+] All weights assigned successfully!\n";
}

// ----------------------------------------------------------------------------
// RunState Memory Allocator
// ----------------------------------------------------------------------------
void allocate_run_state(RunState* s, Config* p) {
    int kv_dim = p->n_kv_heads * (p->dim / p->n_heads);

    s->x = new float[p->dim];
    s->xb = new float[p->dim];
    s->xb2 = new float[p->dim];
    s->hb = new float[p->intermediate_dim];
    s->hb2 = new float[p->intermediate_dim];
    s->q = new float[p->dim];
    s->k = new float[kv_dim];
    s->v = new float[kv_dim];
    s->att = new float[p->n_heads * p->seq_len];
    s->logits = new float[p->vocab_size];

    // KV Cache allocation
    size_t kv_cache_size = (size_t)p->n_layers * p->seq_len * kv_dim;
    s->key_cache = new float[kv_cache_size]();
    s->value_cache = new float[kv_cache_size]();
}

// ----------------------------------------------------------------------------
// Main Execution Function
// ----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string model_path = "llama3_3b.bin";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "========================================\n";
    std::cout << " Llama 3.2 3B Inference Engine (C++)\n";
    std::cout << "========================================\n";

    Config config;
    TransformerWeights weights;
    std::vector<float> weights_data;

    // Model load karo
    load_weights(model_path, &config, &weights, weights_data);

    // Activations aur KV Cache allocate karo
    RunState state;
    allocate_run_state(&state, &config);

    // Dummy Prompt token sequence (Example: [128000, 791, 14016] -> "<|begin_of_text|> Hello world")
    // Real pipeline me tokenizer.json se text -> token convert kiya jata hai
    std::vector<int> prompt_tokens = {128000, 9906, 1917, 30}; // Example prompt tokens
    int num_prompt_tokens = prompt_tokens.size();
    int max_new_tokens = 20;

    std::cout << "\n[Inference Starting... Generating " << max_new_tokens << " tokens]\n";

    int token = prompt_tokens[0];
    int pos = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (pos < num_prompt_tokens + max_new_tokens) {
        // Forward pass: token aur current position pass karein
        forward(&config, &weights, &state, token, pos);

        int next_token;
        if (pos < num_prompt_tokens - 1) {
            // Prompt ingestion phase: agla token prompt se hi aayega
            next_token = prompt_tokens[pos + 1];
        } else {
            // Generation phase: model ke output logits se next token sample karo (Argmax)
            next_token = sample_argmax(state.logits, config.vocab_size);
            std::cout << "Token ID: " << next_token << " " << std::flush;
        }

        token = next_token;
        pos++;

        // Stop generation agar End-Of-Sequence token aa jaye
        if (pos >= num_prompt_tokens && (token == 128001 || token == 128009)) {
            std::cout << "\n[EOS reached]";
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "\n\nInference completed in " << elapsed_sec << " seconds.\n";

    return 0;
}
