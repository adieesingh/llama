# Llama 3.2 3B Inference Engine (Python Downloader + Pure C++ Runtime)

Yeh repository Llama 3.2 3B model ke weights ko Hugging Face se download karke ek clean binary format me export karne aur bina kisi heavy external framework (jaise PyTorch C++, ONNX, ya CUDA toolkit) ke **Pure C++17** me inference run karne ka complete implementation aur explanation provide karta hai.

---

## 📌 Table of Contents
1. [Llama 3.2 3B Architecture Specs](#llama-32-3b-architecture-specs)
2. [Files Overview](#files-overview)
3. [Python Code: download_and_export.py (Line-by-Line Explanation)](#1-python-code-download_and_exportpy)
4. [C++ Code: llama3_inference.cpp (Line-by-Line Explanation)](#2-c-code-llama3_inferencecpp)
   - [Data Structures](#a-data-structures)
   - [Math Kernels](#b-math-kernels)
   - [Transformer Forward Pass](#c-transformer-forward-pass)
   - [Generation Loop & Sampling](#d-sampling--auto-regressive-generation-loop)
5. [Setup & Execution Guide](#3-setup--execution-guide)

---

## 📐 Llama 3.2 3B Architecture Specs

Llama 3.2 3B model Meta ka state-of-the-art compact LLM hai:
* **Hidden Dimension (`dim`)**: `3072` (Har token ka internal representation vector size).
* **Intermediate Dimension (`intermediate_dim`)**: `8192` (Feed-Forward Network ka expanded layer size).
* **Layers (`n_layers`)**: `28` (Transformer blocks ki total sankhya).
* **Query Attention Heads (`n_heads`)**: `24` ($\text{Head Dim} = 3072 / 24 = 128$).
* **Key/Value Attention Heads (`n_kv_heads`)**: `8` (Grouped-Query Attention: 3 Query Heads ek KV Head share karte hain).
* **Vocabulary Size (`vocab_size`)**: `128,256` (TikToken / Llama 3 BPE tokenizer).
* **RoPE Base Theta**: `500,000.0` (Positional encoding frequency base).
* **Normalization**: RMSNorm with $\epsilon = 10^{-5}$.
* **Activation**: SwiGLU (Swish Gated Linear Unit).

---

## 📁 Files Overview

1. **`download_and_export.py`**: Hugging Face se model files (.safetensors aur config.json) download karta hai aur unhe ek single optimized binary file (`llama3_3b.bin`) me sequentially dump karta hai.
2. **`llama3_inference.cpp`**: Standalone C++17 source code jo bina kisi dependency ke binary file ko read karta hai, memory allocate karta hai, KV cache maintain karta hai aur auto-regressive generation chalata hai.

---

## 1. Python Code: `download_and_export.py`

### 1.1 Imports aur Setup
```python
import os
import struct
import json
from huggingface_hub import snapshot_download
from safetensors import safe_open
import numpy as np
```
* **Kyu kiya?**:
  * `snapshot_download`: Hugging Face repo se pure model weights ko download karne ke liye.
  * `safe_open`: Hugging Face ke `.safetensors` files ko memory-efficient aur fast tarike se read karne ke liye bina PyTorch load kiye.
  * `struct`: C++ compatible binary headers pack karne ke liye.
  * `numpy`: Weights ko float32 array me convert karke raw bytes me convert karne ke liye.

### 1.2 Model IDs aur Paths
```python
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LOCAL_DIR = "./llama-3.2-3b-weights"
OUTPUT_BIN = "./llama3_3b.bin"
```
* **Kyu kiya?**: Target model repository aur local storage location set kiya. Llama-3.2 ek gated model hai, isliye Hugging Face CLI par login hona zaroori hai.

### 1.3 Download Function
```python
def download_llama_weights():
    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    return snapshot_path
```
* **Kyu kiya?**: Sirf zaroori files (`.safetensors` aur `config.json`) download hoti hain. Non-essential framework files (jaise TensorFlow ya Flax) ko ignore karke disk space bacha li jati hai.

### 1.4 Binary Export aur Header Packing
```python
magic = 0x6c6c6133  # 'lla3' in hex
header = struct.pack(
    "iiiiiiii",
    magic, dim, intermediate_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
)
f_out.write(header)
```
* **Kyu kiya?**: C++ program ko model ke dimensions pata hone chahiye taaki memory dynamically allocate ho sake.
  * `magic` number (`lla3`) yeh verify karta hai ki binary file valid hai aur corrupt nahi hui.
  * Baaki 7 integers C++ ko batate hain ki dimension kitna hai, layers kitni hain, etc.

### 1.5 Sequential Tensor Dumping
```python
write_tensor("model.embed_tokens.weight")
for i in range(n_layers):
    write_tensor(f"model.layers.{i}.input_layernorm.weight")
    write_tensor(f"model.layers.{i}.self_attn.q_proj.weight")
    write_tensor(f"model.layers.{i}.self_attn.k_proj.weight")
    write_tensor(f"model.layers.{i}.self_attn.v_proj.weight")
    write_tensor(f"model.layers.{i}.self_attn.o_proj.weight")
    write_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
    write_tensor(f"model.layers.{i}.mlp.gate_proj.weight")
    write_tensor(f"model.layers.{i}.mlp.up_proj.weight")
    write_tensor(f"model.layers.{i}.mlp.down_proj.weight")
write_tensor("model.norm.weight")
write_tensor("lm_head.weight")
```
* **Kyu kiya?**: Safetensors dictionary format me hota hai jise C++ me bina heavy JSON parser ke read karna mushkil hota hai. Isliye hum saare tensors ko ek fixed, predefined sequence me binary format me likh dete hain. C++ program ek hi `fread` call me poore model ko seedha RAM me load kar leta hai.

---

## 2. C++ Code: `llama3_inference.cpp`

### A. Data Structures

#### 1. Config Struct
```cpp
struct Config {
    int dim;              // 3072
    int intermediate_dim; // 8192
    int n_layers;         // 28
    int n_heads;          // 24
    int n_kv_heads;       // 8
    int vocab_size;       // 128256
    int seq_len;          // 2048
};
```
* **Kyu?**: Model architecture ke dimensions ko store karta hai.

#### 2. TransformerWeights Struct
```cpp
struct TransformerWeights {
    float* token_embedding_table;
    float* rms_att_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* rms_ffn_weight;
    float* w_gate;
    float* w_up;
    float* w_down;
    float* rms_final_weight;
    float* w_cls;
};
```
* **Kyu?**: Model weights RAM me ek giant float array me load hote hain. Yeh struct un weights ke starting addresses (pointers) hold karta hai. Isme zero memory overhead hota hai.

#### 3. RunState Struct
```cpp
struct RunState {
    float *x;           // Current activation vector [dim]
    float *xb;          // Normalized activation buffer [dim]
    float *xb2;         // Second buffer for residual addition [dim]
    float *hb;          // SwiGLU activation buffer [intermediate_dim]
    float *hb2;         // SwiGLU gate buffer [intermediate_dim]
    float *q;           // Query vector [3072]
    float *k;           // Key vector [1024]
    float *v;           // Value vector [1024]
    float *att;         // Attention scores buffer [n_heads, seq_len]
    float *logits;      // Output token probabilities/scores [vocab_size]
    float *key_cache;   // Past tokens ke Keys
    float *value_cache; // Past tokens ke Values
};
```
* **Kyu?**: Inference ke time har token generate karne ke liye temporary memory chahiye hoti hai. Agar hum loop ke andar baar-baar `new` ya `malloc` karenge to program slow ho jayega. Isliye saare buffers pehle hi allocate ho jate hain.

---

### B. Math Kernels

#### 1. Root Mean Square Normalization (`rmsnorm`)
```cpp
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
```
* **Kyu use kiya?**:
  * Standard LayerNorm me mean ($\mu$) aur variance dono nikalna padta hai.
  * Llama me **RMSNorm** use hota hai jisme mean subtract karne ki zaroorat nahi hoti:
    $$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}, \quad \text{out}_i = \frac{x_i}{\text{RMS}(x)} \times w_i$$
  * **Fayda**: Faster execution on hardware bina kisi model accuracy degradation ke.

#### 2. Matrix-Vector Multiplication (`matmul`)
```cpp
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
```
* **Kyu use kiya?**:
  * LLM inference me har projection ($Q, K, V, O$, FFN Gate, Up, Down, LM Head) ek Matrix-Vector product hoti hai ($y = W \cdot x$).
  * `#pragma omp parallel for`: OpenMP multi-threading se CPU ke saare physical aur virtual cores ko use karta hai taaki matrix calculation lightning fast ho.

#### 3. Rotary Position Embeddings (`apply_rope`)
```cpp
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
```
* **Kyu use kiya?**:
  * Attention mechanism order-agnostic hoti hai (usse pata nahi hota kaun sa word pehle aaya aur kaun sa baad me).
  * RoPE har feature pair $(v_0, v_1)$ ko 2D rotation matrix se rotate kar deta hai:
    $$\begin{pmatrix} v_0' \\ v_1' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} v_0 \\ v_1 \end{pmatrix}$$
  * **Theta = 500,000 kyu?**: Llama 3 me context length bohot lambi hoti hai (128k tak), isliye higher base theta use kiya gaya hai taaki angles saturate na hon.

#### 4. Softmax Function
```cpp
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
```
* **Kyu use kiya?**: Attention scores ko probabilities me convert karta hai (sum = 1.0). `max_val` subtract karne se `std::exp` me numerical overflow (float `inf`) nahi hota.

---

### C. Transformer Forward Pass

Forward pass ek single token ko position `pos` par process karta hai:

1. **Token Embedding Lookup**:
   ```cpp
   float* content_row = &(w->token_embedding_table[token * dim]);
   std::memcpy(s->x, content_row, dim * sizeof(float));
   ```
   * Input token integer ID hota hai. Embedding table se us token ka 3072-dim vector `s->x` me copy ho jata hai.

2. **Pre-Attention Normalization**:
   ```cpp
   rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);
   ```

3. **$Q, K, V$ Projections**:
   ```cpp
   matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
   matmul(s->k, s->xb, w->wk + l * kv_dim * dim, dim, kv_dim);
   matmul(s->v, s->xb, w->wv + l * kv_dim * dim, dim, kv_dim);
   ```
   * Query ($Q$) size: $3072$ ($24 \text{ heads} \times 128$).
   * Key ($K$) aur Value ($V$) size: sirf $1024$ ($8 \text{ heads} \times 128$) kyunki yeh **Grouped-Query Attention** use kar raha hai!

4. **RoPE & KV-Cache**:
   ```cpp
   apply_rope(s->q, pos, head_dim, p->n_heads);
   apply_rope(s->k, pos, head_dim, p->n_kv_heads);

   int cache_offset = l * p->seq_len * kv_dim + pos * kv_dim;
   std::memcpy(s->key_cache + cache_offset, s->k, kv_dim * sizeof(float));
   std::memcpy(s->value_cache + cache_offset, s->v, kv_dim * sizeof(float));
   ```
   * Current token ke Key aur Value ko cache me save kar liya jata hai taaki agle tokens aane par dobara recalculate na karna pade ($O(N^2) \to O(N)$ speedup).

5. **Grouped-Query Attention (GQA)**:
   ```cpp
   int kv_mul = p->n_heads / p->n_kv_heads; // 24 / 8 = 3
   int kv_h = h / kv_mul; // Query head 'h' shares KV head 'kv_h'
   ```
   * Har 3 Query Heads ek hi Key/Value head ko share karte hain. Isse memory bandwidth 3x bach jati hai aur cache chhota rehta hai.
   * Causal attention dot-product $Q \cdot K^T / \sqrt{d}$ calculate hota hai tokens $0 \dots pos$ tak, fir softmax aur Value accumulation: $\text{xb} = \sum \text{score} \times V$.

6. **Attention Output Projection & Residual Connection 1**:
   ```cpp
   matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
   for (int i = 0; i < dim; i++) {
       s->x[i] += s->xb2[i]; // x = x + Attn_Out
   }
   ```
   * Attention output ko residual stream me add kar diya jata hai taaki initial information loose na ho.

7. **SwiGLU Feed-Forward Network (FFN)**:
   ```cpp
   rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);
   matmul(s->hb,  s->xb, w->w_gate + l * intermediate_dim * dim, dim, intermediate_dim);
   matmul(s->hb2, s->xb, w->w_up   + l * intermediate_dim * dim, dim, intermediate_dim);

   for (int i = 0; i < intermediate_dim; i++) {
       float g = s->hb[i];
       float silu_g = g / (1.0f + std::exp(-g));
       s->hb[i] = silu_g * s->hb2[i]; // SiLU(Gate) * Up
   }
   matmul(s->xb, s->hb, w->w_down + l * dim * intermediate_dim, intermediate_dim, dim);
   ```
   * **SwiGLU activation**: Gate aur Up projections ki element-wise multiplication hoti hai:
     $$\text{SwiGLU}(x) = (\text{SiLU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})) \cdot W_{\text{down}}$$
   * Yeh standard ReLU ya GELU se zyada expressive hota hai.

8. **Residual Connection 2**:
   ```cpp
   for (int i = 0; i < dim; i++) {
       s->x[i] += s->xb[i]; // x = x + FFN_Out
   }
   ```

9. **Final RMSNorm aur LM Head Classifier**:
   ```cpp
   rmsnorm(s->x, s->x, w->rms_final_weight, dim);
   matmul(s->logits, s->x, w->w_cls, dim, p->vocab_size);
   ```
   * 28 layers ke baad activation ko normalize karke vocabulary ($128,256$) ke saare tokens ke prediction scores (logits) generate kiye jate hain.

---

### D. Sampling & Auto-Regressive Generation Loop

```cpp
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
```
* **Kyu?**: Greedy decoding — vocabulary me jis token ka logit score sabse zyada hai, wahi hamara next predicted token hoga.

```cpp
while (pos < num_prompt_tokens + max_new_tokens) {
    forward(&config, &weights, &state, token, pos);

    int next_token;
    if (pos < num_prompt_tokens - 1) {
        next_token = prompt_tokens[pos + 1]; // Prompt Ingestion (Prefill)
    } else {
        next_token = sample_argmax(state.logits, config.vocab_size); // Generation (Decode)
        std::cout << "Token ID: " << next_token << " " << std::flush;
    }

    token = next_token;
    pos++;

    if (pos >= num_prompt_tokens && (token == 128001 || token == 128009)) {
        break; // Stop when End-Of-Sequence token is reached
    }
}
```
* **Kyu?**:
  1. **Prefill Phase**: Prompt ke saare tokens ek-ek karke feed hote hain taaki KV cache initialize ho jaye.
  2. **Decode Phase**: Model pichle predicted token ko lekar agla token predict karta hai.
  3. **EOS Check**: Agar `<|end_of_text|>` (`128001`) ya `<|eot_id|>` (`128009`) generate ho jaye to generation automatically stop ho jati hai.

---

## 3. Setup & Execution Guide

### Step 1: Python Dependencies Install Karein
```bash
pip install huggingface_hub safetensors numpy
```

### Step 2: Hugging Face Login (Zaroori for Llama 3.2)
Meta-Llama 3.2 gated repo hai, isliye Hugging Face par terms accept karke token se login karein:
```bash
huggingface-cli login
```

### Step 3: Model Download aur Binary Export
```bash
python download_and_export.py
```
Yeh script `llama-3.2-3b-weights` folder me weights download karegi aur `llama3_3b.bin` export karegi.

### Step 4: C++ Inference Compile & Run Karein

#### GCC / Clang (Linux, macOS, MinGW on Windows):
```bash
g++ -O3 -std=c++17 -fopenmp llama3_inference.cpp -o llama3
./llama3 llama3_3b.bin
```

#### MSVC (Visual Studio Developer Command Prompt on Windows):
```cmd
cl /O2 /std:c++17 /openmp llama3_inference.cpp /Fe:llama3.exe
llama3.exe llama3_3b.bin
```

---

## 🚀 Summary
* **No Bloat**: Kisi 100MB library ya heavyweight runtime ki zaroorat nahi hai.
* **Readable Math**: Transformer architecture ke saare operations (RMSNorm, RoPE, GQA Attention, SwiGLU) line-by-line transparent hain.
* **Hardware Friendly**: Multi-threaded execution via OpenMP for fast CPU inference.
