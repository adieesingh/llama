"""
download_and_export.py

1. Hugging Face se Llama-3.2-3B weights download karta hai.
2. Safetensors format ko extract karke ek clean binary file (model.bin) me export karta hai,
   taaki C++ me bina kisi external library ke seedha load kiya ja sake.
"""

import os
import struct
import json
from huggingface_hub import snapshot_download
from safetensors import safe_open
import numpy as np

# Step 1: Model ID aur Directory define karein
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  # Ya "meta-llama/Llama-3.2-3B"
LOCAL_DIR = "./llama-3.2-3b-weights"
OUTPUT_BIN = "./llama3_3b.bin"

def download_llama_weights():
    print(f"[*] Downloading {MODEL_ID} from Hugging Face...")
    print("[!] Note: Make sure you have logged in via `huggingface-cli login` and accepted the Llama license.")
    
    # Hugging Face se saare model files download karna
    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Sirf safetensors aur config chahiye
    )
    print(f"[+] Download complete at: {snapshot_path}")
    return snapshot_path

def export_to_binary(model_dir, output_bin):
    """
    Safetensors files ko read karke raw float32 binary format me save karta hai.
    C++ code is binary file ko direct memory-map ya fread karega.
    """
    print("[*] Reading config.json...")
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Llama 3.2 3B Hyperparameters
    vocab_size = config.get("vocab_size", 128256)
    dim = config.get("hidden_size", 3072)
    intermediate_dim = config.get("intermediate_size", 8192)
    n_layers = config.get("num_hidden_layers", 28)
    n_heads = config.get("num_attention_heads", 24)
    n_kv_heads = config.get("num_key_value_heads", 8)
    max_seq_len = 2048  # Inference ke liye maximum context length

    print(f"Model Architecture:")
    print(f" - Vocab Size: {vocab_size}")
    print(f" - Hidden Dim: {dim}")
    print(f" - Intermediate Dim: {intermediate_dim}")
    print(f" - Layers: {n_layers}")
    print(f" - Query Heads: {n_heads}")
    print(f" - KV Heads: {n_kv_heads}")

    # Safetensors files dhundho
    safetensor_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    safetensor_files.sort()

    # Saare tensors ko ek dictionary me index karein
    tensors = {}
    print(f"[*] Found {len(safetensor_files)} safetensors file(s). Loading index...")
    for sf in safetensor_files:
        with safe_open(sf, framework="numpy", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

    print(f"[+] Total tensors loaded: {len(tensors)}")

    # Binary file write karna shuru karein
    print(f"[*] Writing binary model file to {output_bin}...")
    with open(output_bin, "wb") as f_out:
        # Header write karein (Magic number + Config integers)
        magic = 0x6c6c6133  # 'lla3' in hex
        header = struct.pack(
            "iiiiiiii",
            magic,
            dim,
            intermediate_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            max_seq_len
        )
        f_out.write(header)

        # Helper function: Tensor ko float32 me convert karke write karna
        def write_tensor(name):
            if name not in tensors:
                raise KeyError(f"Tensor {name} not found in model weights!")
            arr = tensors[name].astype(np.float32)
            f_out.write(arr.tobytes())

        # 1. Token Embeddings: [vocab_size, dim]
        print("Writing embedding table...")
        write_tensor("model.embed_tokens.weight")

        # 2. Transformer Layer Weights (0 to n_layers-1)
        for i in range(n_layers):
            print(f"Writing layer {i+1}/{n_layers}...")
            # Attention RMSNorm
            write_tensor(f"model.layers.{i}.input_layernorm.weight")
            # Query, Key, Value, Output projection weights
            write_tensor(f"model.layers.{i}.self_attn.q_proj.weight")
            write_tensor(f"model.layers.{i}.self_attn.k_proj.weight")
            write_tensor(f"model.layers.{i}.self_attn.v_proj.weight")
            write_tensor(f"model.layers.{i}.self_attn.o_proj.weight")
            # Post-Attention / FFN RMSNorm
            write_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
            # Feed-Forward Network: Gate, Up, Down projections (SwiGLU)
            write_tensor(f"model.layers.{i}.mlp.gate_proj.weight")
            write_tensor(f"model.layers.{i}.mlp.up_proj.weight")
            write_tensor(f"model.layers.{i}.mlp.down_proj.weight")

        # 3. Final RMSNorm
        print("Writing final RMSNorm...")
        write_tensor("model.norm.weight")

        # 4. Output Classifier / LM Head: [vocab_size, dim]
        # (Llama me lm_head aksar embed_tokens se tied ya alag hota hai)
        if "lm_head.weight" in tensors:
            print("Writing LM Head...")
            write_tensor("lm_head.weight")
        else:
            print("LM Head tied to embed_tokens. Writing embed_tokens as lm_head...")
            write_tensor("model.embed_tokens.weight")

    print(f"[SUCCESS] Export completed! Model binary saved to: {output_bin}")

if __name__ == "__main__":
    # Step 1: Download weights (HF Token required for Llama-3.2)
    # model_dir = download_llama_weights()
    
    # Agar pehle se downloaded hai to aap local folder path de sakte hain:
    model_dir = LOCAL_DIR
    if os.path.exists(model_dir):
        export_to_binary(model_dir, OUTPUT_BIN)
    else:
        print(f"Directory {model_dir} not found. Running download...")
        download_llama_weights()
        export_to_binary(model_dir, OUTPUT_BIN)
