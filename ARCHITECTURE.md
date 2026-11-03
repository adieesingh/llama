# Llama 3.2 3B: Under The Hood Architecture Deep Dive

Yeh document explain karta hai ki **Llama 3.2 3B** ke andar Silicon, Memory, aur Tensor level par **architecture ke andar exactly kya hota hai**.

---

## 🏭 High-Level Mental Model

Sabse aasan tareeqa is architecture ko samajhne ka yeh hai:

> **"Llama ek 28-Floor (Layers) ki Factory hai, aur aapka Token ek Conveyor Belt par chalne wala 3072-dimensional Vector hai."**

```
+--------------------------------------------------------------------------+
|                        RAW TEXT: "The sky is"                            |
+--------------------------------------------------------------------------+
                                     │ (Tokenizer)
                                     ▼
                        Token IDs: [791, 6422, 374]
                                     │
                                     ▼
           Embedding Lookup (Table of 128,256 × 3072 floats)
                                     │
                                     ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                     THE RESIDUAL STREAM (dim = 3072)                     ║
║  (Ek continuous vector jo layer 1 se 28 tak travel karta hai)            ║
╚══════════════════════════════════════════════════════════════════════════╝
                                     │
             ┌───────────────────────┴───────────────────────┐
             │       LAYER 1 TO 28 (Har Layer me 2 Blocks)   │
             │                                               │
             │   1. Attention Block (Communication):         │
             │      "Dusre tokens se context mangwana"       │
             │      • RMSNorm -> Q, K, V -> RoPE             │
             │      • GQA (24 Q heads : 8 KV heads)          │
             │      • Softmax -> KV Cache Lookup             │
             │      • Residual Add: x = x + Attention_Out    │
             │                                               │
             │   2. SwiGLU FFN Block (Computation / Memory): │
             │      "Apne internal facts/knowledge recall"   │
             │      • RMSNorm -> Gate & Up Projections       │
             │      • SiLU Activation -> Down Projection     │
             │      • Residual Add: x = x + FFN_Out          │
             └───────────────────────┬───────────────────────┘
                                     │ (Repeat 28 Times)
                                     ▼
                            Final RMSNorm (3072)
                                     │
                                     ▼
               LM Head Projection (3072 -> 128,256 Vocab Logits)
                                     │
                                     ▼
                  Argmax / Sampling: Next Token = "blue" (ID: 6411)
```

---

## 1. Residual Stream: Model ki "Central Highway"

Poore transformer ka sabse critical concept hai **Residual Stream**. 

Jab token embedding lookup hoti hai, toh token ek **3072 numbers ki list (`float32`)** ban jata hai. 
* Yeh 3072 numbers token ka **"Current State of Mind"** hote hain.
* Shuruat me isme sirf word ka basic/generic meaning hota hai (e.g., "Apple" ek fruit ya company hai).
* **Layers is vector ko replace nahi karti**, balki isme thoda-thoda information **Add (`+`)** karti chali jati hain:
  $$x_{\text{new}} = x_{\text{old}} + \Delta x$$

28 layers ke baad, yeh 3072 numbers itne rich ho jate hain ki isme grammar, context, reasoning, aur agla word kya hoga — sab encode ho chuka hota hai.

---

## 2. Block 1: Self-Attention (Token-to-Token Communication)

Transformer me **Attention ka kaam hai tokens ke beech baatcheet karwana**. Akela "bank" word kuch nahi batata, lekin agar pichle tokens "river" ya "money" hain, toh attention us meaning ko extract karta hai.

### A. Q, K, V ka Asli Matlab (Analogy)
Architecture level par, model current token ke vector se 3 cheezein calculate karta hai:
1. **Query ($Q$)**: *"Mujhe kis information ki talaash hai?"* (e.g., Verb dhoond raha hai ki Subject kaun hai).
2. **Key ($K$)**: *"Mere paas kya information hai?"* (e.g., "Main ek subject hoon").
3. **Value ($V$)**: *"Agar meri Key match ho gayi, toh mujhse yeh actual content le lo."*

### B. RoPE (Rotary Position Embeddings) — Space aur Geometry
Attention ko sequence ka order pata nahi hota ("Dog bites Man" vs "Man bites Dog").
* Llama 3.2 har feature pair ko ek 2D vector $(x, y)$ manta hai aur usse clock ki suiyon ki tarah **angle $\theta$ par rotate** kar deta hai.
* Jab token 1 ki Query aur token 5 ki Key ka dot product hota hai:
  $$\text{Rotation}(Q, \text{pos}=5) \cdot \text{Rotation}(K, \text{pos}=1) = \text{Relative Distance } (5 - 1 = 4)$$
* Model ko absolute position yaad rakhne ki zaroorat nahi padti; dot product khud-b-khud unke beech ka **distance ($4$)** calculate kar leta hai.

### C. GQA (Grouped-Query Attention) — Silicon & Memory Bottleneck Solution
Llama 3.2 3B me:
* **24 Query Heads** hain.
* Lekin sirf **8 Key/Value Heads** hain!

> **Kyu kiya gaya aisa?**  
> LLM run karte waqt GPU/CPU ki computing power (FLOPs) problem nahi hoti, balki **RAM se data transfer ki speed (Memory Bandwidth)** bottleneck hoti hai.  
> Har token par poora KV-Cache RAM se CPU registers me lana padta hai. 8 KV heads rakhne se KV-Cache ka size **3 guna chhota ($24 / 8 = 3$)** ho jata hai, jisse model 3x tez generate karta hai!

```
Query Heads (24):  [Q0, Q1, Q2]   [Q3, Q4, Q5]  ...  [Q21, Q22, Q23]
                         \ /             \ /                 \ /
KV Heads (8):           [KV 0]          [KV 1]   ...        [KV 7]
```

### D. KV-Cache — Past Tokens ka Dimag
Jab hum 100th token generate kar rahe hote hain:
* Hame token 1 se 99 tak ke Query vectors ki zaroorat **nahi** hoti.
* Hame sirf unke **Keys ($K$) aur Values ($V$)** chahiye hote hain taaki naya token unpar dhyan (attention) de sake.
* Isliye hum unhe RAM me save kar lete hain (`RunState.key_cache`). Agar KV Cache na ho, toh har naye word ke liye pura paragraph shuru se calculate karna padega ($O(N^2)$ lag jayega).

---

## 3. Block 2: SwiGLU Feed-Forward Network (The Knowledge Vault)

Attention sirf information ko **ek token se dusre token me move** karta hai. Lekin model ke facts aur world knowledge kahan store hote hain? (Jaise *"Paris is the capital of France"*).

Woh store hote hain **FFN (Feed-Forward Network)** ke weights me!

### SwiGLU Architecture:
Llama standard neural networks ki tarah simple linear layers use nahi karta, yeh **SwiGLU** use karta hai:
```
             Input x (dim = 3072)
               /               \
              /                 \
       w_gate [8192 × 3072]     w_up [8192 × 3072]
            │                        │
            ▼                        │
          SiLU                       │
    (Swish Activation)               │
            │                        │
            └───────────┬────────────┘
                        ▼
                 Element-wise (×)
                        │
                        ▼ (intermediate = 8192)
                     w_down [3072 × 8192]
                        │
                        ▼
             Output (dim = 3072)
```

1. **Expansion ($3072 \to 8192$)**: Vector ko achanak 8192 dimensions me expand kiya jata hai. High dimension space me model facts ko clearly separate aur correlate kar sakta hai.
2. **The Gate Mechanism**: `w_gate` ek gate ki tarah act karta hai. `SiLU(Gate) * Up` me gate decide karta hai ki 8192 facts me se kaun se activate honge aur kaun se block (zero) honge.
3. **Compression ($8192 \to 3072$)**: Relevant information ko wapas original 3072 dimension me pack karke residual stream me add kar diya jata hai.

---

## 4. Hardware Reality: Prefill vs Decode Phase

Inference me do alag-alag phases hote hain jo hardware ko alag tareeqe se stress karte hain:

| Feature | 1. Prefill Phase (Prompt Processing) | 2. Decode Phase (Token Generation) |
| :--- | :--- | :--- |
| **Input** | Poora prompt ek saath (`"Translate this to French..."`) | Sirf 1 pichla token |
| **Math Type** | **Matrix-Matrix Multiplication (GEMM)** | **Matrix-Vector Multiplication (GEMV)** |
| **Bottleneck** | **Compute-Bound** (CPU/GPU ALU units 100% busy hote hain) | **Memory-Bound** (RAM ki read speed par atka hota hai) |
| **KV Cache** | Saare prompt tokens ka cache ek saath banta hai | Har step par sirf 1 naya KV vector append hota hai |

> **Under the Hood Insight:**  
> Decode phase me model ke pure 3 Billion parameters (~6 GB to 12 GB weights) ko **har single token generate karne ke liye** RAM se CPU cache me laana padta hai! Yahi wajah hai ki LLM generation speed aapke RAM bandwidth (GB/s) par depend karti hai, na ki CPU GHz par.

---

## 5. Output: Logits se Human Word tak

28 layers complete hone ke baad hamare paas ek final 3072-dimensional vector hota hai.
1. **Classifier (`lm_head`)**: Is 3072 vector ko model ke vocabulary matrix ($128,256 \times 3072$) se multiply karte hain.
2. **128,256 Scores (Logits)**: Dictionary ke har possible token ko ek score milta hai:
   * Token `" blue"`: `18.4`
   * Token `" dark"`: `12.1`
   * Token `" banana"`: `-4.8`
3. **Argmax / Softmax**: Sabse highest score wale token ID (`blue`) ko select kar liya jata hai aur output screen par print kar diya jata hai.
4. Yahi naya token ab input ban kar agle cycle me chala jata hai!
