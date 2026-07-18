# GPT-2 desde cero en C++

Implementación completa de GPT-2 (124M) en C++ puro. Sin PyTorch, sin TensorFlow, sin ninguna librería de machine learning. Solo la STL y matemáticas.

Carga los pesos reales de OpenAI, ejecuta el forward pass completo del transformer y genera texto coherente.

**Estado:** Fase 0 completada — el modelo genera texto.

---

## Qué hace

```
Prompt: "The quick brown fox"

Greedy:      "The quick brown foxes are a great way to get a little
              bit of a kick out of your dog."

Temp 0.8:    "The quick brown foxes that lived at the farm of William
              T. T. Smith, on the corner of 2nd and 2nd Streets..."
```

---

## Estructura

### Headers (el motor)

| Archivo | Qué contiene |
|---|---|
| `matrix.h` | Clase Matrix (row-major), matmul, add, transpose |
| `safetensors.h` | Lector del formato .safetensors con mini-parser JSON propio |
| `layers.h` | Embeddings (token + posición) y LayerNorm |
| `softmax.h` | Softmax por filas con estabilidad numérica |
| `attention.h` | Máscara causal, atención de una cabeza, multi-head, capa Conv1D |
| `block.h` | GELU, feed-forward, bloque transformer completo |
| `gpt2.h` | El modelo: carga los 160 tensores y ejecuta el forward pass |
| `generacion.h` | Tokenizer (decodificación byte-level BPE) y sampling |

### Programas (solo uno con `main` a la vez)

| Archivo | Para qué |
|---|---|
| `generar_texto.cpp` | **El principal.** Genera texto (greedy y temperatura+top-k) |
| `inspeccionar_gpt2.cpp` | Lista los 160 tensores del modelo con sus formas |
| `prueba_final.cpp` | Verifica que el modelo predice bien (números, alfabeto) |
| `diagnostico_capas.cpp` | "Logit lens": magnitud de la señal capa por capa. Para depurar |
| `probar_gpt2.cpp` | Prueba con un caso específico |
| `test_*.cpp` | Tests unitarios de cada pieza |

---

## Setup

### 1. Requisitos

- Visual Studio Community con workload **"Desarrollo para el escritorio con C++"**
  (o cualquier compilador con C++17)
- ~2 GB de espacio libre
- Configuración **Versión (Release)** y **x64** — en Debug es 10-20x más lento

### 2. Descargar el modelo

Los pesos y el vocabulario no están en el repo (pesan demasiado). Bájalos con PowerShell:

```powershell
# Modelo: 548 MB
Invoke-WebRequest -Uri "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors" -OutFile "model.safetensors"

# Vocabulario: ~1 MB
Invoke-WebRequest -Uri "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json" -OutFile "vocab.json"
```

Verifica los tamaños:

```powershell
(Get-Item "model.safetensors").Length   # ~548000000
(Get-Item "vocab.json").Length          # ~1000000
```

Si `model.safetensors` pesa unos pocos KB, bajaste el pointer file de Git LFS en vez del modelo. Usa la URL con `/resolve/main/` (no `/blob/main/`).

### 3. Ajustar rutas

Dentro de cada `.cpp` está la ruta a los archivos del modelo. Cámbiala a donde los guardaste:

```cpp
std::string ruta_modelo = "C:\\ruta\\a\\model.safetensors";
std::string ruta_vocab  = "C:\\ruta\\a\\vocab.json";
```

Las barras invertidas van dobles (`\\`), o usa barras normales (`/`) que también funcionan en Windows.

### 4. Compilar

**Visual Studio:** crea un proyecto de consola C++, agrega todos los `.h` y **un solo** `.cpp` con `main`. Compila en Versión x64.

**Terminal (g++):**
```bash
g++ -std=c++17 -O2 generar_texto.cpp -o generar_texto
./generar_texto
```

---

## Arquitectura implementada

GPT-2 small (124M parámetros):

```
tokens
  ↓
embeddings (wte + wpe)          [n × 768]
  ↓
┌─────────────────────────┐
│  bloque transformer ×12 │
│    LayerNorm            │
│    Multi-Head Attention │  (12 cabezas × 64 dim)
│    + residual           │
│    LayerNorm            │
│    Feed-Forward         │  (768 → 3072 → 768, GELU)
│    + residual           │
└─────────────────────────┘
  ↓
LayerNorm final
  ↓
proyección a logits (weight tying con wte)   [n × 50257]
  ↓
sampling → siguiente token
```

Hiperparámetros: 12 capas, 12 cabezas, d_model 768, d_ff 3072, vocab 50257, contexto máx 1024.

---

## Notas técnicas

**Layout Conv1D.** GPT-2 guarda `c_attn`, `c_proj` y `c_fc` ya traspuestos. La operación es `entrada · W + bias`, **sin** transponer W. Si transpones por instinto, las dimensiones no cuadran.

**Máscara causal.** Se pone `-1e10` (no 0) donde `j > i`, porque tras el softmax `exp(-1e10) ≈ 0`. Con 0 esas posiciones sí recibirían peso.

**Softmax estable.** Restar el máximo de la fila antes de exponenciar no es opcional: sin eso, `exp(1000)` es infinito y todo se vuelve NaN.

**Los logits negativos no son un bug.** En un softmax solo importan las *diferencias* entre logits, no sus valores absolutos. Un offset constante en todos los logits no cambia nada.

**Tokenizer byte-level.** El vocabulario guarda los tokens "disfrazados": el espacio es `Ġ` (U+0120), el salto de línea es `Ċ` (U+010A). Hay que deshacer ese mapeo para obtener el texto real.

---

## Lecciones de depuración

Dos que costaron tiempo y vale la pena recordar:

1. **Elige casos de prueba con respuesta objetivamente conocida.** Probar que GPT-2 predice "Paris" para "The capital of France is" fue mala idea: el modelo de 124M tiene poco recuerdo factual y no lo predice de forma confiable. Los patrones deterministas (secuencias numéricas, el alfabeto) sí funcionan como prueba.

2. **Mide la cantidad correcta.** Los logits absolutos no dicen nada sobre si el modelo funciona. Lo que importa es la confianza: cuánto sobresale el ganador sobre el resto.

Cuando el modelo genere basura, el bug casi siempre está en la atención (máscara mal, Q·Kᵀ transpuesto al revés) o en cómo se cargaron los pesos. Usa `diagnostico_capas.cpp` para ver dónde se degrada la señal.

---

## Roadmap

- [x] **Fase 0** — GPT-2 desde cero, genera texto coherente
- [ ] **Fase 1** — Optimización: KV cache, multithreading, SIMD
- [ ] **Fase 2** — Cuantización y formato GGUF
- [ ] **Fase 3** — Qwen2.5 (GQA, RoPE, RMSNorm) + CUDA

El objetivo final es un motor de inferencia propio para un asistente personal, sin depender de Ollama ni de ninguna librería externa.
