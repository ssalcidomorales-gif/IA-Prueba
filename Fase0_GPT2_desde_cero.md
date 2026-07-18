# Fase 0 — GPT-2 (124M) desde cero en C++

**Objetivo:** cargar los pesos de GPT-2 small, ejecutar el forward pass completo del transformer a mano, y generar texto coherente. Todo en C++, CPU, float32. Sin librerías de ML — solo tú, los números y las matemáticas.

**Cuándo lo lograste:** cuando le das un prompt y escupe texto que tiene sentido. Ese día entendiste el transformer por dentro.

**Ritmo:** igual que OpenGL — concepto el sábado, implementación el domingo. Estimado ~6-8 fines de semana para Fase 0.

**Regla de oro:** verifica cada pieza por separado antes de seguir. Un transformer con un bug en la capa 3 genera basura y no sabrás dónde está. Ve componente por componente, comparando contra valores conocidos.

---

## Por qué GPT-2 y no Qwen directo

GPT-2 es el "hola mundo" de implementar transformers desde cero:

- **Arquitectura simple:** attention estándar (sin GQA, sin RoPE), activación GELU, LayerNorm clásico.
- **Pesos en float32:** sin cuantización que decodificar. Eso lo dejamos para la Fase 2.
- **Todo documentado:** el modelo original de OpenAI, más re-implementaciones abiertas para comparar cada paso.
- **Pequeño:** 124M parámetros caben en RAM sin drama (~500 MB en float32).

Qwen2.5 añade GQA, RoPE, RMSNorm, cuantización y un tokenizer más complejo. Todo eso lo montas *después*, cuando ya domines el esqueleto. Empezar por Qwen sería como construir la SAP-1 arrancando por el pipeline.

---

## Referencias clave (tenlas abiertas siempre)

- **llm.c de Karpathy** — GPT-2 en C puro. *La* referencia. Lee `train_gpt2.c` (el forward pass está ahí completo y comentado).
- **"Let's build GPT" de Karpathy (YouTube)** — construye un GPT en Python paso a paso, explicando cada bloque. Míralo antes de codear.
- **"The Illustrated GPT-2" de Jay Alammar** — el attention explicado con dibujos.
- **Paper "Attention Is All You Need"** — la fuente. No para copiar, para entender de dónde sale todo.
- **cppreference.com** — para las partes de C++ moderno que uses.

Sugerencia: mira el video de Karpathy *primero*, entiende la versión Python, y luego traduces los conceptos a C++ tú mismo. No copies llm.c línea por línea — úsalo para desatorarte cuando algo no cuadre.

---

## Arquitectura de GPT-2 124M (la meta a construir)

```
Texto de entrada
     ↓
[Tokenizer BPE]  →  lista de token IDs
     ↓
[Token embeddings]  +  [Positional embeddings]   →  matriz [seq_len × 768]
     ↓
┌─────────────────────────────────┐
│  Bloque Transformer  × 12        │
│                                 │
│   LayerNorm                     │
│   Multi-Head Attention (12 cab) │
│   + residual                    │
│   LayerNorm                     │
│   Feed-Forward (768→3072→768)   │
│   + residual                    │
└─────────────────────────────────┘
     ↓
[LayerNorm final]
     ↓
[Proyección a vocabulario]  →  logits [seq_len × 50257]
     ↓
[Sampler]  →  siguiente token
```

Números de GPT-2 small que vas a necesitar:
- Dimensión del modelo (d_model): **768**
- Capas: **12**
- Cabezas de atención: **12** (cada una de dimensión 64)
- Dimensión del feed-forward: **3072** (4 × 768)
- Tamaño del vocabulario: **50257**
- Contexto máximo: **1024** tokens

---

## FIN DE SEMANA 1 — Fundamentos matemáticos y estructura del proyecto

**Sábado (concepto):**
- Repasa multiplicación de matrices: si no la tienes fresca a nivel de "lo puedo programar dormido", este es el momento. Todo el transformer es matmul.
- Entiende qué es un embedding: una tabla de búsqueda. Token ID → vector de 768 números.
- Mira el video de Karpathy "Let's build GPT" completo. No codees, solo entiende el flujo.

**Domingo (implementación):**
- Estructura del proyecto en Visual Studio o CMake.
- Implementa tu tipo básico de matriz/tensor. Empieza simple: una struct con un `std::vector<float>`, filas y columnas.
- Implementa y **testea** estas operaciones aisladas, cada una con un caso que verifiques a mano:
  - `matmul(A, B)` — multiplicación de matrices
  - `add(A, B)` — suma elemento a elemento
  - `transpose(A)`

**Checkpoint:** tienes una clase Matrix con matmul que da resultados correctos en ejemplos de 2×2 que calculaste a mano. Nada de modelo todavía — solo los ladrillos.

---

## FIN DE SEMANA 2 — Cargar los pesos y el tokenizer

**Sábado (concepto):**
- Entiende cómo se guardan los pesos de GPT-2. La forma más fácil: exportar los pesos de GPT-2 a un archivo binario plano con un script de Python (usando la versión de HuggingFace o los checkpoints originales), y luego leerlos en C++.
- Entiende BPE (Byte Pair Encoding) a alto nivel: cómo el texto se parte en tokens. No lo vas a implementar desde cero todavía — usarás las tablas de GPT-2.

**Domingo (implementación):**
- Script de Python (de un solo uso) que baja GPT-2 124M y vuelca cada tensor de pesos a un `.bin` que tú puedas leer. Guarda también las dimensiones.
- En C++: función que lee ese `.bin` y llena tus estructuras de Matrix con los pesos. Verifica leyendo el primer y último valor de un tensor y comparándolo con Python.
- Carga el vocabulario del tokenizer (los archivos `vocab.json` y `merges.txt` de GPT-2). De momento basta con poder convertir token IDs → texto (decodificar); el encode BPE completo lo puedes dejar para más tarde o portar.

**Checkpoint:** tu programa en C++ carga los ~124M de pesos sin crashear, y puedes imprimir valores de un tensor que coinciden con los de Python. Puedes convertir una lista de token IDs a texto legible.

---

## FIN DE SEMANA 3 — Embeddings y LayerNorm

**Sábado (concepto):**
- Token embedding + positional embedding: cómo se suman. El token te dice *qué* palabra, la posición te dice *dónde* está.
- LayerNorm: la fórmula exacta (resta media, divide por desviación estándar, escala y desplaza con parámetros aprendidos gamma/beta). Entiende *por qué* estabiliza.

**Domingo (implementación):**
- Función de embedding: token IDs → matriz [seq_len × 768] buscando en la tabla y sumando posiciones.
- Implementa `layernorm(x, gamma, beta)`. Testéala con un vector pequeño calculado a mano.

**Checkpoint:** metes una lista de tokens y sale la matriz de embeddings correcta. LayerNorm da los números que esperas en un caso chico.

---

## FIN DE SEMANA 4 — Self-Attention (el corazón)

**Sábado (concepto):**
- *Este es el fin de semana más importante.* Dedícale tiempo real.
- Entiende Q, K, V: cada token genera una query, una key y un value (multiplicando por matrices de pesos).
- Entiende el mecanismo: scores = Q·Kᵀ, escalar por √(dim), aplicar máscara causal (un token no ve el futuro), softmax, multiplicar por V.
- Entiende multi-head: haces esto 12 veces en paralelo con trozos de 64 dimensiones, y concatenas.
- Vuelve a "The Illustrated GPT-2" de Alammar tantas veces como necesites.

**Domingo (implementación):**
- Implementa `softmax` (con el truco de restar el máximo para estabilidad numérica). Testéala.
- Implementa la máscara causal.
- Implementa una sola cabeza de atención. Verifica dimensiones en cada paso.
- Extiéndelo a las 12 cabezas + la proyección de salida.

**Checkpoint:** una capa de atención que toma [seq_len × 768] y devuelve [seq_len × 768]. Aún no sabes si los *valores* son correctos (eso se ve al ensamblar todo), pero las dimensiones cuadran y no crashea.

---

## FIN DE SEMANA 5 — Feed-Forward y el bloque completo

**Sábado (concepto):**
- Feed-forward: dos capas lineales con una activación GELU en medio (768 → 3072 → 768).
- GELU: la fórmula (o su aproximación con tanh, que es la que usa GPT-2).
- Conexiones residuales: por qué se suma la entrada a la salida de cada sub-bloque. Y el orden exacto de LayerNorm/atención/residual en GPT-2 (pre-norm).

**Domingo (implementación):**
- Implementa GELU.
- Implementa el feed-forward completo.
- Ensambla el **bloque transformer completo**: LN → atención → residual → LN → FFN → residual. Respeta el orden exacto de GPT-2.

**Checkpoint:** un bloque transformer completo que entra y sale en [seq_len × 768]. Lo puedes llamar en bucle.

---

## FIN DE SEMANA 6 — Ensamblar el modelo y el primer forward pass

**Sábado (concepto):**
- Repasa el flujo completo de principio a fin (mira el diagrama de arriba).
- Entiende la proyección final: de 768 dimensiones a 50257 logits (uno por cada token posible del vocabulario). En GPT-2 se reutiliza la matriz de embeddings (weight tying).
- Entiende qué es un logit: un puntaje sin normalizar de qué tan probable es cada siguiente token.

**Domingo (implementación):**
- Apila los 12 bloques.
- Añade el LayerNorm final y la proyección a logits.
- Ejecuta el **primer forward pass completo** con un prompt corto.
- **Verificación crítica:** compara tus logits con los de la implementación de referencia (HuggingFace en Python) para el *mismo* input. Deben coincidir hasta varios decimales. Si no, tienes un bug — este es el momento de cazarlo, con todo aislado.

**Checkpoint:** metes tokens, sale un vector de 50257 logits que coincide con la referencia. **Si llegas aquí, ganaste** — el modelo funciona, solo falta convertir logits en texto.

---

## FIN DE SEMANA 7 — Sampling y generación de texto

**Sábado (concepto):**
- Cómo convertir logits en el siguiente token:
  - **Greedy:** toma el de mayor logit. Simple, determinista, aburrido.
  - **Temperatura:** divide los logits antes del softmax para controlar aleatoriedad.
  - **Top-k / Top-p:** limita a los candidatos más probables.
- El bucle autoregresivo: generas un token, lo agregas al input, vuelves a correr, repites.

**Domingo (implementación):**
- Implementa sampling greedy primero (más fácil de depurar).
- Implementa el bucle de generación: predecir → agregar token → repetir hasta N tokens o hasta el token de fin.
- Conecta el decodificador del tokenizer para ver texto real.
- **Momento de la verdad:** dale un prompt y mira qué genera.
- Añade temperatura y top-k para que el texto sea más natural.

**Checkpoint:** le das "The meaning of life is" y GPT-2 continúa con texto coherente en inglés. **Fase 0 completada.**

---

## FIN DE SEMANA 8 (colchón) — Limpieza y comprensión

- Refactoriza el código ahora que entiendes el todo.
- Añade comentarios que expliquen *por qué*, no *qué*.
- Mide cuántos tokens por segundo generas (será lento — normal).
- Escribe en un archivo tus notas de qué aprendiste y qué te costó. Te servirá en las fases siguientes.
- Juega: cambia la temperatura, prueba prompts distintos, observa cómo se degrada con secuencias largas.

---

## Errores comunes que te vas a encontrar (y son normales)

- **Genera basura total:** casi siempre es un bug en attention (máscara mal, o Q·Kᵀ transpuesto al revés) o los pesos cargados en el orden equivocado (row-major vs column-major). Compara logits capa por capa contra Python.
- **NaN o infinitos:** softmax sin el truco de restar el máximo, o una división por cero en LayerNorm. Revisa estabilidad numérica.
- **Dimensiones que no cuadran:** lleva un comentario con la forma `[filas × cols]` esperada en cada paso. El 90% de los bugs se ven ahí.
- **Es lentísimo:** normal en Fase 0. La optimización es Fase 1. No la hagas ahora — primero que funcione.
- **Los pesos no cargan igual que en Python:** cuidado con el layout de memoria de los tensores. PyTorch guarda row-major; asegúrate de leer con la misma convención.

---

## Qué viene después (para que tengas el mapa completo)

- **Fase 1 — Optimización:** KV cache (no recalcular todo cada token), multithreading, quizá SIMD. Que deje de ser doloroso de lento.
- **Fase 2 — Cuantización y GGUF:** leer el formato binario real de los modelos modernos y decodificar pesos en Q4/Q8. Aquí ya cargas modelos "de verdad".
- **Fase 3 — Qwen2.5 + CUDA:** la arquitectura moderna (GQA, RoPE, RMSNorm) y tu RTX 4060 entrando al fin en juego. Este es el motor que reemplaza a Ollama en tu Jarvis.

Pero nada de eso importa hasta que GPT-2 genere su primera frase coherente. Un paso a la vez.

---

*Concepto el sábado, código el domingo. Verifica cada pieza contra la referencia antes de seguir. Cuando algo genere basura, el bug casi siempre está en attention o en cómo cargaste los pesos.*
