# ComfyUI-WordEmbeddings

Utilities for exploring and composing **word embeddings** in ComfyUI using `gensim` models. Use these nodes to inspect neighborhoods, traverse between concepts, and solve free‑form word “equations”.

**Included nodes (core):**
- **WordEmbeddingsLoader** — pick a gensim model by name.
- **WordEmbeddingsExplorer** — list nearest neighbors to a word.
- **WordEmbeddingsInterpolator** — sample words along the path between two words.
- **WordEmbeddingsEquation** — evaluate expressions like `king - man + woman`.

Default model: **`glove-wiki-gigaword-50`** (small & fast).

---

## Installation

> If you’re on **ComfyUI Windows Portable**, use its embedded Python.

1) Place this repo at:
```
ComfyUI/custom_nodes/ComfyUI-WordEmbeddings
```

2) Install dependencies:

**Windows (PowerShell)**
```powershell
cd .\ComfyUI_windows_portable\python_embeded
.\python.exe -m pip install --upgrade pip
.\python.exe -m pip install -r ..\ComfyUI\custom_nodes\ComfyUI-WordEmbeddingsequirements.txt
```

**macOS / Linux**
```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-WordEmbeddings/requirements.txt
```

3) Start ComfyUI. Nodes appear under **“ComfyUI-WordEmbeddings”**.

> **Model downloads:** On first use, `gensim` downloads the selected model to your user cache (e.g., `~/gensim-data` or `%USERPROFILE%\gensim-data`). To override, set `GENSIM_DATA_DIR` before launching ComfyUI.

---

## Quick start

1. Drop **WordEmbeddingsLoader** → choose `glove-wiki-gigaword-50` (fast).
2. Feed its `model_name` into:
   - **Explorer** (type a word, get top‑k neighbors), or
   - **Interpolator** (set `word_a`, `word_b`, choose `stops`), or
   - **Equation** (enter an expression like `paris - france + italy`).

Save any `STRING` output using your preferred text saver node.

---

## Nodes

### 1) WordEmbeddingsLoader
Select a gensim model via dropdown.

**Inputs**
- `model_name` *(choice)* — list of installed/available `gensim` models (default: `glove-wiki-gigaword-50` if present).

**Outputs**
- `model_name` *(STRING)* — pass to other nodes.

---

### 2) WordEmbeddingsExplorer
Return nearest neighbors for a single query word.

**Inputs**
- `model_name` *(STRING)* — from Loader.
- `word` *(STRING)* — e.g., `king`.
- `k` *(INT)* — neighbors to return (default 10).

**Outputs**
- `words_csv` *(STRING)* — comma‑separated neighbors.
- `debug_csv` *(STRING)* — `word:similarity` pairs.

**Example**
- `word = king`, `k = 10`  
  → `words_csv`: `prince,queen,monarch,...`  
  → `debug_csv`: `prince:0.78,queen:0.77,...`

---

### 3) WordEmbeddingsInterpolator
Sample along the path from `word_a` to `word_b` and report the nearest word at each interior step.

**Inputs**
- `model_name` *(STRING)* — from Loader.
- `word_a`, `word_b` *(STRING)* — endpoints (e.g., `king`, `queen`).
- `stops` *(INT)* — interior samples (no endpoints; default 12).
- *(optional)* `topn` *(INT)* — candidate pool size for nearest lookup (default 1000).
- *(optional)* `method` *(choice)* — `slerp` (default) or `lerp`.

**Outputs**
- `words_csv` *(STRING)* — samples only, comma‑separated.
- `debug_txt` *(STRING)* — per‑step diagnostics (`cos` to A/B/mid, etc.).

**Notes**
- `slerp` (spherical linear interpolation) generally respects direction on the embedding hypersphere better than `lerp`.

---

### 4) WordEmbeddingsEquation
Solve free‑form vector arithmetic from a text expression.

**Expression syntax**
- `+`, `-`, `plus`, `minus`
- Numeric coefficients: `2*woman`, `2.5 man`
- Ignores punctuation like `?`, `->`
- Anything after `=` or `equals` is ignored  
  (e.g., `king - man + woman = ?` works the same as without `= ?`)

**Inputs**
- `model_name` *(STRING)* — from Loader (default `glove-wiki-gigaword-50`).
- `expression` *(STRING)* — e.g., `king - man + woman`.
- `k` *(INT)* — number of results to return (default 10).
- *(optional)* `topn_pool` *(INT)* — pool size from `similar_by_vector` (default 2000).
- *(optional)* `normalize_terms` *(BOOL)* — L2‑normalize each term vector before summing (default `True`; recommended).
- *(optional)* `allow_inputs` *(BOOL)* — include input words in results (default `False`).

**Outputs**
- `words_csv` *(STRING)* — top‑k candidates.
- `debug_txt` *(STRING)* — parse details, used/missing terms, cosine scores.

**Examples**
- `king - man + woman` → often `queen`
- `paris - france + italy` → often `rome`
- `son - man + woman` → often `daughter`

---

## Tips & troubleshooting

- **Speed:** `glove-wiki-gigaword-50` is the quickest to download and load. Larger models (e.g., GoogleNews 300) can be richer but heavier.
- **Case:** Nodes attempt a lowercase fallback if the exact case is OOV.
- **First-run slowness:** The first time a model is selected, `gensim` will download it; subsequent runs use the cache.
- **Top‑N pools:** If an interpolated sample returns odd neighbors, try raising `topn` / `topn_pool`.

---

## License

Released under the **MIT License**. See the included `LICENSE` file.
