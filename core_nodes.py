# ComfyUI-WordEmbeddings
# Nodes: Loader (dropdown), Interpolator (STRING), Explorer (STRING), Equation (STRING)

import math
import re
import numpy as np

# ---------- gensim helpers ----------
def _lazy_api():
    import gensim.downloader as api
    return api

def _list_gensim_models():
    small_default = "glove-wiki-gigaword-50"
    try:
        api = _lazy_api()
        info = api.info()
        names = list(info.get("models", {}).keys())
        if not names:
            return [small_default]
        if small_default in names:
            names.remove(small_default)
            names.insert(0, small_default)
        return names
    except Exception:
        return [small_default]

_MODEL_CHOICES = _list_gensim_models()
_MODEL_CACHE = {}

def get_model(name: str):
    api = _lazy_api()
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = api.load(name)
    return _MODEL_CACHE[name]

def _get_vec(model, w: str):
    try:
        return model[w]
    except KeyError:
        lw = w.lower()
        if hasattr(model, "key_to_index") and lw in model.key_to_index:
            return model[lw]
        raise

# ---------- math helpers ----------
def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float((u @ v) / (nu * nv))

def _slerp(a, b, t):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot > 0.9995:
        v = (1 - t) * a + t * b
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    theta = math.acos(dot)
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)

def _nearest_to(v, model, topn=1000, seen=None):
    cand = model.similar_by_vector(v, topn=topn)
    best = None
    best_sim = -1.0
    for w, _ in cand:
        if seen and w in seen:
            continue
        x = model[w]
        sim = _cos(v, x)
        if sim > best_sim:
            best_sim = sim
            best = (w, x, best_sim)
    return best

# ---------- Nodes ----------
class WordEmbeddingsLoader:
    """Dropdown of available gensim models. Outputs only the selected model_name (STRING)."""
    @classmethod
    def INPUT_TYPES(cls):
        small_default = "glove-wiki-gigaword-50"
        default_choice = (
            small_default if small_default in _MODEL_CHOICES
            else (_MODEL_CHOICES[0] if _MODEL_CHOICES else small_default)
        )
        return {"required": {"model_name": (_MODEL_CHOICES, {"default": default_choice})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_name",)
    FUNCTION = "choose"
    CATEGORY = "ComfyUI-WordEmbeddings"

    def choose(self, model_name):
        return (model_name,)

class WordEmbeddingsInterpolator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"multiline": False, "default": "glove-wiki-gigaword-50"}),
                "word_a": ("STRING", {"multiline": False, "default": "king"}),
                "word_b": ("STRING", {"multiline": False, "default": "queen"}),
                "stops": ("INT", {"default": 12, "min": 1, "max": 256}),
            },
            "optional": {
                "topn": ("INT", {"default": 1000, "min": 50, "max": 5000}),
                "method": (["slerp", "lerp"], {"default": "slerp"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("words_csv", "debug_txt")
    FUNCTION = "interpolate"
    CATEGORY = "ComfyUI-WordEmbeddings"

    def interpolate(self, model_name, word_a, word_b, stops, topn=1000, method="slerp"):
        try:
            model = get_model(model_name)
        except Exception as e:
            return ("", f"[ComfyUI-WordEmbeddings] Error loading model '{model_name}': {e}")

        try:
            A = _get_vec(model, word_a)
        except KeyError:
            return ("", f"[ComfyUI-WordEmbeddings] Not in vocab: {word_a}")
        try:
            B = _get_vec(model, word_b)
        except KeyError:
            return ("", f"[ComfyUI-WordEmbeddings] Not in vocab: {word_b}")

        ts = np.linspace(1.0 / (stops + 1), 1.0 - 1.0 / (stops + 1), stops)
        seen = {word_a, word_b, word_a.lower(), word_b.lower()}
        words = []
        debug_lines = []

        mid = (A + B) / 2
        cos_AB = _cos(A, B)

        cos_vword_A = 1.000
        cosA_A = 1.000
        cosB_A = cos_AB
        centr_A = _cos(mid, A)
        debug_lines.append(
            f"00 t=0.000 word={word_a}  cos(v,word)={cos_vword_A:.3f}  cosA={cosA_A:.3f}  cosB={cosB_A:.3f}  centr={centr_A:.3f}  (endpoint)"
        )

        for i, t in enumerate(ts, 1):
            v = _slerp(A, B, float(t)) if method == "slerp" else ((1 - t) * A + t * B)
            pick = _nearest_to(v, model, topn=topn, seen=seen)
            if not pick:
                debug_lines.append(f"{i:02d} t={t:.3f}  (no candidate)")
                continue
            w, x, sim_v = pick
            seen.add(w)
            words.append(w)

            cA = _cos(A, x)
            cB = _cos(B, x)
            cent = _cos(mid, x)
            debug_lines.append(
                f"{i:02d} t={t:.3f} word={w}  cos(v,word)={sim_v:.3f}  cosA={cA:.3f}  cosB={cB:.3f}  centr={cent:.3f}"
            )

        idx_B = len(ts) + 1
        cos_vword_B = 1.000
        cosA_B = cos_AB
        cosB_B = 1.000
        centr_B = _cos(mid, B)
        debug_lines.append(
            f"{idx_B:02d} t=1.000 word={word_b}  cos(v,word)={cos_vword_B:.3f}  cosA={cosA_B:.3f}  cosB={cosB_B:.3f}  centr={centr_B:.3f}  (endpoint)"
        )

        words_csv = ",".join(words)
        debug_txt = "\n".join(debug_lines) if debug_lines else "(no samples)"
        return (words_csv, debug_txt)

class WordEmbeddingsExplorer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"multiline": False, "default": "glove-wiki-gigaword-50"}),
                "word": ("STRING", {"multiline": False, "default": "king"}),
                "k": ("INT", {"default": 10, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("words_csv", "debug_csv")
    FUNCTION = "explore"
    CATEGORY = "ComfyUI-WordEmbeddings"

    def explore(self, model_name, word, k):
        try:
            model = get_model(model_name)
        except Exception as e:
            return ("", f"[ComfyUI-WordEmbeddings] Error loading model '{model_name}': {e}")
        try:
            _ = _get_vec(model, word)
        except KeyError:
            return ("", f"[ComfyUI-WordEmbeddings] Not in vocab: {word}")
        sims = model.most_similar(word, topn=k)
        words = [w for w, _ in sims]
        debug = [f"{w}:{round(score,3)}" for w, score in sims]
        return (",".join(words), ",".join(debug))

class WordEmbeddingsEquation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "glove-wiki-gigaword-50"}),
                "expression": ("STRING", {"default": "king - man + woman", "multiline": False}),
                "k": ("INT", {"default": 10, "min": 1, "max": 100}),
            },
            "optional": {
                "topn_pool": ("INT", {"default": 2000, "min": 100, "max": 20000}),
                "normalize_terms": ("BOOLEAN", {"default": True}),
                "allow_inputs": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("words_csv", "debug_txt")
    FUNCTION = "run"
    CATEGORY = "ComfyUI-WordEmbeddings"

    def _norm(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _parse_expression(self, expr: str):
        if not expr:
            return [], []
        cut = re.split(r"(?:=|equals)", expr, flags=re.IGNORECASE)
        expr = cut[0]
        s = expr.replace(",", " ").replace("\t", " ").strip()
        tokens = [t for t in re.split(r"\s+", s) if t]
        sign = +1.0
        pairs = []
        skipped = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in {"+", "plus"}:
                sign = +1.0
                i += 1
                continue
            if tok in {"-", "minus"}:
                sign = -1.0
                i += 1
                continue
            m = re.match(r"^([+-]?\d+(?:\.\d+)?)\*(.+)$", tok)
            if m:
                coef = float(m.group(1)); word = m.group(2)
                pairs.append((word, sign * coef)); sign = +1.0; i += 1; continue
            mnum = re.match(r"^[+-]?\d+(?:\.\d+)?$", tok)
            if mnum and i + 1 < len(tokens):
                try:
                    coef = float(tok); word = tokens[i + 1]
                    pairs.append((word, sign * coef)); sign = +1.0; i += 2; continue
                except Exception:
                    pass
            if tok in {"?", "->", "=>"}:
                i += 1; continue
            pairs.append((tok, sign * 1.0)); sign = +1.0; i += 1
        return pairs, skipped

    def run(self, model_name, expression, k, topn_pool=2000, normalize_terms=True, allow_inputs=False):
        try:
            model = get_model(model_name)
        except Exception as e:
            return ("", f"[ComfyUI-WordEmbeddings] Error loading model '{model_name}': {e}")
        pairs, skipped = self._parse_expression(expression)
        if not pairs:
            return ("", f"[ComfyUI-WordEmbeddings] Empty or unparsable expression: '{expression}'")
        used = []
        missing = []
        vec = None
        for word, w in pairs:
            try:
                v = _get_vec(model, word)
                if normalize_terms:
                    v = self._norm(v)
                v = w * v
                vec = v if vec is None else (vec + v)
                used.append((word, w))
            except KeyError:
                missing.append(word)
        if vec is None:
            miss = ", ".join(missing) if missing else "(none)"
            return ("", f"[ComfyUI-WordEmbeddings] No valid words found in vocab. Missing: {miss}")
        target = self._norm(vec)
        try:
            cand = model.similar_by_vector(target, topn=int(topn_pool))
        except Exception as e:
            return ("", f"[ComfyUI-WordEmbeddings] Search error: {e}")
        input_words = {w for (w, _) in used}
        out = []
        for cw, approx_cos in cand:
            if not allow_inputs and cw in input_words:
                continue
            try:
                x = _get_vec(model, cw)
            except KeyError:
                continue
            cos_t = float(_cos(target, x))
            out.append((cw, float(approx_cos), cos_t))
            if len(out) >= int(k):
                break
        if not out:
            dbg = [
                "[ComfyUI-WordEmbeddings] No results after filtering.",
                f"expr={expression}",
                f"used_terms={used}",
                f"missing_terms={missing}",
                f"allow_inputs={allow_inputs}  topn_pool={topn_pool}  normalize_terms={normalize_terms}",
            ]
            return ("", "\n".join(dbg))
        words_csv = ",".join([w for (w, _, __) in out])
        debug_lines = [
            f"expr={expression}",
            f"used_terms={used}",
            f"missing_terms={missing}",
            f"normalize_terms={normalize_terms}  allow_inputs={allow_inputs}  topn_pool={topn_pool}",
        ]
        for i, (w2, ac, ct) in enumerate(out, 1):
            debug_lines.append(f"{i:02d} {w2}  approx_cos(target,cand)={ac:.3f}  cos(target,cand)={ct:.3f}")
        return (words_csv, "\n".join(debug_lines))

# ---------- Registration ----------
NODE_CLASS_MAPPINGS = {
    "WordEmbeddingsLoader": WordEmbeddingsLoader,
    "WordEmbeddingsInterpolator": WordEmbeddingsInterpolator,
    "WordEmbeddingsExplorer": WordEmbeddingsExplorer,
    "WordEmbeddingsEquation": WordEmbeddingsEquation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WordEmbeddingsLoader": "ComfyUI-WordEmbeddings: Loader",
    "WordEmbeddingsInterpolator": "ComfyUI-WordEmbeddings: Interpolator",
    "WordEmbeddingsExplorer": "ComfyUI-WordEmbeddings: Explorer",
    "WordEmbeddingsEquation": "ComfyUI-WordEmbeddings: Equation",
}
