# ComfyUI-WordEmbeddings
# Core nodes with a typed WE_MODEL handle:
#   - WordEmbeddingsLoader            (pretrained dropdown -> WE_MODEL)
#   - WordEmbeddingsLocalModelLoader  (local file -> WE_MODEL)
#   - WordEmbeddingsExplorer          (WE_MODEL + word -> neighbors)
#   - WordEmbeddingsInterpolator      (WE_MODEL + word_a + word_b -> samples)
#   - WordEmbeddingsEquation          (WE_MODEL + expression -> analogy)
#   - WordEmbeddingsTokenNeighbors    (WE_MODEL + word -> neighbor words + cosines)
#   - WordEmbeddingsTokenAxis         (WE_MODEL + word (+pos,neg) -> projection & cosines + summary)
#   - WordEmbeddingsTokenAxis2D       (WE_MODEL + token + 2 axes -> x,y + summary)
#   - WordEmbeddingsTokenAxis3D       (WE_MODEL + token + 3 axes -> x,y,z + summary)
#   - WordEmbeddingsTokenCentrality   (WE_MODEL + word -> cosine to global mean & norm)

import os
import re
import math
import numpy as np
from dataclasses import dataclass

# ---------- gensim helpers ----------
def _lazy_api():
    import gensim.downloader as api
    return api

def _list_gensim_models():
    SMALL_DEFAULT = "glove-wiki-gigaword-50"
    try:
        api = _lazy_api()
        info = api.info()
        names = list(info.get("models", {}).keys())
        if not names:
            return [SMALL_DEFAULT]
        if SMALL_DEFAULT in names:
            names.remove(SMALL_DEFAULT)
            names.insert(0, SMALL_DEFAULT)
        return names
    except Exception:
        return [SMALL_DEFAULT]

try:
    from gensim.models import KeyedVectors
except Exception:
    KeyedVectors = None

_MODEL_CHOICES = _list_gensim_models()

# ---------- WE_MODEL handle ----------
@dataclass
class WEModel:
    name: str     # model label (e.g., "glove-wiki-gigaword-50" or "local:foo.bin")
    kv: object    # gensim KeyedVectors or equivalent

def _load_pretrained(name: str):
    api = _lazy_api()
    return api.load(name)

def _load_local_keyedvectors(path: str, fmt: str = "auto"):
    if KeyedVectors is None:
        raise RuntimeError("gensim KeyedVectors not available (pip install gensim).")
    p = path.strip().strip('"').strip("'")
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")

    if fmt == "kv":
        return KeyedVectors.load(p, mmap="r")
    if fmt == "word2vec_bin":
        return KeyedVectors.load_word2vec_format(p, binary=True)
    if fmt == "word2vec_text":
        return KeyedVectors.load_word2vec_format(p, binary=False)

    # auto detect
    ext = os.path.splitext(p)[1].lower()
    if ext in [".kv", ".kv2"]:
        return KeyedVectors.load(p, mmap="r")
    if ext in [".bin", ".bin.gz"]:
        return KeyedVectors.load_word2vec_format(p, binary=True)
    if ext in [".txt", ".vec", ".gz"]:
        return KeyedVectors.load_word2vec_format(p, binary=False)

    # last attempts
    try:
        return KeyedVectors.load(p, mmap="r")
    except Exception:
        return KeyedVectors.load_word2vec_format(p, binary=False)

# ---------- vector helpers ----------
def _kv_has_word(kv, w: str) -> bool:
    if hasattr(kv, "key_to_index"):
        return w in kv.key_to_index
    try:
        _ = kv[w]
        return True
    except Exception:
        return False

def _kv_get_vec_case_insensitive(kv, w: str):
    try:
        return kv[w]
    except KeyError:
        lw = w.lower()
        if _kv_has_word(kv, lw):
            return kv[lw]
        raise

def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float((u @ v) / (nu * nv))

def _slerp(a, b, t):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot > 0.9995:
        v = (1 - t) * a + t * b
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    theta = math.acos(dot)
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)

def _nearest_to(v, kv, topn=1000, seen=None):
    cand = kv.similar_by_vector(v, topn=topn)
    best = None
    best_sim = -1.0
    for w, _ in cand:
        if seen and w in seen:
            continue
        x = kv[w]
        sim = _cos(v, x)
        if sim > best_sim:
            best_sim = sim
            best = (w, x, best_sim)
    return best

# ---------- caching for heavy global stats ----------
def _get_or_make_mean_vec(kv):
    mv = getattr(kv, "_we_mean_vec", None)
    if mv is not None:
        return mv
    try:
        M = kv.get_normed_vectors()  # (n_words, dim), rows L2-normalized
    except Exception:
        V = getattr(kv, "vectors", None)
        if V is None:
            raise RuntimeError("KeyedVectors has no 'vectors' matrix.")
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        M = V / norms
    mean_vec = M.mean(axis=0)
    n = np.linalg.norm(mean_vec)
    mean_vec = mean_vec / n if n > 0 else mean_vec
    setattr(kv, "_we_mean_vec", mean_vec)
    return mean_vec

def _get_or_make_max_count(kv):
    maxc = getattr(kv, "_we_max_count", None)
    if maxc is not None:
        return maxc
    max_count = 0
    if hasattr(kv, "key_to_index"):
        for w in kv.key_to_index.keys():
            try:
                c = kv.get_vecattr(w, "count")
                if c is not None and c > max_count:
                    max_count = int(c)
            except Exception:
                continue
    setattr(kv, "_we_max_count", int(max_count))
    return int(max_count)

# ---------- Nodes ----------
class WordEmbeddingsLoader:
    """Pretrained (gensim downloader) loader that outputs a WE_MODEL handle."""
    @classmethod
    def INPUT_TYPES(cls):
        small_default = "glove-wiki-gigaword-50"
        default_choice = (
            small_default if small_default in _MODEL_CHOICES
            else (_MODEL_CHOICES[0] if _MODEL_CHOICES else small_default)
        )
        return {"required": {"model_name": (_MODEL_CHOICES, {"default": default_choice})}}

    # Single output => easier quick-create suggestions
    RETURN_TYPES = ("WE_MODEL",)
    RETURN_NAMES = ("we_model",)
    FUNCTION = "load"
    CATEGORY = "WordEmbeddings"

    def load(self, model_name):
        kv = _load_pretrained(model_name)
        return (WEModel(model_name, kv),)

class WordEmbeddingsLocalModelLoader:
    """Local file loader that outputs a WE_MODEL handle."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"path": ("STRING", {"default": "", "multiline": False})},
            "optional": {
                "format": (["auto", "kv", "word2vec_bin", "word2vec_text"], {"default": "auto"}),
                "label": ("STRING", {"default": "local", "multiline": False}),
            }
        }

    RETURN_TYPES = ("WE_MODEL",)
    RETURN_NAMES = ("we_model",)
    FUNCTION = "load"
    CATEGORY = "WordEmbeddings"

    def load(self, path, format="auto", label="local"):
        kv = _load_local_keyedvectors(path, fmt=format)
        name = f"{label}:{os.path.basename(path)}"
        return (WEModel(name, kv),)

class WordEmbeddingsExplorer:
    """Neighbors of a single word from WE_MODEL."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"we_model": ("WE_MODEL",)},
            "optional": {
                "word": ("STRING", {"multiline": False, "default": "king"}),
                "k": ("INT", {"default": 10, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("words_csv", "debug_csv")
    FUNCTION = "explore"
    CATEGORY = "WordEmbeddings"

    def explore(self, we_model, word="king", k=10):
        kv = we_model.kv
        try:
            _ = _kv_get_vec_case_insensitive(kv, word)
        except KeyError:
            return ("", f"Not in vocab: {word}")
        sims = kv.most_similar(word, topn=int(k))
        words = [w for w, _ in sims]
        debug = [f"{w}:{round(score,3)}" for w, score in sims]
        return (",".join(words), ",".join(debug))

class WordEmbeddingsInterpolator:
    """Interpolate between two words on the unit sphere and pick nearest words."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"we_model": ("WE_MODEL",)},
            "optional": {
                "word_a": ("STRING", {"multiline": False, "default": "king"}),
                "word_b": ("STRING", {"multiline": False, "default": "queen"}),
                "stops": ("INT", {"default": 12, "min": 1, "max": 256}),
                "topn": ("INT", {"default": 1000, "min": 50, "max": 5000}),
                "method": (["slerp", "lerp"], {"default": "slerp"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("words_csv", "debug_txt")
    FUNCTION = "interpolate"
    CATEGORY = "WordEmbeddings"

    def interpolate(self, we_model, word_a="king", word_b="queen", stops=12, topn=1000, method="slerp"):
        kv = we_model.kv
        try:
            A = _kv_get_vec_case_insensitive(kv, word_a)
        except KeyError:
            return ("", f"Not in vocab: {word_a}")
        try:
            B = _kv_get_vec_case_insensitive(kv, word_b)
        except KeyError:
            return ("", f"Not in vocab: {word_b}")

        ts = np.linspace(1.0 / (stops + 1), 1.0 - 1.0 / (stops + 1), int(stops))
        seen = {word_a, word_b, word_a.lower(), word_b.lower()}
        words = []
        debug_lines = []

        mid = (A + B) / 2
        cos_AB = _cos(A, B)

        # Endpoint A
        debug_lines.append(
            f"00 t=0.000 word={word_a}  cos(v,word)=1.000  cosA=1.000  cosB={cos_AB:.3f}  centr={_cos(mid,A):.3f}  (endpoint)"
        )

        # Interior samples
        for i, t in enumerate(ts, 1):
            v = _slerp(A, B, float(t)) if method == "slerp" else ((1 - t) * A + t * B)
            pick = _nearest_to(v, kv, topn=int(topn), seen=seen)
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

        # Endpoint B
        idx_B = len(ts) + 1
        debug_lines.append(
            f"{idx_B:02d} t=1.000 word={word_b}  cos(v,word)=1.000  cosA={cos_AB:.3f}  cosB=1.000  centr={_cos(mid,B):.3f}  (endpoint)"
        )

        return (",".join(words), "\n".join(debug_lines) if debug_lines else "(no samples)")

class WordEmbeddingsEquation:
    """Vector arithmetic from a free-text expression (e.g., 'king - man + woman')."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"we_model": ("WE_MODEL",)},
            "optional": {
                "expression": ("STRING", {"default": "king - man + woman", "multiline": False}),
                "k": ("INT", {"default": 10, "min": 1, "max": 100}),
                "topn_pool": ("INT", {"default": 2000, "min": 100, "max": 20000}),
                "normalize_terms": ("BOOLEAN", {"default": True}),
                "allow_inputs": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("words_csv", "debug_txt")
    FUNCTION = "run"
    CATEGORY = "WordEmbeddings"

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
                sign = +1.0; i += 1; continue
            if tok in {"-", "minus"}:
                sign = -1.0; i += 1; continue
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

    def run(self, we_model, expression="king - man + woman", k=10, topn_pool=2000, normalize_terms=True, allow_inputs=False):
        kv = we_model.kv

        pairs, skipped = self._parse_expression(expression)
        if not pairs:
            return ("", f"Empty or unparsable expression: '{expression}'")

        used = []
        missing = []
        vec = None

        for word, w in pairs:
            try:
                v = _kv_get_vec_case_insensitive(kv, word)
                if normalize_terms:
                    v = self._norm(v)
                v = w * v
                vec = v if vec is None else (vec + v)
                used.append((word, w))
            except KeyError:
                missing.append(word)

        if vec is None:
            miss = ", ".join(missing) if missing else "(none)"
            return ("", f"No valid words found in vocab. Missing: {miss}")

        target = self._norm(vec)

        try:
            cand = kv.similar_by_vector(target, topn=int(topn_pool))
        except Exception as e:
            return ("", f"Search error: {e}")

        input_words = {w for (w, _) in used}
        out = []
        for cw, approx_cos in cand:
            if not allow_inputs and cw in input_words:
                continue
            try:
                x = kv[cw]
            except KeyError:
                continue
            cos_t = float(_cos(target, x))
            out.append((cw, float(approx_cos), cos_t))
            if len(out) >= int(k):
                break

        if not out:
            dbg = [
                "No results after filtering.",
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

# -------------------- NEW TOKEN NODES --------------------

class WordEmbeddingsTokenNeighbors:
    """Top-k neighbors for a single token. Outputs words CSV and cosine CSV."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"we_model": ("WE_MODEL",)},
            "optional": {
                "word": ("STRING", {"default": "king", "multiline": False}),
                "k": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("words_csv", "cosines_csv")
    FUNCTION = "run"
    CATEGORY = "WordEmbeddings"

    def run(self, we_model, word="king", k=10):
        kv = we_model.kv
        try:
            _ = _kv_get_vec_case_insensitive(kv, word)
        except KeyError:
            return ("", "")
        sims = kv.most_similar(word, topn=int(k))
        words = [w for w, _ in sims]
        cosines = [f"{float(c):.6f}" for _, c in sims]
        return (",".join(words), ",".join(cosines))

# -------- Token Axis (1D) with human summary --------
class WordEmbeddingsTokenAxis:
    """
    Project a token onto one semantic axis "left|right".
    Outputs:
      - x          : FLOAT in [-1, 1]  (positive => toward LEFT pole)
      - summary    : STRING (human description combining x with cosine-based confidence)
      - cos_left   : FLOAT = cos(token, left_mean)
      - cos_right  : FLOAT = cos(token, right_mean)

    Axis format examples:
      "man,boy,he,him|woman,girl,she,her"
      "royal,king,kingly|common,peasant"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"we_model": ("WE_MODEL",)},
            "optional": {
                "token": ("STRING", {"default": "king", "multiline": False}),
                "axis": ("STRING", {
                    "default": "man,boy,he,him,his,king,father,brother,husband,actor"
                               "|woman,girl,she,her,hers,queen,mother,sister,wife,actress",
                    "multiline": True
                }),
                "token_can_be_phrase": ("BOOLEAN", {"default": True}),
                "lowercase": ("BOOLEAN", {"default": True}),
                # how wide a band counts as “similar/equal” on the axis
                "neutral_eps": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                # how tiny x must be to say “equal” (otherwise we say “similar”)
                "equal_eps": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("x", "summary", "cos_left", "cos_right")
    FUNCTION = "run"
    CATEGORY = "WordEmbeddings"

    # ------------ helpers ------------
    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @staticmethod
    def _split_axis(axis_str: str):
        s = (axis_str or "").strip()
        if "|" in s:
            left_raw, right_raw = s.split("|", 1)
        else:
            left_raw, right_raw = s, ""
        L = [w.strip() for w in left_raw.split(",") if w.strip()]
        R = [w.strip() for w in right_raw.split(",") if w.strip()]
        return L, R, left_raw.strip(), right_raw.strip()

    @staticmethod
    def _first_term(label_side: str):
        if not label_side:
            return "?"
        return (label_side.split(",")[0].strip() or "?")

    @classmethod
    def _avg_unit(cls, kv, terms, lowercase=True):
        vecs = []
        for t in terms:
            w = t.lower() if lowercase else t
            try:
                v = _kv_get_vec_case_insensitive(kv, w)
                vecs.append(cls._unit(v))
            except KeyError:
                pass
        if not vecs:
            return None
        return cls._unit(np.mean(vecs, axis=0))

    @classmethod
    def _phrase_vec(cls, kv, token, token_can_be_phrase=True, lowercase=True):
        pieces = [token] if not token_can_be_phrase else [w for w in token.replace(",", " ").split() if w]
        vecs = []
        for w in pieces:
            ww = w.lower() if lowercase else w
            try:
                vecs.append(cls._unit(_kv_get_vec_case_insensitive(kv, ww)))
            except KeyError:
                pass
        if not vecs:
            return None
        return cls._unit(np.mean(vecs, axis=0))

    @staticmethod
    def _confidence_phrase(cos_left, cos_right):
        # magnitudes and gap
        C_L, C_R = abs(cos_left), abs(cos_right)
        C_max, C_min = max(C_L, C_R), min(C_L, C_R)
        gap = abs(C_L - C_R)

        # thresholds (tune if you like)
        BOTH_MIN = 0.35
        BOTH_STRONG = 0.55
        GAP_CLEAR = 0.18
        MAX_MIN = 0.45
        MAX_STRONG = 0.65
        LOW_MAX = 0.25

        if C_max < LOW_MAX:
            return "with low confidence"

        # near both poles (high & similar)
        if (C_min >= BOTH_MIN) and (gap < 0.10):
            level = "strong" if C_min >= BOTH_STRONG else "moderate"
            return f"with {level} confidence"

        # one pole clearly dominates
        if (gap >= GAP_CLEAR) and (C_max >= MAX_MIN):
            side = "left" if (C_L >= C_R) else "right"
            level = "strong" if (C_max >= MAX_STRONG and gap >= 0.30) else "moderate"
            return f"with {level} confidence in {side}"

        # otherwise, middling
        return "with moderate confidence"

    # ------------ main ------------
    def run(self, we_model, token="king",
            axis="man,boy,he,him,his,king,father,brother,husband,actor|woman,girl,she,her,hers,queen,mother,sister,wife,actress",
            token_can_be_phrase=True, lowercase=True, neutral_eps=0.08, equal_eps=0.02):

        kv = we_model.kv

        # token vector
        v = self._phrase_vec(kv, token, token_can_be_phrase=token_can_be_phrase, lowercase=lowercase)
        if v is None:
            return (0.0, f"{token} not found in the model vocabulary.", 0.0, 0.0)

        # poles
        L_terms, R_terms, L_raw, R_raw = self._split_axis(axis)
        p = self._avg_unit(kv, L_terms, lowercase=lowercase) if L_terms else None
        n = self._avg_unit(kv, R_terms, lowercase=lowercase) if R_terms else None

        # cosines to each pole mean
        cos_left = float(v @ p) if p is not None else 0.0
        cos_right = float(v @ n) if n is not None else 0.0

        # axis direction (fallback to single-sided axis if one pole missing)
        if p is not None and n is not None:
            axis_dir = self._unit(p - n)
        elif p is not None:
            axis_dir = p
        elif n is not None:
            axis_dir = -n
        else:
            # no usable axis
            L = self._first_term(L_raw) if L_raw else "left"
            R = self._first_term(R_raw) if R_raw else "right"
            summary = f"{token} on axis '{L}|{R}': axis unavailable (no pole terms found)."
            return (0.0, summary, cos_left, cos_right)

        # projection scalar
        x = float(v @ axis_dir)

        # human summary: keep left/right order
        L = self._first_term(L_raw) if L_raw else "left"
        R = self._first_term(R_raw) if R_raw else "right"

        # descriptor from x
        x_abs = abs(x)
        if x_abs <= equal_eps:
            dir_phrase = f"equal {L} and {R}"
        elif x_abs < neutral_eps:
            dir_phrase = f"similar {L} and {R}"
        else:
            dir_phrase = (f"more {L} than {R}") if x >= 0 else (f"less {L} than {R}")

        # confidence from cosines
        conf_phrase = self._confidence_phrase(cos_left, cos_right)

        summary = f"{token} is {dir_phrase}, {conf_phrase}."

        return (x, summary, cos_left, cos_right)


class WordEmbeddingsTokenAxis2D:
    """
    Project a token on TWO semantic axes and emit:
      x (FLOAT), y (FLOAT), summary (STRING),
      x_cos_left, x_cos_right, y_cos_left, y_cos_right (FLOATs).

    Summary includes only axes considered "strong" (see thresholds).
    Example:
      king is more male than female, with strong confidence; and more royal than common, with moderate confidence in left
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"we_model": ("WE_MODEL",)},
            "optional": {
                "token": ("STRING", {"default": "king", "multiline": False}),
                "axis1": ("STRING", {"default": "man,boy,he,him,his,father,brother,husband|woman,girl,she,her,hers,mother,sister,wife", "multiline": True}),
                "axis2": ("STRING", {"default": "royal,king,kingly,noble|common,commoner,peasant,ordinary", "multiline": True}),
                "token_can_be_phrase": ("BOOLEAN", {"default": True}),
                "lowercase": ("BOOLEAN", {"default": True}),
                # direction wording
                "neutral_eps": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "equal_eps":   ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005}),
                # strength gating for summary (axes must pass all to be included)
                "strong_proj": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strong_cos":  ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strong_gap":  ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES  = ("x", "y", "summary", "x_cos_left", "x_cos_right", "y_cos_left", "y_cos_right")
    FUNCTION = "run"
    CATEGORY = "WordEmbeddings"

    # ---------- helpers ----------
    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @staticmethod
    def _split_axis(axis_str: str):
        s = (axis_str or "").strip()
        if "|" in s:
            left_raw, right_raw = s.split("|", 1)
        else:
            left_raw, right_raw = s, ""
        L = [w.strip() for w in left_raw.split(",") if w.strip()]
        R = [w.strip() for w in right_raw.split(",") if w.strip()]
        return L, R, left_raw.strip(), right_raw.strip()

    @staticmethod
    def _first_term(label_side: str):
        if not label_side:
            return "?"
        return (label_side.split(",")[0].strip() or "?")

    @classmethod
    def _avg_unit(cls, kv, terms, lowercase=True):
        vecs = []
        for t in terms:
            w = t.lower() if lowercase else t
            try:
                v = _kv_get_vec_case_insensitive(kv, w)
                vecs.append(cls._unit(v))
            except KeyError:
                pass
        if not vecs:
            return None
        return cls._unit(np.mean(vecs, axis=0))

    @classmethod
    def _phrase_vec(cls, kv, token, token_can_be_phrase=True, lowercase=True):
        parts = [token] if not token_can_be_phrase else [w for w in token.replace(",", " ").split() if w]
        vecs = []
        for w in parts:
            ww = w.lower() if lowercase else w
            try:
                vecs.append(cls._unit(_kv_get_vec_case_insensitive(kv, ww)))
            except KeyError:
                pass
        if not vecs:
            return None
        return cls._unit(np.mean(vecs, axis=0))

    @staticmethod
    def _direction_phrase(x, L, R, neutral_eps, equal_eps):
        ax = abs(x)
        if ax <= equal_eps:
            return f"equal {L} and {R}"
        if ax < neutral_eps:
            return f"similar {L} and {R}"
        return (f"more {L} than {R}") if x >= 0 else (f"less {L} than {R}")

    @staticmethod
    def _confidence_level_and_phrase(cos_left, cos_right):
        C_L, C_R = abs(cos_left), abs(cos_right)
        C_max, C_min = max(C_L, C_R), min(C_L, C_R)
        gap = abs(C_L - C_R)

        BOTH_MIN = 0.35
        BOTH_STRONG = 0.55
        GAP_CLEAR = 0.18
        MAX_MIN = 0.45
        MAX_STRONG = 0.65
        LOW_MAX = 0.25

        if C_max < LOW_MAX:
            return "low", "with low confidence"

        if (C_min >= BOTH_MIN) and (gap < 0.10):
            level = "strong" if C_min >= BOTH_STRONG else "moderate"
            return level, f"with {level} confidence"

        if (gap >= GAP_CLEAR) and (C_max >= MAX_MIN):
            side = "left" if (C_L >= C_R) else "right"
            level = "strong" if (C_max >= MAX_STRONG and gap >= 0.30) else "moderate"
            return level, f"with {level} confidence in {side}"

        return "moderate", "with moderate confidence"

    @classmethod
    def _proj_and_cos(cls, kv, v, left_terms, right_terms, lowercase=True):
        Lm = cls._avg_unit(kv, left_terms, lowercase=lowercase) if left_terms else None
        Rm = cls._avg_unit(kv, right_terms, lowercase=lowercase) if right_terms else None

        cos_left  = float(v @ Lm) if (v is not None and Lm is not None) else 0.0
        cos_right = float(v @ Rm) if (v is not None and Rm is not None) else 0.0

        if Lm is not None and Rm is not None:
            axis_dir = cls._unit(Lm - Rm)
        elif Lm is not None:
            axis_dir = Lm
        elif Rm is not None:
            axis_dir = -Rm
        else:
            return 0.0, cos_left, cos_right, False

        proj = float(v @ axis_dir) if v is not None else 0.0
        return proj, cos_left, cos_right, True

    # ---------- main ----------
    def run(self, we_model, token="king",
            axis1="man,boy,he,him,his,father,brother,husband|woman,girl,she,her,hers,mother,sister,wife",
            axis2="royal,king,kingly,noble|common,commoner,peasant,ordinary",
            token_can_be_phrase=True, lowercase=True,
            neutral_eps=0.08, equal_eps=0.02,
            strong_proj=0.35, strong_cos=0.65, strong_gap=0.30):

        kv = we_model.kv
        v = self._phrase_vec(kv, token, token_can_be_phrase=token_can_be_phrase, lowercase=lowercase)
        if v is None:
            return (0.0, 0.0, f"{token} not found in the model vocabulary.", 0.0, 0.0, 0.0, 0.0)

        # parse axes
        a1L, a1R, a1L_raw, a1R_raw = self._split_axis(axis1)
        a2L, a2R, a2L_raw, a2R_raw = self._split_axis(axis2)

        # compute projections & cosines
        x, x_cos_left, x_cos_right, ok1 = self._proj_and_cos(kv, v, a1L, a1R, lowercase=lowercase)
        y, y_cos_left, y_cos_right, ok2 = self._proj_and_cos(kv, v, a2L, a2R, lowercase=lowercase)

        # build per-axis clauses (but only include "strong" ones)
        clauses = []

        if ok1:
            dir1 = self._direction_phrase(x, self._first_term(a1L_raw), self._first_term(a1R_raw), neutral_eps, equal_eps)
            lvl1, conf1 = self._confidence_level_and_phrase(x_cos_left, x_cos_right)
            strong1 = (abs(x) >= strong_proj) and (
                max(abs(x_cos_left), abs(x_cos_right)) >= strong_cos or
                (max(abs(x_cos_left), abs(x_cos_right)) >= 0.55 and abs(abs(x_cos_left) - abs(x_cos_right)) >= strong_gap)
            )
            if strong1:
                clauses.append(f"{dir1}, {conf1}")

        if ok2:
            dir2 = self._direction_phrase(y, self._first_term(a2L_raw), self._first_term(a2R_raw), neutral_eps, equal_eps)
            lvl2, conf2 = self._confidence_level_and_phrase(y_cos_left, y_cos_right)
            strong2 = (abs(y) >= strong_proj) and (
                max(abs(y_cos_left), abs(y_cos_right)) >= strong_cos or
                (max(abs(y_cos_left), abs(y_cos_right)) >= 0.55 and abs(abs(y_cos_left) - abs(y_cos_right)) >= strong_gap)
            )
            if strong2:
                clauses.append(f"{dir2}, {conf2}")

        if not clauses:
            summary = f"{token} shows no strong alignment on the provided axes."
        elif len(clauses) == 1:
            summary = f"{token} is {clauses[0]}"
        else:
            summary = f"{token} is {clauses[0]} and {clauses[1]}"

        return (x, y, summary, x_cos_left, x_cos_right, y_cos_left, y_cos_right)


class WordEmbeddingsTokenAxis3D:
    """
    Project a token on THREE semantic axes and emit:
      x, y, z (FLOATs), summary (STRING),
      x_cos_left/x_cos_right, y_cos_left/y_cos_right, z_cos_left/z_cos_right (FLOATs).

    Summary includes only axes considered "strong" (see thresholds).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"we_model": ("WE_MODEL",)},
            "optional": {
                "token": ("STRING", {"default": "king", "multiline": False}),
                "axis1": ("STRING", {"default": "man,boy,he,him,his,father,brother,husband|woman,girl,she,her,hers,mother,sister,wife", "multiline": True}),
                "axis2": ("STRING", {"default": "hot,warm,heat|cold,cool,freezing", "multiline": True}),
                "axis3": ("STRING", {"default": "royal,king,kingly,noble|common,commoner,peasant,ordinary", "multiline": True}),
                "token_can_be_phrase": ("BOOLEAN", {"default": True}),
                "lowercase": ("BOOLEAN", {"default": True}),
                # direction wording
                "neutral_eps": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "equal_eps":   ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005}),
                # strength gating for summary
                "strong_proj": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strong_cos":  ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strong_gap":  ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (
        "FLOAT","FLOAT","FLOAT","STRING",
        "FLOAT","FLOAT","FLOAT","FLOAT","FLOAT","FLOAT"
    )
    RETURN_NAMES = (
        "x","y","z","summary",
        "x_cos_left","x_cos_right","y_cos_left","y_cos_right","z_cos_left","z_cos_right"
    )
    FUNCTION = "run"
    CATEGORY = "WordEmbeddings"

    # ---- reuse helpers from 2D ----
    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @staticmethod
    def _split_axis(axis_str: str):
        s = (axis_str or "").strip()
        if "|" in s:
            left_raw, right_raw = s.split("|", 1)
        else:
            left_raw, right_raw = s, ""
        L = [w.strip() for w in left_raw.split(",") if w.strip()]
        R = [w.strip() for w in right_raw.split(",") if w.strip()]
        return L, R, left_raw.strip(), right_raw.strip()

    @staticmethod
    def _first_term(label_side: str):
        if not label_side:
            return "?"
        return (label_side.split(",")[0].strip() or "?")

    @classmethod
    def _avg_unit(cls, kv, terms, lowercase=True):
        vecs = []
        for t in terms:
            w = t.lower() if lowercase else t
            try:
                v = _kv_get_vec_case_insensitive(kv, w)
                vecs.append(cls._unit(v))
            except KeyError:
                pass
        if not vecs:
            return None
        return cls._unit(np.mean(vecs, axis=0))

    @classmethod
    def _phrase_vec(cls, kv, token, token_can_be_phrase=True, lowercase=True):
        parts = [token] if not token_can_be_phrase else [w for w in token.replace(",", " ").split() if w]
        vecs = []
        for w in parts:
            ww = w.lower() if lowercase else w
            try:
                vecs.append(cls._unit(_kv_get_vec_case_insensitive(kv, ww)))
            except KeyError:
                pass
        if not vecs:
            return None
        return cls._unit(np.mean(vecs, axis=0))

    @staticmethod
    def _direction_phrase(x, L, R, neutral_eps, equal_eps):
        ax = abs(x)
        if ax <= equal_eps:
            return f"equal {L} and {R}"
        if ax < neutral_eps:
            return f"similar {L} and {R}"
        return (f"more {L} than {R}") if x >= 0 else (f"less {L} than {R}")

    @staticmethod
    def _confidence_level_and_phrase(cos_left, cos_right):
        C_L, C_R = abs(cos_left), abs(cos_right)
        C_max, C_min = max(C_L, C_R), min(C_L, C_R)
        gap = abs(C_L - C_R)

        BOTH_MIN = 0.35
        BOTH_STRONG = 0.55
        GAP_CLEAR = 0.18
        MAX_MIN = 0.45
        MAX_STRONG = 0.65
        LOW_MAX = 0.25

        if C_max < LOW_MAX:
            return "low", "with low confidence"

        if (C_min >= BOTH_MIN) and (gap < 0.10):
            level = "strong" if C_min >= BOTH_STRONG else "moderate"
            return level, f"with {level} confidence"

        if (gap >= GAP_CLEAR) and (C_max >= MAX_MIN):
            side = "left" if (C_L >= C_R) else "right"
            level = "strong" if (C_max >= MAX_STRONG and gap >= 0.30) else "moderate"
            return level, f"with {level} confidence in {side}"

        return "moderate", "with moderate confidence"

    @classmethod
    def _proj_and_cos(cls, kv, v, left_terms, right_terms, lowercase=True):
        Lm = cls._avg_unit(kv, left_terms, lowercase=lowercase) if left_terms else None
        Rm = cls._avg_unit(kv, right_terms, lowercase=lowercase) if right_terms else None

        cos_left  = float(v @ Lm) if (v is not None and Lm is not None) else 0.0
        cos_right = float(v @ Rm) if (v is not None and Rm is not None) else 0.0

        if Lm is not None and Rm is not None:
            axis_dir = cls._unit(Lm - Rm)
        elif Lm is not None:
            axis_dir = Lm
        elif Rm is not None:
            axis_dir = -Rm
        else:
            return 0.0, cos_left, cos_right, False

        proj = float(v @ axis_dir) if v is not None else 0.0
        return proj, cos_left, cos_right, True

    def run(self, we_model, token="king",
            axis1="man,boy,he,him,his,father,brother,husband|woman,girl,she,her,hers,mother,sister,wife",
            axis2="hot,warm,heat|cold,cool,freezing",
            axis3="royal,king,kingly,noble|common,commoner,peasant,ordinary",
            token_can_be_phrase=True, lowercase=True,
            neutral_eps=0.08, equal_eps=0.02,
            strong_proj=0.35, strong_cos=0.65, strong_gap=0.30):

        kv = we_model.kv
        v = self._phrase_vec(kv, token, token_can_be_phrase=token_can_be_phrase, lowercase=lowercase)
        if v is None:
            return (
                0.0, 0.0, 0.0,
                f"{token} not found in the model vocabulary.",
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )

        # parse
        a1L, a1R, a1L_raw, a1R_raw = self._split_axis(axis1)
        a2L, a2R, a2L_raw, a2R_raw = self._split_axis(axis2)
        a3L, a3R, a3L_raw, a3R_raw = self._split_axis(axis3)

        # projections & cosines
        x, x_cos_left, x_cos_right, ok1 = self._proj_and_cos(kv, v, a1L, a1R, lowercase=lowercase)
        y, y_cos_left, y_cos_right, ok2 = self._proj_and_cos(kv, v, a2L, a2R, lowercase=lowercase)
        z, z_cos_left, z_cos_right, ok3 = self._proj_and_cos(kv, v, a3L, a3R, lowercase=lowercase)

        # build clauses only for strong axes
        clauses = []

        if ok1:
            dir1 = self._direction_phrase(x, self._first_term(a1L_raw), self._first_term(a1R_raw), neutral_eps, equal_eps)
            lvl1, conf1 = self._confidence_level_and_phrase(x_cos_left, x_cos_right)
            strong1 = (abs(x) >= strong_proj) and (
                max(abs(x_cos_left), abs(x_cos_right)) >= strong_cos or
                (max(abs(x_cos_left), abs(x_cos_right)) >= 0.55 and abs(abs(x_cos_left) - abs(x_cos_right)) >= strong_gap)
            )
            if strong1:
                clauses.append(f"{dir1}, {conf1}")

        if ok2:
            dir2 = self._direction_phrase(y, self._first_term(a2L_raw), self._first_term(a2R_raw), neutral_eps, equal_eps)
            lvl2, conf2 = self._confidence_level_and_phrase(y_cos_left, y_cos_right)
            strong2 = (abs(y) >= strong_proj) and (
                max(abs(y_cos_left), abs(y_cos_right)) >= strong_cos or
                (max(abs(y_cos_left), abs(y_cos_right)) >= 0.55 and abs(abs(y_cos_left) - abs(y_cos_right)) >= strong_gap)
            )
            if strong2:
                clauses.append(f"{dir2}, {conf2}")

        if ok3:
            dir3 = self._direction_phrase(z, self._first_term(a3L_raw), self._first_term(a3R_raw), neutral_eps, equal_eps)
            lvl3, conf3 = self._confidence_level_and_phrase(z_cos_left, z_cos_right)
            strong3 = (abs(z) >= strong_proj) and (
                max(abs(z_cos_left), abs(z_cos_right)) >= strong_cos or
                (max(abs(z_cos_left), abs(z_cos_right)) >= 0.55 and abs(abs(z_cos_left) - abs(z_cos_right)) >= strong_gap)
            )
            if strong3:
                clauses.append(f"{dir3}, {conf3}")

        if not clauses:
            summary = f"{token} shows no strong alignment on the provided axes."
        elif len(clauses) == 1:
            summary = f"{token} is {clauses[0]}"
        elif len(clauses) == 2:
            summary = f"{token} is {clauses[0]} and {clauses[1]}"
        else:
            summary = f"{token} is {clauses[0]}, {clauses[1]}, and {clauses[2]}"

        return (
            x, y, z, summary,
            x_cos_left, x_cos_right, y_cos_left, y_cos_right, z_cos_left, z_cos_right
        )


class WordEmbeddingsTokenCentrality:
    """
    Centrality of a token relative to the model's global mean direction.

    Outputs:
      - centrality (FLOAT in [-1, 1]) : cosine(v̂, mean̂)
      - norm       (FLOAT)            : vector length of the raw token embedding
      - summary    (STRING)           : human-friendly description

    Notes:
      - |centrality| ≈ how "typical/generic" the token is in this space.
        * high  |c| -> very central / typical
        * mid   |c| -> somewhat central
        * low   |c| -> distinctive / outlier
      - The sign of centrality is usually not semantically meaningful across models,
        but you can optionally mention orientation if desired.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"we_model": ("WE_MODEL",)},
            "optional": {
                "token": ("STRING", {"default": "king", "multiline": False}),
                "token_can_be_phrase": ("BOOLEAN", {"default": True}),
                "lowercase": ("BOOLEAN", {"default": True}),
                # Summary tuning
                "strong_thr": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "moderate_thr": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "include_orientation": ("BOOLEAN", {"default": False}),   # mention sign of centrality
                "orientation_eps": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.2, "step": 0.005}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("centrality", "norm", "summary")
    FUNCTION = "run"
    CATEGORY = "WordEmbeddings"

    # ---------- helpers ----------
    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @classmethod
    def _phrase_vec(cls, kv, token, token_can_be_phrase=True, lowercase=True):
        parts = [token] if not token_can_be_phrase else [w for w in token.replace(",", " ").split() if w]
        vecs = []
        for w in parts:
            ww = w.lower() if lowercase else w
            try:
                vecs.append(cls._unit(_kv_get_vec_case_insensitive(kv, ww)))
            except KeyError:
                pass
        if not vecs:
            return None, 0.0
        raw = np.mean(vecs, axis=0)
        return cls._unit(raw), float(np.linalg.norm(raw))

    @staticmethod
    def _tier_phrase(mag, strong_thr, moderate_thr):
        if mag >= strong_thr:
            return "highly central (very typical)"
        if mag >= moderate_thr:
            return "moderately central"
        return "weakly central (distinctive/outlier)"

    @staticmethod
    def _orientation_phrase(cent, orientation_eps):
        if cent >= orientation_eps:
            return "with the average orientation"
        if cent <= -orientation_eps:
            return "with opposite-to-average orientation"
        return "with near-orthogonal orientation"

    # ---------- main ----------
    def run(self, we_model,
            token="king",
            token_can_be_phrase=True, lowercase=True,
            strong_thr=0.65, moderate_thr=0.35,
            include_orientation=False, orientation_eps=0.05):

        kv = we_model.kv

        vhat, norm_val = self._phrase_vec(kv, token, token_can_be_phrase=token_can_be_phrase, lowercase=lowercase)
        if vhat is None:
            return (0.0, 0.0, f"{token} not found in the model vocabulary.")

        mv = _get_or_make_mean_vec(kv)          # already unit-length
        cent = float(vhat @ mv)                  # cosine in [-1, 1]
        mag = abs(cent)

        tier = self._tier_phrase(mag, strong_thr, moderate_thr)
        if include_orientation:
            orient = self._orientation_phrase(cent, orientation_eps)
            summary = f"{token} is {tier}, {orient}"
        else:
            summary = f"{token} is {tier}"

        return (cent, norm_val, summary)


# ---------- Registration ----------
NODE_CLASS_MAPPINGS = {
    "WordEmbeddingsLoader": WordEmbeddingsLoader,
    "WordEmbeddingsLocalModelLoader": WordEmbeddingsLocalModelLoader,
    "WordEmbeddingsExplorer": WordEmbeddingsExplorer,
    "WordEmbeddingsInterpolator": WordEmbeddingsInterpolator,
    "WordEmbeddingsEquation": WordEmbeddingsEquation,

    # new token nodes
    "WordEmbeddingsTokenNeighbors": WordEmbeddingsTokenNeighbors,
    "WordEmbeddingsTokenAxis": WordEmbeddingsTokenAxis,
    "WordEmbeddingsTokenAxis2D": WordEmbeddingsTokenAxis2D,
    "WordEmbeddingsTokenAxis3D": WordEmbeddingsTokenAxis3D,
    "WordEmbeddingsTokenCentrality": WordEmbeddingsTokenCentrality,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WordEmbeddingsLoader": "WordEmbeddings: Loader",
    "WordEmbeddingsLocalModelLoader": "WordEmbeddings: Local Loader",
    "WordEmbeddingsExplorer": "WordEmbeddings: Explorer",
    "WordEmbeddingsInterpolator": "WordEmbeddings: Interpolator",
    "WordEmbeddingsEquation": "WordEmbeddings: Equation",

    # new token nodes
    "WordEmbeddingsTokenNeighbors": "WordEmbeddings: Token Neighbors",
    "WordEmbeddingsTokenAxis": "WordEmbeddings: Token Axis",
    "WordEmbeddingsTokenAxis2D": "WordEmbeddings: Axis 2D",
    "WordEmbeddingsTokenAxis3D": "WordEmbeddings: Axis 3D",
    "WordEmbeddingsTokenCentrality": "WordEmbeddings: Token Centrality",
}
