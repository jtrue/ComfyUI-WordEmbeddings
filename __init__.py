# __init__.py  — ComfyUI-WordEmbeddings

EXPERIMENTAL = True  # set to False before pushing

# ---------- Core nodes ----------
from .core_nodes import (
    WordEmbeddingsLoader,
    WordEmbeddingsExplorer,
    WordEmbeddingsInterpolator,
)

NODE_CLASS_MAPPINGS = {
    "WordEmbeddingsLoader": WordEmbeddingsLoader,
    "WordEmbeddingsExplorer": WordEmbeddingsExplorer,
    "WordEmbeddingsInterpolator": WordEmbeddingsInterpolator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WordEmbeddingsLoader": "ComfyUI-WordEmbeddings: Loader",
    "WordEmbeddingsExplorer": "ComfyUI-WordEmbeddings: Explorer",
    "WordEmbeddingsInterpolator": "ComfyUI-WordEmbeddings: Interpolator",
}

# ---------- Experimental nodes (safe to skip if missing/broken) ----------
if EXPERIMENTAL:
    try:
        from .antonym_node import WordEmbeddingsAntonym
        NODE_CLASS_MAPPINGS["WordEmbeddingsAntonym"] = WordEmbeddingsAntonym
        NODE_DISPLAY_NAME_MAPPINGS["WordEmbeddingsAntonym"] = "ComfyUI-WordEmbeddings: Antonym (Experimental)"
    except Exception as e:
        print("[ComfyUI-WordEmbeddings] Skipping Antonym (Experimental):", e)

    try:
        from .analogy_node import WordEmbeddingsAnalogyEquation
        NODE_CLASS_MAPPINGS["WordEmbeddingsAnalogyEquation"] = WordEmbeddingsAnalogyEquation
        NODE_DISPLAY_NAME_MAPPINGS["WordEmbeddingsAnalogyEquation"] = "ComfyUI-WordEmbeddings: Analogy (Experimental)"
    except Exception as e:
        print("[ComfyUI-WordEmbeddings] Skipping Analogy (Experimental):", e)

    try:
        from .polysemy_prompt_node import WordEmbeddingsPromptPolysemy
        NODE_CLASS_MAPPINGS["WordEmbeddingsPromptPolysemy"] = WordEmbeddingsPromptPolysemy
        NODE_DISPLAY_NAME_MAPPINGS["WordEmbeddingsPromptPolysemy"] = "ComfyUI-WordEmbeddings: Prompt Polysemy (Experimental)"
    except Exception as e:
        print("[ComfyUI-WordEmbeddings] Skipping Prompt Polysemy (Experimental):", e)

    try:
        from .prompt_rectify_node import WordEmbeddingsPromptRectify
        NODE_CLASS_MAPPINGS["WordEmbeddingsPromptRectify"] = WordEmbeddingsPromptRectify
        NODE_DISPLAY_NAME_MAPPINGS["WordEmbeddingsPromptRectify"] = "ComfyUI-WordEmbeddings: Prompt Rectify (Experimental)"
    except Exception as e:
        print("[ComfyUI-WordEmbeddings] Skipping Prompt Rectify (Experimental):", e)

    # NEW: Prompt Resolution
    try:
        from .prompt_resolution_node import WordEmbeddingsPromptResolution
        NODE_CLASS_MAPPINGS["WordEmbeddingsPromptResolution"] = WordEmbeddingsPromptResolution
        NODE_DISPLAY_NAME_MAPPINGS["WordEmbeddingsPromptResolution"] = "ComfyUI-WordEmbeddings: Prompt Resolution (Experimental)"
    except Exception as e:
        print("[ComfyUI-WordEmbeddings] Skipping Prompt Resolution (Experimental):", e)
