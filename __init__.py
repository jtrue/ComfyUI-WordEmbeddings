# __init__.py — ComfyUI-WordEmbeddings (core only)

from .core_nodes import (
    WordEmbeddingsLoader,
    WordEmbeddingsExplorer,
    WordEmbeddingsInterpolator,
    WordEmbeddingsEquation,
)

NODE_CLASS_MAPPINGS = {
    "WordEmbeddingsLoader": WordEmbeddingsLoader,
    "WordEmbeddingsExplorer": WordEmbeddingsExplorer,
    "WordEmbeddingsInterpolator": WordEmbeddingsInterpolator,
    "WordEmbeddingsEquation": WordEmbeddingsEquation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WordEmbeddingsLoader": "ComfyUI-WordEmbeddings: Loader",
    "WordEmbeddingsExplorer": "ComfyUI-WordEmbeddings: Explorer",
    "WordEmbeddingsInterpolator": "ComfyUI-WordEmbeddings: Interpolator",
    "WordEmbeddingsEquation": "ComfyUI-WordEmbeddings: Equation",
}
