# __init__.py — ComfyUI-WordEmbeddings

from .core_nodes import (
    WordEmbeddingsLoader,
    WordEmbeddingsLocalModelLoader,
    WordEmbeddingsExplorer,
    WordEmbeddingsInterpolator,
    WordEmbeddingsEquation,
    WordEmbeddingsTokenNeighbors,
    WordEmbeddingsTokenAxis,
    WordEmbeddingsTokenAxis2D,
    WordEmbeddingsTokenAxis3D,
    WordEmbeddingsTokenCentrality,
)

NODE_CLASS_MAPPINGS = {
    "WordEmbeddingsLoader": WordEmbeddingsLoader,
    "WordEmbeddingsLocalModelLoader": WordEmbeddingsLocalModelLoader,
    "WordEmbeddingsExplorer": WordEmbeddingsExplorer,
    "WordEmbeddingsInterpolator": WordEmbeddingsInterpolator,
    "WordEmbeddingsEquation": WordEmbeddingsEquation,
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
    "WordEmbeddingsTokenNeighbors": "WordEmbeddings: Token Neighbors",
    "WordEmbeddingsTokenAxis": "WordEmbeddings: Token Axis",
    "WordEmbeddingsTokenAxis2D": "WordEmbeddings: Token Axis2D",
    "WordEmbeddingsTokenAxis3D": "WordEmbeddings: Token Axis3D",
    "WordEmbeddingsTokenCentrality": "WordEmbeddings: Token Centrality",
}
