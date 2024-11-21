from pathlib import Path
from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

ModelName = Literal["bert-base-uncased", "bert-base-cased"]

# Old models that don't belong to any organization
# Source https://github.com/UKPLab/sentence-transformers/blob/c0fc0e8238f7f48a1e92dc90f6f96c86f69f1e02/sentence_transformers/SentenceTransformer.py#L219
__MODEL_HUB_ORGANIZATION__ = "sentence-transformers"
__BASIC_TRANSFORMER_MODELS__ = (
    "bert-base-cased-finetuned-mrpc",
    "bert-base-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "bert-base-uncased",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking",
    "bert-large-cased",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-uncased-whole-word-masking",
    "bert-large-uncased",
    "camembert-base",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "distilbert-base-uncased",
    "distilgpt2",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-base",
    "roberta-large-mnli",
    "roberta-large-openai-detector",
    "roberta-large",
)


@pl.api.register_expr_namespace("candle")
class CandleExt:
    """
    Extension for Polars to use deep learning models using the candle library.

    Polars namespace: candle

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "text": [
    ...         "This is a test.",
    ...         "This is another test.",
    ...     ],
    ... })
    >>> df.with_columns(
    ...     pl.col("text").candle.embed_text("bert-base-cased")
    ... )
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def embed_text(
        self,
        model_repo: str,
        pooling: Literal["max", "sum", "mean"] = "mean",
        normalize: bool = False,
        device: Literal["cpu", "gpu"] = "cpu",
    ) -> pl.Expr:
        """
        Embed text using a pre-trained model.

        Parameters
        ----------
        model_repo
                The repository name of the text embedding model to use. E.g. "sentence-transformers/all-MiniLM-L6-v2".
        pooling
                The pooling strategy to use. One of "max", "sum", or "mean".
        normalize
                Whether to normalize (L2) the embeddings - meaning that all embeddings will have a length of 1.
        device
                The device to use for the model. One of "cpu" or "gpu". Will select either a Metal or CUDA device
                depending on the availability.

        Returns
        -------
        Expr
                An expression with the embedded text.
        """
        if "/" not in model_repo:
            model_repo = __MODEL_HUB_ORGANIZATION__ + "/" + model_repo

        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="embed_text",
            args=[self._expr],
            kwargs={
                "model_repo": model_repo,
                "pooling": pooling,
                "normalize": normalize,
                "device": device,
            },
            is_elementwise=True,
        )
