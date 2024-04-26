from pathlib import Path
from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

ModelName = Literal["bert-base-uncased", "bert-base-cased"]


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
    ...     pl.col("text").candle.embed_text("bert-base-uncased")
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
