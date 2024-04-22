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

	def embed_text(self, model_repo: str, pooling: Literal["max", "sum", "mean"] = "mean") -> pl.Expr:
		"""
		Embed text using a pre-trained model.

		Parameters
		----------
		model_repo
			The repository name of the text embedding model to use. E.g. "sentence-transformers/all-MiniLM-L6-v2".
		pooling
			The pooling strategy to use. One of "max", "sum", or "mean".

		Returns
		-------
		Expr
			An expression with the embedded text.
		"""

		return register_plugin_function(
			plugin_path=Path(__file__).parent,
			function_name="embed_text",
			args=[self._expr],
			kwargs={"model_repo": model_repo, "pooling": pooling},
			is_elementwise=True,
		)
