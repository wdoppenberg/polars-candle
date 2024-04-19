# `polars-candle`

A [`polars`](https://pola.rs/) extension for running [`candle`](https://github.com/huggingface/candle) ML
models on `polars` DataFrames. 

# Example

Pull any applicable model from Huggingface, such as the recently released 
[Snowflake model](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs), and embed text using a simple API.

```python
import polars as pl
import polars_candle  # ignore: F401

df = pl.DataFrame({"s": ["This is a sentence", "This is another sentence"]})

df = df.with_columns(
    pl.col("s").candle.embed_text("Snowflake/snowflake-arctic-embed-xs").alias("s_embedding")
)
print(df)
# ┌──────────────────────────┬───────────────────────────────────┐
# │ s                        ┆ s_embedding                       │
# │ ---                      ┆ ---                               │
# │ str                      ┆ array[f32, 384]                   │
# ╞══════════════════════════╪═══════════════════════════════════╡
# │ This is a sentence       ┆ [-0.056457, 0.559411, … -0.20403… │
# │ This is another sentence ┆ [-0.117206, 0.336827, … 0.174078… │
# └──────────────────────────┴───────────────────────────────────┘
```

Currently, Bert, JinaBert, and Distilbert models are supported. More models will be added in the future. Check 
my other repository [`wdoppenberg/glowrs`](https://github.com/wdoppenberg/glowrs) to learn more about the underlying 
implementation for sentence embedding.

# Installation

Clone the repository and install the package using:

```bash
pip install .
```

_Note:_ PyPI package is not available yet, will be in the future.

If you're on a Mac with an ARM processor, the library will install with Metal acceleration by default. 
Should you want more control over the installation, you can install the package using:

```bash
maturin develop --release -F <feature>
```

Where `<feature>` can be one of the following:
* `metal` Install with Metal acceleration.
* `cuda` Install with CUDA acceleration.
* `accelerate` Install with the Accelerate framework.


# Roadmap

- [x] Embed text using Bert, JinaBert, and Distilbert models.
- [ ] Add more models.
- [ ] More configuration options for embedding (e.g. pooling strategy, device selection, etc.).
- [ ] Support & test streaming workloads.


# Credits

- Massive thanks to [`polars`](https://pola.rs/) & their contributors for providing a blazing fast DataFrame library 
with the ability to extend it with custom functions using [`pyo3-polars`](https://github.com/pola-rs/pyo3-polars).
- Great work so far by Huggingface on [`candle`](https://github.com/huggingface/candle) for providing a simple
interface to run ML models.


# Note

This is a work in progress and the API might change in the future. Feel free to open an issue if you have any
suggestions or improvements.
