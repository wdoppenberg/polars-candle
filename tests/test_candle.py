import polars as pl

import polars_candle  # noqa: F401


def test_basic_two_sentences():
    df = pl.DataFrame({"s": ["This is a sentence", "This is another sentence"]})

    df = df.with_columns(
        pl.col("s").candle.embed_text("Snowflake/snowflake-arctic-embed-xs").alias("s_embedding")
    )
    print(df)
    assert df["s_embedding"].dtype == pl.Array

    df = df.explode("s_embedding")

    assert df["s_embedding"].dtype == pl.Float32


def test_basic_with_none():
    df = pl.DataFrame({"s": ["This is a sentence", None, "This is another sentence", None]})

    df = df.with_columns(
        pl.col("s").candle.embed_text("Snowflake/snowflake-arctic-embed-xs").alias("s_embedding")
    )
    print(df)
    assert df["s_embedding"].dtype == pl.Array

    df = df.explode("s_embedding")

    assert df["s_embedding"].dtype == pl.Float32
