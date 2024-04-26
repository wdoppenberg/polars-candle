import polars as pl

import polars_candle  # noqa: F401
import polars_distance  # noqa: F401


def test_basic_two_sentences():
    df = pl.DataFrame({"s": ["This is a sentence", "This is another sentence"]})

    df = df.with_columns(
        pl.col("s")
        .candle.embed_text("Snowflake/snowflake-arctic-embed-xs")
        .alias("s_embedding")
    )
    print(df)
    assert df["s_embedding"].dtype == pl.Array

    df_similarity = (
        df.join(df, on="s_embedding", how="cross", suffix="_1")
        .filter(pl.col("s") != pl.col("s_1"))
        .with_columns(
            pl.col("s_embedding").dist_arr.cosine("s_embedding_1").alias("cos_dist")
        )
    )

    print(df_similarity)

    assert df_similarity["cos_dist"].mean() < 0.1

    df = df.explode("s_embedding")

    assert df["s_embedding"].dtype == pl.Float32


def test_basic_two_sentences_with_gpu():
    df = pl.DataFrame({"s": ["This is a sentence", "This is another sentence"]})

    df = df.with_columns(
        pl.col("s")
        .candle.embed_text("Snowflake/snowflake-arctic-embed-xs", device="gpu")
        .alias("s_embedding")
    )


def test_basic_with_none():
    df = pl.DataFrame(
        {"s": ["This is a sentence", None, "This is another sentence", None]}
    )

    df = df.with_columns(
        pl.col("s")
        .candle.embed_text("Snowflake/snowflake-arctic-embed-xs")
        .alias("s_embedding")
    )
    print(df)

    # Check if the None values are still there
    assert df["s_embedding"].null_count() == 2

    # Check if the None values are in the correct position
    df_check = df.with_columns(
        pl.col("s").is_null().alias("is_null"),
        pl.col("s_embedding").is_null().alias("is_null_embedding"),
    )
    assert df_check.select(
        pl.col("is_null").eq(pl.col("is_null_embedding")).all()
    ).item()


def test_pooling():
    df = pl.DataFrame({"s": ["This is a sentence", "This is another sentence"]})

    df = df.with_columns(
        pl.col("s")
        .candle.embed_text("Snowflake/snowflake-arctic-embed-xs", pooling="max")
        .alias("s_embedding")
    )
    print(df)
    assert df["s_embedding"].dtype == pl.Array

    df = df.explode("s_embedding")

    assert df["s_embedding"].dtype == pl.Float32
    assert df["s_embedding"].max() > 0.0


def test_normalize():
    df = pl.DataFrame({"s": ["This is a sentence"]})

    df = df.with_columns(
        pl.col("s")
        .candle.embed_text("Snowflake/snowflake-arctic-embed-xs", normalize=True)
        .alias("s_embedding")
    )

    df = df.explode("s_embedding")
    # Check if the embedding's length is 1

    assert df.select(pl.col("s_embedding").pow(2)).sum().item() - 1.0 < 1e-5


def test_lazyframe():
    lf = pl.DataFrame({"s": ["This is a sentence", "This is another sentence"]}).lazy()

    df = lf.with_columns(
        pl.col("s")
        .candle.embed_text("Snowflake/snowflake-arctic-embed-xs")
        .alias("s_embedding")
    ).collect()

    print(df)
    assert df["s_embedding"].dtype == pl.Array

    df_similarity = (
        df.join(df, on="s_embedding", how="cross", suffix="_1")
        .filter(pl.col("s") != pl.col("s_1"))
        .with_columns(
            pl.col("s_embedding").dist_arr.cosine("s_embedding_1").alias("cos_dist")
        )
    )

    print(df_similarity)

    assert df_similarity["cos_dist"].mean() < 0.1

    df_expl = df.explode("s_embedding")

    assert df_expl["s_embedding"].dtype == pl.Float32
