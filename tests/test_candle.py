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


TEST_SENTENCES = (
    "The sun was shining brightly on the beach.",
    "Artificial intelligence is revolutionizing healthcare.",
    "The new smartphone has a battery life of over 10 hours.",
    "Climate change is causing severe droughts in Africa.",
    "The first human settlement on Mars is scheduled for 2050.",
    "The novel 'To Kill a Mockingbird' is a classic of American literature.",
    "The human brain can process information up to 300 words per minute.",
    "The concept of time travel remains purely theoretical.",
    "The new coffee shop in town offers a unique blend of Ethiopian and Colombian beans.",
    "NASA's Curiosity rover has been exploring Mars since 2012.",
    "A team of scientists has discovered a new species of jellyfish in the Great Barrier Reef.",
    "The average lifespan of a blue whale is around 80 years.",
    "The first recorded Olympics took place in ancient Greece.",
    "There are over 7,000 languages spoken worldwide.",
    "The world's largest waterfall is actually located underwater.",
    "A group of friends went hiking and got lost in the woods.",
    "The concept of dark matter remains a mystery to scientists.",
    "Amazon has acquired Whole Foods Market for $13.7 billion.",
    "The first manned mission to the moon was Apollo 11.",
    "The average human heart beats around 3 billion times during their lifetime.",
    "There are over 100,000 known species of plants on Earth.",
    "A new study suggests that meditation can reduce stress by 40%.",
    "The world's largest snowflake was recorded in Montana.",
    "The first iPhone was released in 2007.",
    "A team of engineers has developed a sustainable method for producing biofuels.",
    "The average person consumes over 100,000 calories per year.",
    "There are over 1.6 billion people who use Facebook every day.",
    "NASA's Hubble Space Telescope has been operational since 1990.",
    "The first recorded human migration took place around 60,000 years ago.",
    "A group of scientists has discovered a new type of dinosaur in Argentina.",
    "The average lifespan of a redwood tree is over 2,000 years.",
    "There are over 3 million known species of insects on Earth.",
    "Amazon's Alexa smart speaker can control up to 100 devices at once.",
    "The world's largest living organism is a fungus that covers over 2,200 acres.",
    "A new study suggests that reading books can improve mental health by 30%.",
    "There are over 5,000 languages spoken in Africa.",
    "The first solar-powered car was developed in the 1970s.",
    "NASA's Mars Reconnaissance Orbiter has been operational since 2006.",
    "The average human nose can detect over 1 trillion different scents.",
    "A group of friends went skydiving and saw a UFO.",
    "The concept of parallel universes remains purely theoretical.",
    "There are over 300,000 known species of fish in the world's oceans.",
    "Amazon has acquired Zappos for $1.2 billion.",
    "The first recorded human rights were outlined in the Magna Carta.",
    "A team of scientists has discovered a new type of black hole.",
    "The average person consumes over 60 pounds of sugar per year.",
    "There are over 3,000 languages spoken in India.",
    "NASA's Kepler space telescope has been operational since 2009.",
    "The world's largest diamond mine is located in South Africa.",
    "A group of friends went camping and saw a bear.",
    "The concept of time dilation remains purely theoretical.",
    "Amazon's Prime Air delivery service uses drones to deliver packages.",
    "The first recorded human migration took place around 40,000 years ago.",
    "There are over 200,000 known species of beetles in the world.",
    "A team of engineers has developed a sustainable method for producing biofuels from algae.",
    "The average person consumes over 3.5 liters of coffee per year.",
    "NASA's Cassini spacecraft has been operational since 1997.",
)


def test_large_text():
    df = pl.DataFrame({"s": TEST_SENTENCES})

    df = df.with_columns(
        pl.col("s")
        .candle.embed_text("Snowflake/snowflake-arctic-embed-xs", device="gpu")
        .alias("s_embedding")
    )

    df_similarity = (
        df.join(df, on="s_embedding", how="cross", suffix="_1")
        .filter(pl.col("s") != pl.col("s_1"))
        .with_columns(
            pl.col("s_embedding").dist_arr.cosine("s_embedding_1").alias("cos_dist")
        )
        .filter(pl.col("cos_dist") < 0.20)
        .select(
            "s",
            "s_1",
            "cos_dist",
        )
    )

    print(df_similarity)
