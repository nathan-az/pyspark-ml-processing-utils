from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.ml.feature import Bucketizer
from decimal import Decimal
from itertools import chain


def split_dataset_by_date(df, date_col, output_col, ordered_split_rules):
    # splits are stored as decimal at first so precision is not lost during addition
    splits = [Decimal("0.")]
    mapper = []

    for i, (group, proportion) in enumerate(ordered_split_rules.items()):
        splits.append(splits[-1] + Decimal(str(proportion)))
        mapper.append((i, group))

    id_to_group_mapper = f.create_map([f.lit(x) for x in chain(*mapper)])

    # after additions are complete, revert to float as it is required by the Spark transformer
    splits = [float(x) for x in splits]
    if not splits[-1] == 1:
        raise ValueError(
            f"Sum of values in ordered_split rules must add to 1.0, not `{splits[-1]}`"
        )

    num_rows = df.count()
    cumulative_percentages = (
        df.groupBy(date_col)
        .count()
        .withColumn("count_perc", f.col("count") / num_rows)
        .withColumn(
            "cum_perc",
            f.round(f.sum("count_perc").over(Window.orderBy(date_col)), 5),
        )
    )

    bucketizer = Bucketizer(splits=splits, inputCol="cum_perc", outputCol=output_col)
    bucketed = bucketizer.transform(cumulative_percentages)
    bucketed = bucketed.withColumn(output_col, id_to_group_mapper[f.col(output_col)])
    bucketed = bucketed.select(date_col, output_col)
    bucketed = f.broadcast(bucketed)

    df = df.join(other=bucketed, on=[date_col], how="inner")

    return df
