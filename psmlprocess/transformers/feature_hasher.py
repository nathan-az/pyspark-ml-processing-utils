from typing import Optional, List, Literal

from pyspark import keyword_only
from pyspark.ml.param import Param
from pyspark.sql import functions as f, DataFrame
from pyspark.ml.base import Transformer
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql.types import StringType


def replace_suffix(s: str, add_suffix: str, remove_suffix: Optional[str]):
    if not remove_suffix or len(remove_suffix) == 0:
        return f"{s}{add_suffix}"

    else:
        s = s[: -len(remove_suffix)]
        return f"{s}{add_suffix}"


def get_features_to_hash(df: DataFrame, cat_features: List[str], max_dim: int):
    approx_counts = (
        df.select(*cat_features)
        .agg(*[f.approx_count_distinct(c).alias(c) for c in cat_features])
        .collect()
    )
    for_hashing = {k for k, v in approx_counts[0].asDict().items() if v > max_dim}
    return for_hashing


VALID_NULL_TREATMENTS = Literal["keep", "hash", "-1"]


def hash_columns(
    df: DataFrame,
    to_hash: List[str],
    after_hash: List[str],
    max_dim: int,
    salt: Optional[str] = "",
    handle_nulls: VALID_NULL_TREATMENTS = "hash",
):
    if not len(to_hash) == len(after_hash):
        raise ValueError(
            f"`to_hash` and `after_hash` must be arrays with same length. {len(to_hash)=}, {len(after_hash)=}"
        )

    if handle_nulls == "hash":
        hash_fn = lambda x: f.abs(
            f.hash(f.concat(f.col(x).cast(StringType()), f.lit(salt))) % max_dim
        )
    elif handle_nulls == "keep":
        hash_fn = lambda x: f.when(f.col(x).isNull(), None).otherwise(
            f.abs(f.hash(f.concat(f.col(x).cast(StringType()), f.lit(salt))) % max_dim)
        )
    elif handle_nulls == "-1":
        hash_fn = lambda x: f.when(f.col(x).isNull(), -1).otherwise(
            f.abs(f.hash(f.concat(f.col(x).cast(StringType()), f.lit(salt))) % max_dim)
        )
    else:
        raise ValueError(
            f"Argument `handle_nulls` must be one of `{VALID_NULL_TREATMENTS}`, got {handle_nulls}"
        )

    for before, after in zip(to_hash, after_hash):
        df = df.withColumn(after, hash_fn(before))

    return df


class MultiFeatureHasher(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(
        self, inputCols=None, outputCols=None, maxDim=32, handleNull="hash", numHashes=1
    ):
        super().__init__()
        self.inputCols = Param(self, "inputCols", "inputCols")
        self.outputCols = Param(self, "outputCols", "inputCols")
        self.maxDim = Param(self, "maxDim", "maxDim")
        self.handleNull = Param(self, "handleNull", "handleNull")
        self.numHashes = Param(self, "numHashes", "numHashes")

        self._setDefault(inputCols=inputCols)
        self._setDefault(outputCols=outputCols)
        self._setDefault(maxDim=maxDim)
        self._setDefault(handleNull=handleNull)
        self._setDefault(numHashes=numHashes)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, maxDim=32, handleNull="hash"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getInputCols(self):
        return self.getOrDefault(self.inputCols)

    def getOutputCols(self):
        return self.getOrDefault(self.outputCols)

    def getMaxDim(self):
        return self.getOrDefault(self.maxDim)

    def getHandleNull(self):
        return self.getOrDefault(self.handleNull)

    def getNumHashes(self):
        return self.getOrDefault(self.numHashes)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCols(self, value):
        return self._set(outputCols=value)

    def setMaxDim(self, value):
        return self._set(maxDim=value)

    def setHandleNull(self, value):
        return self._set(handleNull=value)

    def setNumHashes(self, value):
        return self._set(numHashes=value)

    def _transform(self, df):
        num_hashes = self.getNumHashes()
        output_cols = self.getOutputCols()

        if num_hashes == 1:
            return hash_columns(
                df,
                to_hash=self.getInputCols(),
                after_hash=self.getOutputCols(),
                max_dim=self.getMaxDim(),
                handle_nulls=self.getHandleNull(),
            )
        else:
            for i in range(num_hashes):
                salt = str(i)
                salted_output_cols = [f"{x}_{salt}" for x in output_cols]
                df = hash_columns(
                    df,
                    to_hash=self.getInputCols(),
                    after_hash=salted_output_cols,
                    max_dim=self.getMaxDim(),
                    salt=salt,
                    handle_nulls=self.getHandleNull(),
                )
            return df
