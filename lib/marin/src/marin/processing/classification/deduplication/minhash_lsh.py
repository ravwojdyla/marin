# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator
import struct
from typing import Any, TypeVar, TypedDict
from dupekit import hash_xxh3_64
from marin.processing.classification.deduplication.minhash import minhash
from marin.processing.classification.deduplication.text_cleaning import clean_text
from zephyr.dataset import Dataset

T = TypeVar("T")


class MinHashLshInputRecord(TypedDict):
    text: str
    # TODO: is it better to make this Any or generic?
    id: Any


# TODO (rav): can we have just a single id that's expected to be clean on the inputs?
class RecordId(TypedDict):
    record_id: Any
    record_id_norm: int


class MinHashLshOutputRecord(TypedDict):
    bucket: int
    ids: list[RecordId]


def minhash_lsh(
    ds: Dataset[MinHashLshInputRecord], vector_length: int = 286, num_bands: int = 26
) -> Dataset[MinHashLshOutputRecord]:
    """
    Vanilla MinHashLSH implementation using zephyr Dataset API

    Args:
        ds: Input dataset with records containing 'text' and 'id' fields
        vector_length: Length of the MinHash signature vector
        num_bands: Number of bands to split the MinHash signature into for LSH

    Returns:
        A dataset of MinHash LSH output records containing 'bucket' and 'ids' fields
    """

    assert (
        vector_length % num_bands == 0
    ), f"vector_length must be divisible by num_bands, got {vector_length} and {num_bands}"

    return (
        ds.flat_map(lambda record: _minhash_lsh(record, vector_length, num_bands))
        # TODO: should this reducer be moved the the connected components?
        .group_by(
            lambda x: x["bucket"],
            _group_reduce,
        ).filter(lambda record: record)
    )


def _minhash_lsh(record: MinHashLshInputRecord, vector_length: int, num_bands: int) -> Iterator[MinHashLshOutputRecord]:
    text = record["text"]
    # TODO: for now assume `id` exists!
    record_id = record["id"]

    if isinstance(record_id, int):
        record_id_norm = record_id
    elif isinstance(record_id, str):
        record_id_norm = hash_xxh3_64(record_id.encode())
    else:
        raise ValueError(f"Unsupported id type: {type(record_id)}")

    text = clean_text(text)

    sig = minhash(set(text.split()), vector_length=vector_length)

    rows_per_band = len(sig) // num_bands

    for band in range(num_bands):
        start = band * rows_per_band
        end = start + rows_per_band

        bucket = sig[start:end]
        bucket_bytes = struct.pack(f"{len(bucket)}d", *bucket)
        bucket_hash = hash_xxh3_64(bucket_bytes)

        yield {
            "bucket": bucket_hash,
            "ids": [RecordId(record_id=record_id, record_id_norm=record_id_norm)],
        }


def _group_reduce(bucket: int, items: Iterator[MinHashLshOutputRecord]) -> MinHashLshOutputRecord | None:
    all_items = list(items)
    if len(all_items) <= 1:
        return None  # No duplicates in this bucket
    return {
        "bucket": bucket,
        "ids": [item_id for item in all_items for item_id in item["ids"]],
    }
