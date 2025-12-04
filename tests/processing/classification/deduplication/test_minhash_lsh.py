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

from marin.processing.classification.deduplication.minhash_lsh import minhash_lsh
from zephyr.dataset import Dataset


def test_minhash_lsh_happy_path(sync_backend):
    input_data = [
        {"text": "the quick brown fox", "id": 1},
        {"text": "the quick brown fox jumps", "id": 2},
        {"text": "lorem ipsum dolor sit amet", "id": 3},
    ]

    ds = Dataset.from_list(input_data)

    lsh_result = minhash_lsh(ds, vector_length=128, num_bands=32)

    output = sync_backend.execute(lsh_result)
    for b in output:
        assert [i["record_id"] for i in b["ids"]] == [1, 2]


def test_minhash_docs(sync_backend, docs):
    input_data = [{"text": text, "id": doc_id} for doc_id, text in docs.items()]

    ds = Dataset.from_list(input_data)

    lsh_result = minhash_lsh(ds, vector_length=128, num_bands=32)

    output = sync_backend.execute(lsh_result)

    for b in output:
        assert [i["record_id"] for i in b["ids"]] == ["doc_1", "doc_1_diff_header"]
