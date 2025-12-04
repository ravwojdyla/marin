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
from dataclasses import dataclass
import logging
from typing import TypedDict
from collections.abc import Sequence

from marin.processing.classification.deduplication.minhash_lsh import MinHashLshOutputRecord, RecordId
from zephyr.backends import Backend
from zephyr.dataset import Dataset
from zephyr.readers import load_file

logger = logging.getLogger(__name__)


# TODO: dataclasses are not writable to parquet ATM, is there something better than TypedDict that works out of the box?
class CCNode(TypedDict):
    node_id: RecordId
    adjacency_list: list[int]
    component_id: int
    changed: bool


@dataclass
class CCMessage: ...


@dataclass
class SelfMessage(CCMessage):
    node: CCNode


@dataclass
class CompMessage(CCMessage):
    component_id: int


# TODO: what's the best way to model the input type?
CCInput = MinHashLshOutputRecord


# TODO (rav): instead of a single cc function that requires a backend, do we want to split it up
# into more functional core functions that can be composed into an imperative flow externally?
def connected_components(
    ds: Dataset[CCInput], backend: Backend, output_dir: str, max_iterations: int = 10
) -> tuple[bool, Sequence]:
    """TODO"""

    curr_it = backend.execute(
        ds
        # Go from bucket -> links
        .flat_map(_gen_links_within_buckets)
        # Construct Node state, init with:
        #  * each node is its own component
        #  * adjacency list from links
        .group_by(
            lambda x: x[0]["record_id_norm"],
            _build_adjacency,
        ).write_jsonl(f"{output_dir}/it_0/part-{{shard:05d}}.jsonl")
    )

    converged = False
    for i in range(1, max_iterations + 1):  # type: ignore[bad-assignment]
        logger.info(f"Connected components iteration {i}...")
        curr_it = backend.execute(
            Dataset.from_list(curr_it)
            .flat_map(load_file)
            .map(lambda record: CCNode(**record))
            .flat_map(_emit_messages)
            .group_by(key=lambda x: x[0], reducer=_reduce_node_step)
            # NOTE: parquet built-in does not support list of int :/
            .write_jsonl(f"{output_dir}/it_{i}/part-{{shard:05d}}.jsonl")
        )

        # Check for convergence
        changes = backend.execute(
            Dataset.from_list(curr_it)
            .flat_map(lambda f: load_file(f, columns=["changed"]))
            .filter(lambda node: node["changed"])
            # TODO: why is there no .count() method?
            .map(lambda _: 1)
            .reduce(sum)
        )

        num_changes = changes[0]

        if num_changes == 0:
            converged = True
            logger.info(f"Connected components converged after {i} iterations.")
            break
        else:
            logger.info(f"Connected components iteration {i} found {num_changes} changes.")

    return converged, curr_it


def _gen_links_within_buckets(record: MinHashLshOutputRecord) -> Iterator[tuple[RecordId, RecordId]]:
    ids = record.get("ids", [])

    norm_ids = [i["record_id_norm"] for i in ids]
    # TODO: this will materialize ids!
    if len(norm_ids) != len(set(norm_ids)):
        raise ValueError("Duplicate para_ids found in bucket during link_reduce.!")

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] == ids[j]:
                continue

            yield (ids[i], ids[j])
            yield (ids[j], ids[i])


def _build_adjacency(node_id: RecordId, links: Iterator[tuple[RecordId, RecordId]]) -> CCNode:
    all_links = list(links)
    return CCNode(
        node_id=all_links[0][0],
        adjacency_list=list(set([link[1]["record_id_norm"] for link in all_links])),
        # init with own id as component
        component_id=node_id,
        changed=True,
    )


def _emit_messages(node: CCNode) -> Iterator[tuple[int, CCMessage]]:
    """
    1. Emit the node structure to itself (to preserve graph topology).
    2. Emit the current component ID to all neighbors.
    """
    # 1. Preserve structure
    yield (node["node_id"]["record_id_norm"], SelfMessage(node=node))

    # 2. Propagate component ID to neighbors
    # (Optimization: Only send if we changed recently, but strictly Hash-to-Min sends always)
    msg = CompMessage(component_id=node["component_id"])
    for neighbor_id in node["adjacency_list"]:
        yield (neighbor_id, msg)


def _reduce_node_step(key: int, incoming: Iterator[tuple[int, CCMessage]]) -> CCNode:
    """
    1. Recover NodeState.
    2. Find minimum component ID from messages.
    3. Update state if a smaller ID is found.
    """
    min_comp = float("inf")
    node_structure: CCNode | None = None

    # Iterate through mixed stream of structure and messages
    for _, msg in incoming:
        if isinstance(msg, SelfMessage):
            node_structure = msg.node
            if node_structure["component_id"] < min_comp:
                min_comp = node_structure["component_id"]
        else:
            assert isinstance(msg, CompMessage)
            remote_comp = msg.component_id
            if remote_comp < min_comp:
                min_comp = remote_comp

    if node_structure is None:
        # Should technically not happen if graph is well-formed
        raise ValueError(f"Lost structure for node {key}")

    if min_comp < node_structure["component_id"]:
        assert isinstance(min_comp, int)
        node_structure["component_id"] = min_comp
        node_structure["changed"] = True
    else:
        node_structure["changed"] = False

    return node_structure
