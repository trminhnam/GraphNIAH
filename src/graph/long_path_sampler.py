"""Sample long simple paper paths from an undirected citation graph.

A papers-only path may follow a citation in either direction, so the graph is
undirected. Finding the longest simple path is NP-hard, so the sampler prunes
with upper bounds instead of exact lengths:

1. A connected component with fewer than ``num_nodes`` papers cannot hold a
   path, so its papers are removed.
2. The graph is split into biconnected components ("blocks"). Two blocks share
   at most one paper (a cut vertex), so a path that enters a block can visit at
   most every paper of that block and then leave through one cut vertex. A
   rerooted dynamic program over the block-cut tree turns this into, for every
   paper and every block around it, the most papers that a path starting there
   can reach. These bounds are exact wherever the graph is a tree.
3. Papers that no ``num_nodes``-paper path can contain are removed, and steps
   1-2 repeat until nothing changes.
4. A randomized depth-first search starts at a paper that can end such a path
   and only steps to neighbors whose bound still allows the target length.
   Inside a block, the bound also subtracts the block papers already on the
   path. On tree-like parts of the graph the search therefore never backtracks.

Each pruning round costs O(V + E). Node ``i`` of a ``CitationGraph`` is the
paper ``corpus_ids[i]``; a ``PrunedGraph`` keeps the subset ``nodes``.
"""

from __future__ import annotations

import random
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class CitationGraph:
    """Undirected citation graph in CSR form, with the citation directions."""

    corpus_ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    # Sorted ``citing * num_nodes + cited`` keys of the directed citations.
    citation_keys: np.ndarray

    @property
    def num_nodes(self) -> int:
        return len(self.corpus_ids)

    @property
    def num_edges(self) -> int:
        return len(self.indices) // 2

    def cites(self, citing: int, cited: int) -> bool:
        key = citing * self.num_nodes + cited
        position = int(np.searchsorted(self.citation_keys, key))
        return (
            position < len(self.citation_keys)
            and int(self.citation_keys[position]) == key
        )


@dataclass(frozen=True)
class BlockBounds:
    """Path-length upper bounds, counted in papers, from the block-cut tree."""

    # Block of every CSR entry; both directions of an edge share it.
    edge_block: np.ndarray
    block_size: np.ndarray
    # Per block, the three cut vertices with the most papers beyond them
    # (``exit_vertices`` is -1 and ``exit_values`` 0 where unused).
    exit_values: np.ndarray
    exit_vertices: np.ndarray
    # Most papers on a simple path that ends at / contains the node.
    end_bound: np.ndarray
    through_bound: np.ndarray
    # Per CSR entry u -> w: most papers after u on a path whose next paper is w.
    entry_bound: np.ndarray


@dataclass(frozen=True)
class PrunedGraph:
    """The papers that may lie on a path, with their bounds."""

    nodes: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    bounds: BlockBounds
    rounds: list[dict[str, int]]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


@dataclass
class SamplerStats:
    attempts: int = 0
    paths: int = 0
    duplicates: int = 0
    exhausted_starts: int = 0
    step_limit_hits: int = 0
    steps: int = 0


def csr_from_pairs(
    rows: np.ndarray,
    cols: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CSR arrays whose rows and per-row columns are sorted."""
    order = np.lexsort((cols, rows))
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=num_nodes), out=indptr[1:])
    return indptr, cols[order].astype(np.int64)


def build_citation_graph(citing: np.ndarray, cited: np.ndarray) -> CitationGraph:
    """Build the undirected graph; self-citations and repeated edges are dropped."""
    citing = np.asarray(citing, dtype=np.int64)
    cited = np.asarray(cited, dtype=np.int64)
    if citing.shape != cited.shape:
        raise ValueError("citing and cited must have the same length")
    keep = citing != cited
    citing, cited = citing[keep], cited[keep]
    corpus_ids, inverse = np.unique(
        np.concatenate([citing, cited]), return_inverse=True
    )
    num_nodes = len(corpus_ids)
    source = inverse[: len(citing)].astype(np.int64)
    target = inverse[len(citing) :].astype(np.int64)
    citation_keys = np.unique(source * num_nodes + target)
    pair_keys = np.unique(
        np.minimum(source, target) * num_nodes + np.maximum(source, target)
    )
    low, high = np.divmod(pair_keys, max(num_nodes, 1))
    indptr, indices = csr_from_pairs(
        np.concatenate([low, high]), np.concatenate([high, low]), num_nodes
    )
    return CitationGraph(corpus_ids, indptr, indices, citation_keys)


def entry_rows(indptr: np.ndarray) -> np.ndarray:
    """Source node of every CSR entry."""
    return np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))


def induced_subgraph(
    indptr: np.ndarray,
    indices: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the CSR arrays of the nodes in ``keep`` and their old indices."""
    kept = np.flatnonzero(keep)
    new_index = np.full(len(keep), -1, dtype=np.int64)
    new_index[kept] = np.arange(len(kept), dtype=np.int64)
    rows = entry_rows(indptr)
    kept_entries = keep[rows] & keep[indices]
    # The renumbering keeps the order, so rows and columns stay sorted.
    sub_rows = new_index[rows[kept_entries]]
    sub_indptr = np.zeros(len(kept) + 1, dtype=np.int64)
    np.cumsum(np.bincount(sub_rows, minlength=len(kept)), out=sub_indptr[1:])
    return sub_indptr, new_index[indices[kept_entries]], kept


def component_size_per_node(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    num_nodes = len(indptr) - 1
    if num_nodes == 0:
        return np.zeros(0, dtype=np.int64)
    matrix = csr_matrix(
        (np.ones(len(indices), dtype=np.int8), indices, indptr),
        shape=(num_nodes, num_nodes),
    )
    _, labels = connected_components(matrix, directed=False)
    return np.bincount(labels)[labels]


def biconnected_edge_blocks(
    indptr: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Label every CSR entry with its block (iterative Hopcroft-Tarjan)."""
    num_nodes = len(indptr) - 1
    if len(indices) == 0:
        return np.zeros(0, dtype=np.int64), 0
    rows = entry_rows(indptr)
    # The reverse entry of u -> w; keys are sorted because the CSR is.
    twin = np.searchsorted(rows * num_nodes + indices, indices * num_nodes + rows)
    twin_list = twin.tolist()
    ptr = indptr.tolist()
    adj = indices.tolist()
    discovery = [-1] * num_nodes
    low = [0] * num_nodes
    edge_block = [-1] * len(adj)
    edge_stack: list[int] = []
    num_blocks = 0
    clock = 0
    for root in range(num_nodes):
        if discovery[root] != -1:
            continue
        discovery[root] = low[root] = clock
        clock += 1
        vertex_stack = [root]
        parent_entry = [-1]
        next_entry = [ptr[root]]
        while vertex_stack:
            u = vertex_stack[-1]
            position = next_entry[-1]
            if position < ptr[u + 1]:
                next_entry[-1] = position + 1
                incoming = parent_entry[-1]
                if incoming != -1 and position == twin_list[incoming]:
                    continue
                w = adj[position]
                if discovery[w] == -1:
                    edge_stack.append(position)
                    discovery[w] = low[w] = clock
                    clock += 1
                    vertex_stack.append(w)
                    parent_entry.append(position)
                    next_entry.append(ptr[w])
                elif discovery[w] < discovery[u]:
                    # Back edge to an ancestor; the other direction is skipped.
                    edge_stack.append(position)
                    if discovery[w] < low[u]:
                        low[u] = discovery[w]
                continue
            vertex_stack.pop()
            next_entry.pop()
            incoming = parent_entry.pop()
            if incoming == -1:
                continue
            parent = vertex_stack[-1]
            if low[u] < low[parent]:
                low[parent] = low[u]
            if low[u] >= discovery[parent]:
                while True:
                    entry = edge_stack.pop()
                    edge_block[entry] = num_blocks
                    edge_block[twin_list[entry]] = num_blocks
                    if entry == incoming:
                        break
                num_blocks += 1
    return np.asarray(edge_block, dtype=np.int64), num_blocks


def offsets(values: np.ndarray, size: int) -> np.ndarray:
    pointer = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(np.bincount(values, minlength=size), out=pointer[1:])
    return pointer


def block_cut_tree_exits(
    block_size: np.ndarray,
    block_cut_ptr: np.ndarray,
    block_cuts: np.ndarray,
    vertex_block_ptr: np.ndarray,
    vertex_blocks: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each block's three best exits as (papers beyond, cut vertex).

    The papers beyond cut vertex ``c`` of block ``b`` are the most papers a
    path starting at ``c`` can add without entering ``b``, not counting ``c``.
    One down pass and one up pass over the block-cut tree give this for every
    (block, cut vertex) pair, whichever side of the tree the block lies on.
    """
    size = block_size.tolist()
    num_blocks = len(size)
    bc_ptr = block_cut_ptr.tolist()
    cuts = block_cuts.tolist()
    vb_ptr = vertex_block_ptr.tolist()
    vblocks = vertex_blocks.tolist()

    parent_cut = [-1] * num_blocks
    order: list[int] = []
    reached = bytearray(num_blocks)
    for root in range(num_blocks):
        if reached[root]:
            continue
        reached[root] = 1
        head = len(order)
        order.append(root)
        while head < len(order):
            block = order[head]
            head += 1
            parent = parent_cut[block]
            for i in range(bc_ptr[block], bc_ptr[block + 1]):
                cut = cuts[i]
                if cut == parent:
                    continue
                for j in range(vb_ptr[cut], vb_ptr[cut + 1]):
                    child = vblocks[j]
                    if child != block:
                        reached[child] = 1
                        parent_cut[child] = cut
                        order.append(child)

    # Down pass: in_down[b] = most papers on a path from b's parent cut vertex
    # into b and below; away_down[c] = most papers from c into its child blocks.
    away_down = [1] * num_nodes
    in_down = [0] * num_blocks
    for block in reversed(order):
        parent = parent_cut[block]
        best = 0
        for i in range(bc_ptr[block], bc_ptr[block + 1]):
            cut = cuts[i]
            if cut != parent and away_down[cut] - 1 > best:
                best = away_down[cut] - 1
        value = size[block] + best
        in_down[block] = value
        if parent != -1 and value > away_down[parent]:
            away_down[parent] = value

    # Up pass: up_exit[b] = papers beyond b's parent cut vertex, seen from b.
    up_exit = [0] * num_blocks
    values = [[0, 0, 0] for _ in range(num_blocks)]
    vertices = [[-1, -1, -1] for _ in range(num_blocks)]
    for block in order:
        parent = parent_cut[block]
        a1 = a2 = a3 = 0
        c1 = c2 = c3 = -1
        if parent != -1 and up_exit[block] > 0:
            a1, c1 = up_exit[block], parent
        for i in range(bc_ptr[block], bc_ptr[block + 1]):
            cut = cuts[i]
            if cut == parent:
                continue
            value = away_down[cut] - 1
            if value > a1:
                a3, c3, a2, c2, a1, c1 = a2, c2, a1, c1, value, cut
            elif value > a2:
                a3, c3, a2, c2 = a2, c2, value, cut
            elif value > a3:
                a3, c3 = value, cut
        values[block] = [a1, a2, a3]
        vertices[block] = [c1, c2, c3]
        for i in range(bc_ptr[block], bc_ptr[block + 1]):
            cut = cuts[i]
            if cut == parent:
                continue
            # Most papers on a path from the cut vertex into this block.
            into_block = size[block] + (a2 if cut == c1 else a1)
            best = second = 0
            best_child = -1
            for j in range(vb_ptr[cut], vb_ptr[cut + 1]):
                child = vblocks[j]
                if child == block:
                    continue
                value = in_down[child]
                if value > best:
                    best, second, best_child = value, best, child
                elif value > second:
                    second = value
            for j in range(vb_ptr[cut], vb_ptr[cut + 1]):
                child = vblocks[j]
                if child != block:
                    sibling = second if child == best_child else best
                    up_exit[child] = max(into_block, sibling) - 1
    return (
        np.asarray(values, dtype=np.int64).reshape(num_blocks, 3),
        np.asarray(vertices, dtype=np.int64).reshape(num_blocks, 3),
    )


def compute_block_bounds(indptr: np.ndarray, indices: np.ndarray) -> BlockBounds:
    """Compute upper bounds on simple path lengths for every node and entry."""
    num_nodes = len(indptr) - 1
    edge_block, num_blocks = biconnected_edge_blocks(indptr, indices)
    if num_blocks == 0:
        ones = np.ones(num_nodes, dtype=np.int64)
        empty = np.zeros((0, 3), dtype=np.int64)
        return BlockBounds(
            edge_block,
            np.zeros(0, dtype=np.int64),
            empty,
            empty,
            ones,
            ones.copy(),
            np.zeros(0, dtype=np.int64),
        )

    rows = entry_rows(indptr)
    pair_block, pair_vertex = np.divmod(
        np.unique(edge_block * num_nodes + rows), num_nodes
    )
    block_size = np.bincount(pair_block, minlength=num_blocks)
    is_cut = np.bincount(pair_vertex, minlength=num_nodes) >= 2
    cut_pairs = is_cut[pair_vertex]
    cut_block = pair_block[cut_pairs]
    cut_vertex = pair_vertex[cut_pairs]
    by_vertex = np.argsort(cut_vertex, kind="stable")
    exit_values, exit_vertices = block_cut_tree_exits(
        block_size,
        offsets(cut_block, num_blocks),
        cut_vertex,
        offsets(cut_vertex, num_nodes),
        cut_block[by_vertex],
        num_nodes,
    )

    def exit_columns(blocks: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            *(exit_values[blocks, column] for column in range(3)),
            *(exit_vertices[blocks, column] for column in range(3)),
        )

    v1, v2, v3, x1, x2, _ = exit_columns(pair_block)
    size = block_size[pair_block]
    is_first = x1 == pair_vertex
    # A path starting at the node and entering the block.
    start_bound = size + np.where(is_first, v2, v1)
    # A path with both neighbors of the node in the block (needs 3+ papers).
    inside_bound = size + np.where(
        is_first, v2 + v3, np.where(x2 == pair_vertex, v1 + v3, v1 + v2)
    )
    inside_bound = np.where(size >= 3, inside_bound, start_bound)

    end_bound = np.ones(num_nodes, dtype=np.int64)
    np.maximum.at(end_bound, pair_vertex, start_bound)
    through_bound = end_bound.copy()
    np.maximum.at(through_bound, pair_vertex, inside_bound)
    # A path whose two neighbors of the node lie in different blocks.
    order = np.lexsort((-start_bound, pair_vertex))
    sorted_vertex = pair_vertex[order]
    sorted_bound = start_bound[order]
    is_run_start = np.ones(len(order), dtype=bool)
    is_run_start[1:] = sorted_vertex[1:] != sorted_vertex[:-1]
    second = np.flatnonzero(~is_run_start & np.roll(is_run_start, 1))
    np.maximum.at(
        through_bound,
        sorted_vertex[second],
        sorted_bound[second - 1] + sorted_bound[second] - 1,
    )

    e1, e2, _, f1, _, _ = exit_columns(edge_block)
    entry_bound = block_size[edge_block] + np.where(f1 == rows, e2, e1) - 1
    return BlockBounds(
        edge_block,
        block_size,
        exit_values,
        exit_vertices,
        end_bound,
        through_bound,
        entry_bound,
    )


def prune_for_path_length(graph: CitationGraph, num_nodes: int) -> PrunedGraph:
    """Remove papers that no simple path of ``num_nodes`` papers can contain."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    nodes = np.arange(graph.num_nodes, dtype=np.int64)
    indptr, indices = graph.indptr, graph.indices
    rounds: list[dict[str, int]] = []
    while True:
        keep = component_size_per_node(indptr, indices) >= num_nodes
        if not keep.all():
            indptr, indices, kept = induced_subgraph(indptr, indices, keep)
            nodes = nodes[kept]
        bounds = compute_block_bounds(indptr, indices)
        keep = bounds.through_bound >= num_nodes
        rounds.append(
            {
                "nodes": len(nodes),
                "edges": len(indices) // 2,
                "blocks": len(bounds.block_size),
                "largest_block": int(bounds.block_size.max(initial=0)),
                "removed_by_bound": int((~keep).sum()),
            }
        )
        if keep.all():
            return PrunedGraph(nodes, indptr, indices, bounds, rounds)
        indptr, indices, kept = induced_subgraph(indptr, indices, keep)
        nodes = nodes[kept]


def canonical_path_key(path: list[int]) -> tuple[int, ...]:
    """A path and its reverse are the same path."""
    forward = tuple(path)
    backward = forward[::-1]
    return min(forward, backward)


class LongPathSampler:
    """Randomized bounded depth-first search over a ``PrunedGraph``.

    Paths are lists of the pruned graph's node indices. Every start paper is
    drawn uniformly from the papers that can end a path, and every step picks
    uniformly among the neighbors whose bound still allows the target length.
    With ``max_paths_per_node``, a node is blocked once that many recorded
    paths contain it, which limits how much the paths overlap. Blocking only
    removes nodes, so every bound stays a valid upper bound.
    """

    def __init__(
        self,
        pruned: PrunedGraph,
        num_nodes: int,
        rng: random.Random,
        *,
        max_steps_per_attempt: int,
        max_paths_per_node: int | None = None,
    ) -> None:
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")
        if max_steps_per_attempt < 1:
            raise ValueError("max_steps_per_attempt must be positive")
        if max_paths_per_node is not None and max_paths_per_node < 1:
            raise ValueError("max_paths_per_node must be positive")
        bounds = pruned.bounds
        # Each node's neighbors, most promising first, so a scan can stop at
        # the first neighbor whose bound is too small.
        order = np.lexsort((-bounds.entry_bound, entry_rows(pruned.indptr)))
        self.num_nodes = num_nodes
        self.rng = rng
        self.max_steps_per_attempt = max_steps_per_attempt
        self.pointer = pruned.indptr.tolist()
        self.neighbors = pruned.indices[order].tolist()
        self.entry_blocks = bounds.edge_block[order].tolist()
        self.negated_entry_bounds = (-bounds.entry_bound[order]).tolist()
        self.block_size = bounds.block_size.tolist()
        self.exit_values = [column.tolist() for column in bounds.exit_values.T]
        self.exit_vertices = [column.tolist() for column in bounds.exit_vertices.T]
        # Blocked nodes stay marked; path nodes are marked during a walk.
        self.unavailable = bytearray(pruned.num_nodes)
        self.starts = np.flatnonzero(bounds.end_bound >= num_nodes).tolist()
        self.seen: set[tuple[int, ...]] = set()
        self.max_paths_per_node = max_paths_per_node
        # Recorded paths through each node (only while a limit is set).
        self.paths_per_node: dict[int, int] = {}
        self.stats = SamplerStats()

    def block_nodes(self, nodes: list[int]) -> None:
        """Never use these nodes again (for example, papers without metadata)."""
        for node in nodes:
            self.unavailable[node] = 1

    def record_path(self, path: list[int]) -> None:
        """Count a sampled or saved path: never repeat it, and apply the node limit."""
        self.seen.add(canonical_path_key(path))
        limit = self.max_paths_per_node
        if limit is None:
            return
        for node in path:
            count = self.paths_per_node.get(node, 0) + 1
            self.paths_per_node[node] = count
            if count >= limit:
                self.unavailable[node] = 1

    def remove_start(self, index: int) -> None:
        self.starts[index] = self.starts[-1]
        self.starts.pop()

    def best_exit(self, block: int, entry: int, current: int) -> int:
        """Papers beyond the best exit of ``block`` other than ``entry``/``current``."""
        for column in range(3):
            vertex = self.exit_vertices[column][block]
            if vertex != entry and vertex != current:
                return self.exit_values[column][block]
        return 0

    def walk(self, start: int) -> tuple[list[int] | None, bool]:
        """Search from ``start``; return (path or None, start proven useless).

        A node's candidates are the neighbor entries whose static bound allows
        the remaining length: a prefix of its sorted entries, found by binary
        search. They are drawn lazily in random order (a sparse Fisher-Yates
        shuffle), and a drawn entry that is on the path or fails the in-block
        bound is discarded, so a step at a hub paper costs O(1), not O(degree).
        """
        target = self.num_nodes
        pointer = self.pointer
        neighbors = self.neighbors
        entry_blocks = self.entry_blocks
        negated_bounds = self.negated_entry_bounds
        block_size = self.block_size
        unavailable = self.unavailable
        randrange = self.rng.randrange
        path = [start]
        unavailable[start] = 1
        # Per path position: block of the current in-block segment, the paper
        # where the path entered it, and papers of that block on the path.
        segment_block = [-1]
        segment_entry = [-1]
        segment_count = [0]
        # Per path position: first candidate entry, untried candidates (-1
        # until first needed), and the sparse shuffle's swapped slots.
        first_entry = [0]
        untried = [-1]
        swapped: list[dict[int, int]] = [{}]
        steps = 0
        try:
            while True:
                length = len(path)
                if length == target:
                    return list(path), False
                current = path[-1]
                if untried[-1] < 0:
                    low = pointer[current]
                    first_entry[-1] = low
                    untried[-1] = (
                        bisect_right(
                            negated_bounds, length - target, low, pointer[current + 1]
                        )
                        - low
                    )
                chosen = -1
                in_block = segment_block[-1]
                slots = swapped[-1]
                while untried[-1]:
                    last = untried[-1] - 1
                    slot = randrange(last + 1)
                    i = first_entry[-1] + slots.get(slot, slot)
                    if slot != last:
                        slots[slot] = slots.get(last, last)
                    untried[-1] = last
                    if unavailable[neighbors[i]]:
                        continue
                    block = entry_blocks[i]
                    if block == in_block and (
                        block_size[block]
                        - segment_count[-1]
                        + self.best_exit(block, segment_entry[-1], current)
                        < target - length
                    ):
                        continue
                    chosen = i
                    break
                if chosen < 0:
                    if length == 1:
                        return None, True
                    unavailable[path.pop()] = 0
                    segment_block.pop()
                    segment_entry.pop()
                    segment_count.pop()
                    first_entry.pop()
                    untried.pop()
                    swapped.pop()
                    continue
                if steps >= self.max_steps_per_attempt:
                    return None, False
                steps += 1
                self.stats.steps += 1
                i = chosen
                block = entry_blocks[i]
                if block == segment_block[-1]:
                    segment_entry.append(segment_entry[-1])
                    segment_count.append(segment_count[-1] + 1)
                else:
                    segment_entry.append(current)
                    segment_count.append(2)
                segment_block.append(block)
                first_entry.append(0)
                untried.append(-1)
                swapped.append({})
                path.append(neighbors[i])
                unavailable[neighbors[i]] = 1
        finally:
            for node in path:
                unavailable[node] = 0

    def sample(self, count: int, *, max_failed_attempts: int) -> Iterator[list[int]]:
        """Yield up to ``count`` new distinct paths.

        Stops early when no start paper is left or after
        ``max_failed_attempts`` attempts in a row without a new path.
        """
        produced = 0
        failures = 0
        while produced < count and failures < max_failed_attempts and self.starts:
            index = self.rng.randrange(len(self.starts))
            start = self.starts[index]
            if self.unavailable[start]:
                self.remove_start(index)
                continue
            self.stats.attempts += 1
            path, exhausted = self.walk(start)
            if path is None:
                failures += 1
                if exhausted:
                    self.stats.exhausted_starts += 1
                    self.remove_start(index)
                else:
                    self.stats.step_limit_hits += 1
                continue
            key = canonical_path_key(path)
            if key in self.seen:
                failures += 1
                self.stats.duplicates += 1
                continue
            self.record_path(path)
            failures = 0
            produced += 1
            self.stats.paths += 1
            yield path
