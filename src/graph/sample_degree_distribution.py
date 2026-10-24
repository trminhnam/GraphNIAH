"""Exact paper degrees in stored QA needle and distractor graphs."""

from collections import Counter
from typing import Any

METRICS = ("in_degree", "out_degree", "total_degree")


def calculate_graph_degrees(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    """Union graphs by paper identity, then count directed citation edges.

    Each graph needs ``expanded_paper_ids`` and ``references_by_paper``.
    """
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for graph in graphs:
        expanded = graph.get("expanded_paper_ids")
        references = graph.get("references_by_paper")
        if not isinstance(expanded, list) or not isinstance(references, dict):
            raise TypeError(
                "Each graph needs expanded_paper_ids and references_by_paper."
            )
        nodes.update(
            p for p in expanded if isinstance(p, str) and p.startswith("paper:")
        )
        for owner, targets in references.items():
            if not isinstance(targets, list):
                raise TypeError(f"References for {owner!r} must be a list.")
            if not owner.startswith("paper:"):
                continue
            nodes.add(owner)
            for target in targets:
                if isinstance(target, str) and target.startswith("paper:"):
                    nodes.add(target)
                    edges.add((owner, target))

    incoming = Counter(target for _, target in edges)
    outgoing = Counter(owner for owner, _ in edges)
    paper_degrees = [
        {
            "paper_id": paper,
            "in_degree": incoming[paper],
            "out_degree": outgoing[paper],
            "total_degree": incoming[paper] + outgoing[paper],
        }
        for paper in sorted(nodes)
    ]
    return {
        "num_papers": len(nodes),
        "num_citation_edges": len(edges),
        "paper_degrees": paper_degrees,
        **{
            f"{metric}_histogram": {
                str(degree): count
                for degree, count in sorted(
                    Counter(p[metric] for p in paper_degrees).items()
                )
            }
            for metric in METRICS
        },
    }


def calculate_sample_degrees(question: dict[str, Any]) -> dict[str, Any]:
    """Degrees over a stored QA sample: its visible graph plus negative trees."""
    main = question.get("reachability", {}).get("visible_graph")
    trees = question.get("paper_negative_trees")
    if not isinstance(main, dict) or not isinstance(trees, list):
        raise TypeError("Each sample needs visible_graph and paper_negative_trees.")
    degrees = calculate_graph_degrees([main, *trees])
    return {
        "num_papers": degrees.pop("num_papers"),
        "num_citation_edges": degrees.pop("num_citation_edges"),
        "num_distractor_trees": len(trees),
        **degrees,
    }


def aggregate_sample_degrees(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool paper observations; repeated papers in different samples count again."""
    result = {
        "num_samples": len(samples),
        "num_paper_observations": sum(s["num_papers"] for s in samples),
        "num_citation_edge_observations": sum(s["num_citation_edges"] for s in samples),
    }
    for metric in METRICS:
        counts: Counter[int] = Counter()
        for sample in samples:
            counts.update({int(k): v for k, v in sample[f"{metric}_histogram"].items()})
        result[f"{metric}_histogram"] = {
            str(degree): count for degree, count in sorted(counts.items())
        }
    return result
