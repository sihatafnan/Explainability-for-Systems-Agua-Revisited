#!/usr/bin/env python3
"""
Plot NN output-mapper saliency results for a given domain.

Usage (from domain folder, e.g., abr/):

    python plot_nn_saliencies.py \
        --saliency_path data/nn_saliency.npz \
        --concepts_file data/concepts.txt \
        --top_k 10

It will generate:
    - nn_saliency_topK_concepts.png
    - nn_saliency_topK_concepts_bins.png
in the same directory as the saliency file.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_concept_names(concepts_file: Path, n_concepts: int):
    """
    Load concept names from a text file.
    Each line is treated as a full label string.
    Falls back to 'Concept i' if anything goes wrong.
    """
    names = [f"Concept {i}" for i in range(n_concepts)]
    if concepts_file is None or not concepts_file.exists():
        print(f"[warn] Concept file {concepts_file} not found. Using generic concept names.")
        return names

    with open(concepts_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    if len(lines) < n_concepts:
        print(f"[warn] Concept file has only {len(lines)} lines, but n_concepts={n_concepts}. "
              "Remaining concepts will use generic names.")
    for i in range(min(len(lines), n_concepts)):
        # You can optionally strip leading indices like '1. ' here if present:
        # e.g., "1. Volatile Network Throughput" -> "Volatile Network Throughput"
        raw = lines[i]
        # Try to remove leading "X. " pattern if it exists
        if raw[:2].isdigit() and raw[2:4] == ". ":
            names[i] = raw[4:]
        else:
            names[i] = raw
    return names


def plot_topK_concepts(per_concept, concept_names, out_path: Path, top_k: int = 10):
    """
    Plot bar chart of top-K most important concepts (global saliency).
    """
    n_concepts = len(per_concept)
    top_k = min(top_k, n_concepts)

    # Sort concepts by importance descending
    ranked = sorted(
        [(i, s) for i, s in enumerate(per_concept)],
        key=lambda x: x[1],
        reverse=True,
    )
    top_indices = [i for (i, _) in ranked[:top_k]]
    top_scores = np.array([per_concept[i] for i in top_indices])
    top_labels = [concept_names[i] for i in top_indices]

    # Shorten labels if they are very long
    def shorten(lbl, max_len=28):
        return lbl if len(lbl) <= max_len else lbl[: max_len - 3] + "..."

    short_labels = [shorten(lbl) for lbl in top_labels]

    x = np.arange(len(top_indices))

    plt.figure(figsize=(8, 4))
    plt.bar(x, top_scores)
    plt.xticks(x, short_labels, rotation=45, ha="right")
    plt.ylabel("Global saliency (|input × gradient|)")
    plt.xlabel("Concept")
    # plt.title(f"Top-{top_k} Most Influential Concepts (NN Output Mapper)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[info] Saved top-K concepts plot to {out_path}")


def plot_topK_concepts_bins(per_concept_bin, per_concept, concept_names,
                            out_path: Path, top_k: int = 10):
    """
    Plot stacked bar chart: for top-K concepts, show bin-wise saliency
    (so that bar height equals total importance, colored segments = bins).
    """
    n_concepts, n_bins = per_concept_bin.shape
    top_k = min(top_k, n_concepts)

    # Rank by global importance
    ranked = sorted(
        [(i, per_concept[i]) for i in range(n_concepts)],
        key=lambda x: x[1],
        reverse=True,
    )
    top_indices = [i for (i, _) in ranked[:top_k]]
    top_labels = [concept_names[i] for i in top_indices]
    top_bins = per_concept_bin[top_indices, :]  # shape (K, n_bins)

    # Shorten labels
    def shorten(lbl, max_len=28):
        return lbl if len(lbl) <= max_len else lbl[: max_len - 3] + "..."

    short_labels = [shorten(lbl) for lbl in top_labels]

    x = np.arange(len(top_indices))

    # Stacked bars: each bin is a segment
    bottoms = np.zeros(len(top_indices))
    plt.figure(figsize=(8, 4))
    for b in range(n_bins):
        segment = top_bins[:, b]
        plt.bar(x, segment, bottom=bottoms, label=f"Bin {b}")
        bottoms += segment

    plt.xticks(x, short_labels, rotation=45, ha="right")
    plt.ylabel("Saliency (per concept, per bin)")
    plt.xlabel("Concept")
    # plt.title(f"Top-{top_k} Concepts – Bin-wise Saliency (Stacked)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[info] Saved top-K stacked bin saliency plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saliency_path",
        type=str,
        required=True,
        help="Path to nn_saliency.npz (e.g., data/nn_saliency.npz)",
    )
    parser.add_argument(
        "--concepts_file",
        type=str,
        default="../abr/data/concept_names.txt",
        help="Path to concepts.txt or concept_names.txt (optional).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top concepts to visualize.",
    )
    args = parser.parse_args()

    sal_path = Path(args.saliency_path)
    if not sal_path.exists():
        raise FileNotFoundError(f"{sal_path} does not exist")

    data = np.load(sal_path)
    # Use your actual keys:
    per_concept = data["saliency_per_concept"]          # (n_concepts,)
    per_concept_bin = data["saliency_per_concept_bin"]  # (n_concepts, n_bins)

    n_concepts, n_bins = per_concept_bin.shape
    print(f"[info] Loaded saliency from {sal_path}")
    print(f"       n_concepts = {n_concepts}, n_bins = {n_bins}")

    concepts_path = Path(args.concepts_file) if args.concepts_file else None
    concept_names = load_concept_names(concepts_path, n_concepts)

    c_names = []
    for cn in concept_names:
        cn = cn[3:]
        c_names.append(cn)
    
    concept_names = c_names
        

    out_dir = sal_path.parent
    out_topk = out_dir / "nn_saliency_topK_concepts.pdf"
    out_bins = out_dir / "nn_saliency_topK_concepts_bins.pdf"

    plot_topK_concepts(per_concept, concept_names, out_topk, top_k=args.top_k)
    plot_topK_concepts_bins(per_concept_bin, per_concept, concept_names, out_bins, top_k=args.top_k)


if __name__ == "__main__":
    main()