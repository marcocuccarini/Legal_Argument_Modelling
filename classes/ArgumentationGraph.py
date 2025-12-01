import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from huggingface_hub import login

import torch
from sentence_transformers import SentenceTransformer, util
import uncertainpy2.gradual as grad 


class ArgumentationGraph:
    def __init__(self, delta=1e-1, epsilon=1e-4):
        self.BAG = grad.BAG()
        self.model = grad.semantics.ContinuousDFQuADModel()
        self.model.BAG = self.BAG
        self.model.approximator = grad.algorithms.RK4(self.model)

        self.DELTA = delta
        self.EPSILON = epsilon

        self.arguments = {}

    # ----------------------
    # Argument + relations
    # ----------------------
    def add_argument(self, name, initial_strength=0.5):
        arg = grad.Argument(name, initial_strength)
        self.arguments[name] = arg
        self.BAG.add_argument(arg)
        return arg

    def add_relation(self, source_name, target_name, relation_type="support", strength=1.0):

        source = self.arguments[source_name]
        target = self.arguments[target_name]

        if relation_type == "support":
            self.BAG.add_support(source, target, strength)
            return

        if relation_type == "attack":
            self.BAG.add_attack(source, target, strength)
            return

        if relation_type == "defense":
            # Create counter-attack node
            node_name = f"{source_name}_defends_{target_name}"
            counter_arg = grad.Argument(node_name, 0.5)
            self.arguments[node_name] = counter_arg
            self.BAG.add_argument(counter_arg)

            # defense = support → attack chain
            self.BAG.add_support(source, counter_arg, strength)
            self.BAG.add_attack(counter_arg, target, strength)
            return

        raise ValueError("relation_type must be 'support', 'attack', or 'defense'.")

    # ----------------------
    # Solving / weights
    # ----------------------
    def solve(self):
        return self.model.solve(self.DELTA, self.EPSILON, True, True)

    def get_weights(self):
        return {name: arg.strength for name, arg in self.arguments.items()}

    def print_weights(self):
        print("Argument strengths:")
        for name, arg in self.arguments.items():
            print(f"{name}: {arg.strength:.4f}")

    def plot_strengths(self):
        plot = grad.plotting.strengthplot(self.model, self.DELTA, self.EPSILON)
        plot.show()

    def run(self):
        self.solve()
        self.print_weights()
        self.plot_strengths()

    # -----------------------------
    # Save graph (GraphML, PNG, TXT)
    # -----------------------------
    def save_graph(self, dataset_name: str, sample_idx: int, save_dir: Path = None):

        graph_dir = (save_dir / dataset_name) if save_dir else (Path("graphs") / dataset_name)
        graph_dir.mkdir(parents=True, exist_ok=True)

        graphml_path = graph_dir / f"sample_{sample_idx}.graphml"
        png_path = graph_dir / f"sample_{sample_idx}.png"
        txt_path = graph_dir / f"sample_{sample_idx}.txt"

        # Build graph
        G = nx.DiGraph()

        # Nodes
        for name, arg in self.arguments.items():
            G.add_node(name, strength=float(arg.strength), text=str(name), type="argument")

        # Edges (safe access to BAG)
        attacks = getattr(self.BAG, "attacks", [])
        supports = getattr(self.BAG, "supports", [])

        for atk in attacks:
            G.add_edge(atk.source.name, atk.target.name,
                       relation="attack", weight=float(atk.strength))

        for sup in supports:
            G.add_edge(sup.source.name, sup.target.name,
                       relation="support", weight=float(sup.strength))

        # Save GraphML
        nx.write_graphml(G, graphml_path)

        # -----------------------------
        # Draw PNG
        # -----------------------------
        strengths = nx.get_node_attributes(G, "strength")
        node_colors = [float(strengths.get(n, 0.5)) for n in G.nodes]

        pos = nx.spring_layout(G, seed=42)

        fig, (ax_graph, ax_legend) = plt.subplots(
            1, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [3, 1]}
        )

        nx.draw_networkx_nodes(
            G, pos, ax=ax_graph,
            node_color=node_colors, cmap="coolwarm",
            node_size=1200, edgecolors="black"
        )

        nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=8)

        attack_edges = [(u, v) for u, v, d in G.edges(data=True) if d["relation"] == "attack"]
        support_edges = [(u, v) for u, v, d in G.edges(data=True) if d["relation"] == "support"]

        nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=attack_edges,
                               edge_color="red", arrows=True, width=2)

        nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=support_edges,
                               edge_color="green", style="dashed", arrows=True, width=2)

        ax_graph.set_title(f"{dataset_name} — Sample {sample_idx}")
        ax_graph.axis("off")

        # Legend panel
        legend_text = "Legend (ID | Strength):\n\n"
        for nid, data in G.nodes(data=True):
            legend_text += f"{nid} | {data['strength']:.3f}\n"

        ax_legend.text(0.01, 0.99, legend_text, va="top", ha="left", fontsize=9)
        ax_legend.axis("off")

        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

        # -----------------------------
        # Save TXT
        # -----------------------------
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f"Graph State: {dataset_name} Sample {sample_idx}\n")
            f.write("="*60 + "\n\n")

            f.write("Nodes:\n")
            for nid, data in G.nodes(data=True):
                f.write(f"{nid} | {data['strength']:.3f}\n")

            f.write("\nEdges:\n")
            for src, tgt, ed in G.edges(data=True):
                f.write(f"{src} -> {tgt} ({ed['relation']})\n")

        print(f"Saved graph → {graphml_path.name}, {png_path.name}, {txt_path.name}")