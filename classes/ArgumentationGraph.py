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
    
    def __init__(self, active_hypotheses: bool = False, debug: bool = True):
        self.G = nx.DiGraph()
        self.bag = grad.BAG()
        self.node_text_map: Dict[str, str] = {}
        self.node_counter = 0
        self.active_hypotheses = active_hypotheses
        self.debug = debug

        # NEW: track nodes and relations added in last extend
        self.last_added_arguments: List[str] = []
        self.last_added_relations: List[Tuple[str, str, str]] = []







    def add_argument(self, text: str, node_type: str = "argument", initial_strength: float = 0.5) -> str:
        node_id = f"{node_type[0].upper()}{self.node_counter}"
        self.node_counter += 1
        self.G.add_node(node_id, type=node_type, text=text, strength=initial_strength)
        self.bag.arguments[node_id] = Argument(node_id, initial_weight=initial_strength)
        self.node_text_map[node_id] = text

        # Track added argument
        self.last_added_arguments.append(node_id)

        return node_id

    def add_relation(self, src: str, tgt: str, relation: str):
        self.G.add_edge(src, tgt, relation=relation)
        if relation == "support":
            self.bag.add_support(self.bag.arguments[src], self.bag.arguments[tgt])
        elif relation == "attack":
            self.bag.add_attack(self.bag.arguments[src], self.bag.arguments[tgt])

        # Track added relation
        self.last_added_relations.append((src, tgt, relation))

    def compute_strengths(self, semantic: str) -> Dict[str, float]:
        
        # Recupera dinamicamente la classe dal modulo semantics
        ModelClass = getattr(grad.semantics, semantic)

        # Istanzia la classe
        model = ModelClass()

        model.BAG = self.bag
        model.approximator = algorithms.RK4(model)
        model.solve(delta=1e-2, epsilon=1e-4)

        strengths: Dict[str, float] = {}
        for arg in self.bag.arguments.values():
            strengths[arg.name] = float(getattr(arg, "strength", arg.get_initial_weight()))

        nx.set_node_attributes(self.G, strengths, "strength")
        return strengths




 




    def get_text_from_id(self, node_id: str) -> str:
        return self.node_text_map.get(node_id, "")

    def print_graph(self, header: str = "Argumentation Graph State") -> str:
        lines = ["="*60, header, "-"*60, "Nodes (id | type | strength | text):"]
        for nid, data in self.G.nodes(data=True):
            lines.append(f"{nid:4s} | {data.get('type','?'):10s} | {data.get('strength',0):.3f} | {data.get('text','')[:200]}")
        lines.append("\nRelations (src -> tgt : relation):")
        for src, tgt, edata in self.G.edges(data=True):
            lines.append(f"{src} -> {tgt} : {edata.get('relation', '?')}")
        lines.append("="*60+"\n")
        txt = "\n".join(lines)
        print(txt)
        return txt

        
    def prune_isolated_arguments(self):
        """Rimuove tutti gli argomenti isolati, ma mai le ipotesi."""
        if not PRUNE_ISOLATED_ARGUMENTS:
            return

        isolated = list(nx.isolates(self.G))
        removed = []

        for nid in isolated:
            ntype = self.G.nodes[nid].get("type", "")
            if ntype == "hypothesis":
                continue  # Never prune hypotheses
            self.G.remove_node(nid)
            self.bag.arguments.pop(nid, None)
            self.node_text_map.pop(nid, None)
            removed.append(nid)

        if self.debug and removed:
            print(f"🧹 Pruned isolated arguments (excluding hypotheses): {removed}")



    def save_graph(self, dataset_name: str, sample_idx: int, save_dir: Path = None):
        """Save the graph (GraphML, PNG, TXT) with clean separated legend panel."""

        # -----------------------------
        # Create directory
        # -----------------------------
        graph_dir = (save_dir / dataset_name) if save_dir else (Path("graphs") / dataset_name)
        graph_dir.mkdir(parents=True, exist_ok=True)

        graphml_path = graph_dir / f"sample_{sample_idx}.graphml"
        png_path     = graph_dir / f"sample_{sample_idx}.png"
        txt_path     = graph_dir / f"sample_{sample_idx}.txt"

        # -----------------------------
        # Save GraphML
        # -----------------------------
        nx.write_graphml(self.G, graphml_path)

        # -----------------------------
        # Prepare data for graph drawing
        # -----------------------------
        strengths = nx.get_node_attributes(self.G, "strength")
        node_colors = [strengths.get(n, 0.5) for n in self.G.nodes]
        pos = nx.spring_layout(self.G, seed=42)

        # -----------------------------
        # Create figure with 2 panels
        # -----------------------------
        fig, (ax_graph, ax_legend) = plt.subplots(
            1, 2,
            figsize=(18, 10),
            gridspec_kw={'width_ratios': [3, 1]}  # left bigger than right
        )

        # -----------------------------
        # Draw graph on left subplot
        # -----------------------------
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax_graph,
            node_color=node_colors,
            cmap="coolwarm",
            node_size=1200,
            edgecolors="black"
        )

        nx.draw_networkx_labels(self.G, pos, ax=ax_graph, font_size=8)

        # Separate edges
        attack_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get("relation") == "attack"]
        support_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get("relation") == "support"]

        nx.draw_networkx_edges(
            self.G, pos, ax=ax_graph,
            edgelist=attack_edges,
            edge_color="red",
            arrows=True,
            arrowsize=20,
            width=2.0
        )

        nx.draw_networkx_edges(
            self.G, pos, ax=ax_graph,
            edgelist=support_edges,
            edge_color="green",
            style="dashed",
            arrows=True,
            arrowsize=20,
            width=1.8
        )

        ax_graph.set_title(f"{dataset_name} — Sample {sample_idx}", fontsize=14)
        ax_graph.axis("off")

        # -----------------------------
        # Legend on right subplot
        # -----------------------------
        legend_text = "Legend (ID | Strength | Text):\n\n"
        for nid, data in self.G.nodes(data=True):
            text_short = data.get('text', '').replace("\n", " ")[:90]  # preview text
            legend_text += f"{nid} | {data.get('strength', 0):.3f}\n{text_short}\n\n"

        ax_legend.text(
            0.01, 0.99, legend_text,
            va="top", ha="left",
            fontsize=9,
            wrap=True
        )
        ax_legend.axis("off")

        # -----------------------------
        # Save PNG
        # -----------------------------
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

        # -----------------------------
        # Save TXT summary
        # -----------------------------
        lines = ["="*60, f"Graph State: {dataset_name} Sample {sample_idx}", "-"*60]

        # Nodes
        lines.append("Nodes (id | type | strength | text):")
        for nid, data in self.G.nodes(data=True):
            lines.append(f"{nid:4s} | {data.get('type','?'):10s} | "
                         f"{data.get('strength',0):.3f} | {data.get('text','')[:200]}")

        # Edges
        lines.append("\nRelations (src -> tgt : relation):")
        for src, tgt, edata in self.G.edges(data=True):
            lines.append(f"{src} -> {tgt} : {edata.get('relation', '?')}")

        # Legend
        lines.append("\nLegend (ID | Strength | Text):")
        for nid, data in self.G.nodes(data=True):
            text_short = data.get('text', '').replace("\n", " ")[:90]
            lines.append(f"{nid:4s} | {data.get('strength',0):.3f} | {text_short}")

        lines.append("="*60 + "\n")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"💾 Saved graph → {graphml_path.name}, {png_path.name}, {txt_path.name}")
