"""Panel chatbot demo with knowledge graph visualization.

Combines the MultiStepAgent (entity creation from natural language)
with a panelini chat UI and a VisNetwork graph that incrementally
visualizes extracted entities and their relationships.
"""

import asyncio
import hashlib
import json
import re
from typing import Any, Dict, Set, Tuple

import panel as pn
from panelini import Panelini
from panelini.panels.visnetwork import VisNetwork

from oold_agent3 import MultiStepAgent
from schema_catalog import get_cached_inventory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys handled specially or internal -- never turned into graph edges
SKIP_KEYS: Set[str] = {"uuid", "type", "meta", "rdf_type"}

# Display / metadata keys -- already represented in the node label or too
# complex (list-of-dicts) to show as simple edges
DISPLAY_KEYS: Set[str] = {
    "name",
    "label",
    "short_name",
    "description",
    "query_label",
    "iri",
    "image",
    "attachments",
}

# Whether to create nodes for literal (non-IRI) property values
SHOW_LITERAL_NODES: bool = False

# Pattern that matches OSW IRIs used as range-property values
IRI_PATTERN = re.compile(r"^(Item|Category|Property|File):OSW[a-f0-9]+$")

# vis-network appearance options
VIS_OPTIONS: dict = {
    "nodes": {
        "shape": "dot",
        "size": 20,
        "font": {"size": 14, "strokeWidth": 3, "strokeColor": "#ffffff"},
        "borderWidth": 2,
        "shadow": True,
    },
    "edges": {
        "arrows": {"to": {"enabled": True}},
        "font": {
            "size": 10,
            "align": "middle",
            "strokeWidth": 3,
            "strokeColor": "#ffffff",
        },
        "smooth": {"type": "cubicBezier"},
    },
    "physics": {
        "enabled": True,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -200,
            "centralGravity": 0.005,
            "springLength": 150,
            "springConstant": 0.05,
            "damping": 0.4,
            "avoidOverlap": 1,
        },
        "stabilization": {"enabled": True, "iterations": 200, "updateInterval": 25},
    },
    "interaction": {"hover": True, "tooltipDelay": 200},
}


# ---------------------------------------------------------------------------
# GraphState -- tracks nodes/edges and computes incremental diffs
# ---------------------------------------------------------------------------


class GraphState:
    """Tracks the current graph and computes diffs for incremental updates."""

    def __init__(self) -> None:
        self.current_nodes: Dict[str, dict] = {}
        self.current_edges: Dict[Tuple[str, str, str], dict] = {}

    # ---- public API -------------------------------------------------------

    def update(
        self,
        new_nodes: Dict[str, dict],
        new_edges: Dict[Tuple[str, str, str], dict],
    ) -> dict:
        """Compute the diff between the current and new graph state.

        Returns a dict suitable for ``VisNetwork.execute_step()``.
        """
        actions: list = []
        created_node_ids: list = []
        updated_node_ids: list = []

        # -- nodes ----------------------------------------------------------
        for nid, node in new_nodes.items():
            if nid not in self.current_nodes:
                actions.append({"action": "addNode", **node})
                created_node_ids.append(nid)
            else:
                old = self.current_nodes[nid]
                if (
                    node.get("label") != old.get("label")
                    or node.get("type") != old.get("type")
                ):
                    actions.append({"action": "updateNode", **node})
                    updated_node_ids.append(nid)

        # -- edges ----------------------------------------------------------
        new_edge_keys: list = []
        for edge_key, edge in new_edges.items():
            if edge_key not in self.current_edges:
                actions.append({"action": "addEdge", **edge})
                new_edge_keys.append(edge_key)

        # -- state updates --------------------------------------------------
        modified_node_ids = created_node_ids + updated_node_ids
        stored_ids = [
            nid
            for nid in new_nodes
            if nid not in modified_node_ids and nid in self.current_nodes
        ]
        if stored_ids:
            actions.append(
                {"action": "updateNodeState", "nodeIds": stored_ids, "state": "stored"}
            )
        if modified_node_ids:
            actions.append(
                {
                    "action": "updateNodeState",
                    "nodeIds": modified_node_ids,
                    "state": "modified",
                }
            )

        # -- persist ---------------------------------------------------------
        self.current_nodes.update(new_nodes)
        self.current_edges.update(new_edges)

        return {
            "actions": actions,
            "created_nodes": created_node_ids,
            "updated_nodes": updated_node_ids,
            "created_edges": new_edge_keys,
            "stored_nodes": stored_ids,
        }

    def clear(self) -> None:
        self.current_nodes.clear()
        self.current_edges.clear()


# ---------------------------------------------------------------------------
# Graph building from agent entities
# ---------------------------------------------------------------------------


def _literal_node_id(entity_id: str, key: str, value: str) -> str:
    """Deterministic ID for a literal-value node."""
    raw = f"{entity_id}:{key}:{value}"
    return "lit_" + hashlib.md5(raw.encode()).hexdigest()[:16]


def _extract_name(data: dict, fallback: str) -> str:
    """Extract a human-readable name from entity JSON data."""
    raw = data.get("name")
    if raw is None:
        labels = data.get("label", [])
        if labels and isinstance(labels, list):
            first = labels[0]
            if isinstance(first, dict):
                return first.get("text", fallback)
            return str(first)
        return fallback
    if isinstance(raw, list):
        return str(raw[0]) if raw else fallback
    return str(raw)


def _process_single_value(
    entity_id: str,
    key: str,
    value: Any,
    nodes: Dict[str, dict],
    edges: Dict[Tuple[str, str, str], dict],
    show_literals: bool = False,
) -> None:
    """Turn a single scalar/string value into nodes + edges."""
    if value is None or isinstance(value, dict):
        return

    str_val = str(value)

    if IRI_PATTERN.match(str_val):
        # Range reference -> link to existing or placeholder node
        if str_val not in nodes:
            nodes[str_val] = {"id": str_val, "label": str_val, "type": "instance"}
        edge_key = (entity_id, str_val, key)
        edges[edge_key] = {"from": entity_id, "to": str_val, "label": key}
    elif show_literals:
        # Literal value -> leaf node (only when enabled)
        lit_id = _literal_node_id(entity_id, key, str_val)
        display = str_val if len(str_val) <= 50 else str_val[:47] + "..."
        nodes[lit_id] = {"id": lit_id, "label": display, "type": "input"}
        edge_key = (entity_id, lit_id, key)
        edges[edge_key] = {"from": entity_id, "to": lit_id, "label": key}


def build_graph_from_entities(
    entities: Dict[str, Any],
    show_literals: bool = SHOW_LITERAL_NODES,
) -> Tuple[Dict[str, dict], Dict[Tuple[str, str, str], dict]]:
    """Convert agent entities into flat node/edge dicts for the graph.

    Args:
        entities: IRI -> OswBaseModel mapping from the agent.
        show_literals: If ``True``, create leaf nodes for literal property
            values.  Defaults to ``SHOW_LITERAL_NODES`` (``False``).

    Returns ``(nodes, edges)`` where *nodes* is keyed by node-id and
    *edges* is keyed by ``(from, to, label)`` tuples.
    """
    nodes: Dict[str, dict] = {}
    edges: Dict[Tuple[str, str, str], dict] = {}
    inventory = get_cached_inventory()

    for iri, entity in entities.items():
        data = json.loads(entity.json(exclude_none=True))
        display_name = _extract_name(data, iri)

        # -- entity instance node -------------------------------------------
        nodes[iri] = {"id": iri, "label": display_name, "type": "instance"}

        # -- "type" -> class node + edge ------------------------------------
        type_list = data.get("type", [])
        if isinstance(type_list, str):
            type_list = [type_list]
        for type_id in type_list:
            class_label = type_id  # fallback
            schema_info = inventory.get_by_schema_id(type_id)
            if schema_info is not None:
                class_label = schema_info.full_path.split(".")[-1]

            if type_id not in nodes:
                nodes[type_id] = {
                    "id": type_id,
                    "label": class_label,
                    "type": "class",
                }
            edge_key = (iri, type_id, "type")
            edges[edge_key] = {"from": iri, "to": type_id, "label": "type"}

        # -- other properties -----------------------------------------------
        for key, value in data.items():
            if key in SKIP_KEYS or key in DISPLAY_KEYS:
                continue

            if value is None:
                continue

            if isinstance(value, list):
                for item in value:
                    _process_single_value(iri, key, item, nodes, edges, show_literals)
            elif isinstance(value, dict):
                # skip objects for now
                continue
            else:
                _process_single_value(iri, key, value, nodes, edges, show_literals)

    return nodes, edges


# ---------------------------------------------------------------------------
# Chat callback
# ---------------------------------------------------------------------------


async def get_response(
    contents: str, user: str, instance: pn.chat.ChatInterface
):
    """Callback invoked by the ChatInterface when the user sends a message."""
    # Run the synchronous agent in a background thread
    try:
        entities = await asyncio.to_thread(agent.invoke, contents)
    except Exception as exc:
        return pn.chat.ChatMessage(
            f"Error: {exc}",
            user="System",
            show_timestamp=False,
            show_reaction_icons=False,
        )

    if not entities:
        return pn.chat.ChatMessage(
            "No entities could be extracted from your description.",
            user="Assistant",
            show_timestamp=False,
            show_reaction_icons=False,
        )

    # Build graph and compute incremental diff
    new_nodes, new_edges = build_graph_from_entities(entities)
    diff = graph_state.update(new_nodes, new_edges)

    if diff["actions"]:
        vis.execute_step({"actions": diff["actions"]})

    # Persist entities so the next invocation can deduplicate
    agent.store_entities()

    # Build a summary for the chat
    lines = []
    for iri, entity in entities.items():
        data = json.loads(entity.json(exclude_none=True))
        name = _extract_name(data, iri)
        type_list = data.get("type", [])
        type_str = type_list[0] if isinstance(type_list, list) and type_list else ""
        # resolve type to class name
        schema_info = get_cached_inventory().get_by_schema_id(type_str)
        if schema_info:
            type_str = schema_info.full_path.split(".")[-1]
        lines.append(f"- **{name}** ({type_str})")

    # Append graph diff summary
    n_created = len(diff["created_nodes"])
    n_updated = len(diff["updated_nodes"])
    n_stored = len(diff["stored_nodes"])
    n_edges = len(diff["created_edges"])
    diff_line = (
        f"Graph diff: Created {n_created} nodes, "
        f"Modified {n_updated} nodes, "
        f"{n_stored} unchanged, "
        f"{n_edges} new edges"
    )

    summary = (
        f"Created {len(entities)} entities:\n\n"
        + "\n".join(lines)
        + f"\n\n{diff_line}"
    )

    return pn.chat.ChatMessage(
        summary,
        user="Assistant",
        show_timestamp=False,
        show_reaction_icons=False,
    )


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

pn.extension()

agent = MultiStepAgent()
graph_state = GraphState()

vis = VisNetwork(nodes=[], edges=[], options=VIS_OPTIONS)

chat_interface = pn.chat.ChatInterface(
    callback=get_response,
    user="User",
    min_width=330,
    show_send=True,
    show_rerun=False,
    show_undo=False,
    show_timestamp=False,
    show_button_name=False,
    show_reaction_icons=False,
    callback_exception="verbose",
)

main_layout = pn.Row(
    pn.Column(
        chat_interface,
        sizing_mode="stretch_both",
        min_width=400,
    ),
    pn.Column(
        pn.pane.Markdown("### Entity Graph"),
        vis,
        sizing_mode="stretch_both",
        min_width=500,
    ),
    sizing_mode="stretch_both",
)

app = Panelini(
    title="LLM4ELN - Entity Graph Builder",
    sidebar_enabled=False,
)
app.main_set(objects=[main_layout])

servable = app.servable()

if __name__ == "__main__":
    pn.serve(servable, port=5011)
