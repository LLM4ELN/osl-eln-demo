# Knowledge Graph Animation Demo

An interactive visualization demonstrating how a knowledge graph is built incrementally from natural language experiment descriptions. The animation shows entity extraction, classification, and graph merging in real-time.

## Live Demos

| Implementation | Interactive Demo |
|----------------|------------------|
| D3.js | [View Demo](kg-d3.html) |
| Vis.js | [View Demo](kg-visjs.html) |
| Cytoscape.js | [View Demo](kg-cytoscape.html) |
| Graphviz | [View Demo](kg-graphviz.html) |

## Animation Preview

The animation demonstrates knowledge graph construction through three experiment entries:

1. **Chapter 1**: *"A tensile test experiment #1 conducted by Dr. Jane Doe, Example Lab Corp."*
2. **Chapter 2**: *"A tensile test experiment #2 conducted by Dr. John Doe, Example Lab Corp."*
3. **Chapter 3**: *"A tensile test experiment #3 conducted by Dr. John Doe, Manager of Example Lab Corp."*

### D3.js
![D3 Knowledge Graph Animation](knowledge-graph-d3.gif)

### Vis.js
![Vis.js Knowledge Graph Animation](knowledge-graph-visjs.gif)

### Cytoscape.js
![Cytoscape Knowledge Graph Animation](knowledge-graph-cytoscape.gif)

### Graphviz
![Graphviz Knowledge Graph Animation](knowledge-graph-graphviz.gif)

## Node Types

| Color | Type | Description |
|-------|------|-------------|
| Gray (#999) | Input | User text shown in quotes |
| Green (#69b3a2) | Instance | Extracted entity |
| Pink (#ff9999) | Class | Type classification |


