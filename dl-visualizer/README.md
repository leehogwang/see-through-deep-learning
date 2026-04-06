# DL Visualizer

DL Visualizer is the application layer behind See Through Deep Learning. It parses PyTorch model source code and renders the result as an interactive graph so researchers can inspect module boundaries, branching, tensor flow, and fusion logic more directly.

## Stack

- React + TypeScript + Vite for the client
- Express for the local API server
- Python parser for static model analysis
- React Flow and Dagre for graph rendering and layout

## Scripts

```bash
npm run dev
```

Starts the Vite development client.

```bash
npm start
```

Starts the local Express server.

```bash
npm run build
```

Builds the production client bundle.

```bash
npm run lint
```

Runs ESLint over the project.

## Key Files

- `parse_model.py`: extracts model structure from Python source
- `server.js`: local API entrypoint
- `src/lib/modelToGraph.ts`: converts parsed model data into graph nodes and edges
- `src/lib/layoutGraph.ts`: layered graph layout with Dagre

## Purpose

This project is designed for static inspection rather than runtime tracing. The goal is to make complex PyTorch architectures easier to read when they contain nested modules, branch merges, masking, or custom fusion paths.
