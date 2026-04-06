# Repository Guidelines

## Project Structure & Module Organization
This workspace currently centers on [`dl-visualizer/`](/home/students/cs/202421012/ai/dl-visualizer), a Vite + React + TypeScript app. Frontend code lives in [`dl-visualizer/src/`](/home/students/cs/202421012/ai/dl-visualizer/src): use `components/` for UI panels and canvas nodes, `lib/` for graph, code-generation, and API helpers, `data/blocks.ts` for block metadata, and `assets/` for static files. [`dl-visualizer/server.js`](/home/students/cs/202421012/ai/dl-visualizer/server.js) serves the built app and local API routes. [`dl-visualizer/parse_model.py`](/home/students/cs/202421012/ai/dl-visualizer/parse_model.py) handles Python model scanning and parsing. Treat `dist/`, `node_modules/`, `tmp_github/`, and `__pycache__/` as generated or temporary.

## Build, Test, and Development Commands
Run commands from [`dl-visualizer/`](/home/students/cs/202421012/ai/dl-visualizer).

- `npm install`: install Node dependencies.
- `npm run dev`: start the Vite development server.
- `npm run build`: run `tsc -b` and produce a production bundle in `dist/`.
- `npm run start`: serve the built app through Express on port `5173`.
- `npm run lint`: run ESLint on all `ts` and `tsx` files.
- `python3 parse_model.py scan <repo>`: list PyTorch model files in a target repository.
- `python3 parse_model.py parse <file>`: inspect one Python model file.

## Coding Style & Naming Conventions
Use TypeScript for UI logic and keep components in PascalCase files such as `FlowCanvas.tsx`. Utility modules use camelCase filenames such as `modelToGraph.ts`. Follow the existing style: 2-space indentation, single quotes, semicolons omitted, and small focused functions. Run `npm run lint` before submitting changes; ESLint is the only enforced formatter/linter currently checked into the repo.

## Testing Guidelines
There is no committed automated test suite yet. Until one is added, treat `npm run lint` and `npm run build` as required pre-merge checks, and manually smoke-test the scan, parse, and save flows through the local UI. When adding tests, keep them near the feature (`src/components/.../*.test.tsx` or `src/lib/*.test.ts`) and prefer Playwright only for end-to-end paths that span the canvas and server API.

## Commit & Pull Request Guidelines
This repository has no Git commit history yet, so start with short imperative commit subjects using a conventional prefix, for example `feat: add model registry badge` or `fix: guard missing git root`. Pull requests should state the problem, summarize the approach, list verification steps, and include screenshots or short recordings for UI changes.

## Agent-Specific Workflow
Researchers working in this repository should keep a clear log of what was changed, how it was verified, and any unresolved risks. Preserve that record in task notes, PR descriptions, or attached reports so the implementation history remains auditable.
