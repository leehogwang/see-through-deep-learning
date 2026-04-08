# Repository Guidelines

## Scope and Source of Truth
- The active product lives in `dl-visualizer/`. Use the root `README.md` for overview only; use `dl-visualizer/package.json`, `dl-visualizer/server.js`, and `benchmarks/manifest.yaml` for executable truth.
- `benchmarks/` is part of the working repo, not just sample data. `manifest.yaml` defines benchmark repos, sample previews, and a `recipe_local` entry that points to `/home/students/cs/202421012/food_weight_estimation/VIF2/recipe_adamine` outside this repository.
- Treat `dl-visualizer/dist/`, `dl-visualizer/node_modules/`, `benchmarks/vendor/`, `benchmarks/reports/screenshots/`, `.codex/`, and `**/__pycache__/` as generated or disposable. Benchmark scripts also rewrite files under `benchmarks/reports/`; regenerate them instead of hand-editing outputs.

## Commands
Run project commands from `dl-visualizer/`.

- `npm install`
- `npm run dev` for the Vite client
- `npm start` for the Express server and local API on port `5173`
- `npm run build` runs `tsc -b && vite build`
- `npm run lint` runs `eslint .`
- `npm run test:edit-flow`, `npm run test:agent-panel`, `npm run test:agent-matrix`, `npm run test:canvas-e2e`, and `npm run test:palette-ui-insertions` are focused verification scripts
- `npm run verify:service` chains the main service/E2E checks
- `npm run benchmarks:fetch`, `npm run benchmarks:run`, and `npm run benchmarks:ui` maintain benchmark repos and reports
- `python3 parse_model.py scan <dir>` and `python3 parse_model.py parse <file>` are the direct parser entrypoints

## Architecture Notes That Matter
- Frontend entry is `dl-visualizer/src/main.tsx` -> `dl-visualizer/src/App.tsx`. `App.tsx` loads the dynamic torch block catalog from `/api/block-catalog` before wiring `AgentPanel`, `FlowCanvas`, `BlockPalette`, and `ScanModal`.
- `dl-visualizer/src/lib/modelToGraph.ts` is the main bridge from parsed model data to React Flow nodes/edges; layout then goes through `src/lib/layoutGraph.ts`.
- `dl-visualizer/server.js` is the backend entrypoint. It serves `dist`, reads `benchmarks/manifest.yaml`, exposes the `/api/*` surface, and owns save/merge behavior.
- Python parsing and trace helpers are spawned from `server.js`. Runtime trace endpoints depend on Bubblewrap (`bwrap`), so missing `bwrap` is an environment/setup issue to check first.
- Save/edit flows never write directly into the target repo. `server.js` creates disposable git worktrees under `/tmp/`, commits there, and merges through `/api/merge-worktree`.
- Agent edit flows depend on the official `codex` CLI transport plus local Codex auth; `server.js` explicitly disables API fallback.

## Working Conventions
- `tsconfig.app.json` enables `strict`, `noUnusedLocals`, and `noUnusedParameters`; remove dead code instead of leaving unused placeholders behind.
- Match the existing frontend style: 2-space indentation, single quotes, and semicolon-free TS/TSX.
- `vite.config.ts` proxies `/api` to `http://localhost:5173`, so API-backed frontend work expects the Express server to be running there.
- For non-trivial UI/server changes, use `npm run lint`, `npm run build`, then the smallest relevant `test:*` script. Use `npm run verify:service` before treating broad flow changes as done.
