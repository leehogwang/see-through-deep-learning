import type { BlockDef, ParamValue } from '../data/blocks'

const BASE = ''  // same origin

export interface ScannedFile {
  file: string
  relative: string
  models: string[]
}

export interface ParsedLayer {
  type: string
  nn_type: string
  params: Record<string, ParamValue>
  label: string
  attr: string
}

export interface ForwardTensorRef {
  token: string
  label: string
  shape?: string
}

export interface ForwardOp {
  id: string
  kind: 'module' | 'op'
  label: string
  inputs: ForwardTensorRef[]
  output: ForwardTensorRef
  block_type?: string | null
  attr?: string | null
  attr_path?: string | null
  module_class?: string | null
  params?: Record<string, ParamValue>
  output_shape?: string
}

export interface ParsedModel {
  name: string
  layers: ParsedLayer[]
  forward_order: string[]
  forward_inputs?: ForwardTensorRef[]
  forward_graph?: ForwardOp[]
  return_outputs?: ForwardTensorRef[]
}

export type TraceMode = 'runtime-exact' | 'exploratory-static' | 'unsupported'
export type ExactnessState = 'runtime_exact' | 'unsupported_import' | 'unsupported_constructor' | 'unsupported_input_resolution' | 'unsupported_runtime_behavior'

export interface GitInfo {
  isGit: boolean
  root?: string
  branch?: string
}

export interface SamplePreview {
  imageUrl?: string
  label?: string
  resolvedPath?: string
  source: 'repo' | 'fallback' | 'none'
  caption?: string
  relativePath?: string
  strategy?: 'explicit-path' | 'auto-detected' | 'fallback' | 'missing'
  datasetEvidence?: string
  width?: number
  height?: number
  mimeType?: string
}

export interface BenchmarkSampleConfig {
  repo_paths: string[]
  repo_globs?: string[]
  fallback_path?: string
  caption?: string
}

export interface BenchmarkEntry {
  id: string
  level: number
  label: string
  repo: string
  task: string
  tags: string[]
  entry_file: string
  registry_dir: string
  model_name: string
  runtime_factory?: string
  local_only?: boolean
  sample_preview: BenchmarkSampleConfig
  available?: boolean
  repoDir?: string
}

export interface LoadedModelPayload {
  model: ParsedModel
  sourceFile: string
  gitInfo: GitInfo
  registry: Record<string, ParsedModel>
  registryDir?: string
  benchmark?: BenchmarkEntry | null
  samplePreview?: SamplePreview | null
  traceMode: TraceMode
  exactness: ExactnessState
  unsupportedReason?: string
  constructorStrategy?: string
  constructorCallable?: string
  inputStrategy?: string
  inputEvidence?: Array<{ name: string; strategy: string }>
  runtimeMs?: number | null
}

export interface CodexStatus {
  transport: 'official-exec' | 'unavailable'
  binaryPath?: string
  authReady: boolean
  authMode?: string
}

export interface CodexValidationResult {
  success: boolean
  message: string
  output?: string
}

export interface CodexEditResult {
  success: boolean
  message: string
  worktreePath: string
  branch: string
  commit: string
  changedFiles: string[]
  diffSummary: string
  output?: string
}

export interface AgentGraphNode {
  id: string
  blockType: string
  label: string
  position: { x: number; y: number }
  params: Record<string, ParamValue>
  outputShape?: string
}

export interface AgentGraphEdge {
  source: string
  target: string
}

export interface AgentGraphSnapshot {
  nodes: AgentGraphNode[]
  edges: AgentGraphEdge[]
}

export interface AgentCanvasAction {
  type: 'add_node' | 'connect' | 'update_params' | 'replace_node' | 'delete_node' | 'move_node' | 'auto_layout' | 'clear_canvas'
  tempId?: string
  nodeRef?: string
  source?: string
  target?: string
  blockType?: string
  params?: Record<string, ParamValue>
  position?: { x: number; y: number }
  direction?: 'LR' | 'TB' | 'RL' | 'BT'
}

export interface AgentChatRequest {
  prompt: string
  selectedBlockType?: string | null
  graph: AgentGraphSnapshot
  availableBlocks: Array<{
    type: string
    label: string
    category: string
    defaultParams?: Record<string, ParamValue>
  }>
  loadedModel?: {
    name: string
    sourceFile: string
    traceMode: string
    exactness: string
  } | null
}

export interface AgentChatResponse {
  reply: string
  actions: AgentCanvasAction[]
}

export async function getBlockCatalog(): Promise<BlockDef[]> {
  const r = await fetch(`${BASE}/api/block-catalog`)
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function runAgentChat(payload: AgentChatRequest): Promise<AgentChatResponse> {
  const r = await fetch(`${BASE}/api/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function scanDir(dir: string): Promise<ScannedFile[]> {
  const r = await fetch(`${BASE}/api/scan?dir=${encodeURIComponent(dir)}`)
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function parseFile(file: string): Promise<{ models: ParsedModel[]; file: string }> {
  const r = await fetch(`${BASE}/api/parse?file=${encodeURIComponent(file)}`)
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function getGitInfo(dir: string): Promise<GitInfo> {
  const r = await fetch(`${BASE}/api/git-info?dir=${encodeURIComponent(dir)}`)
  return r.json()
}

export async function createWorktree(repoRoot: string, sourceFile: string) {
  const r = await fetch(`${BASE}/api/worktree`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repoRoot, sourceFile }),
  })
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json() as Promise<{ worktreePath: string; branch: string }>
}

// Returns { ModelClassName: ParsedModel } for ALL models across all .py files in dir
export async function parseDirRegistry(dir: string): Promise<Record<string, ParsedModel>> {
  const r = await fetch(`${BASE}/api/parse-dir?dir=${encodeURIComponent(dir)}`)
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function getBenchmarks(): Promise<BenchmarkEntry[]> {
  const r = await fetch(`${BASE}/api/benchmarks`)
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function loadBenchmark(id: string): Promise<LoadedModelPayload> {
  const r = await fetch(`${BASE}/api/load-benchmark?id=${encodeURIComponent(id)}`)
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function loadModel(sourceFile: string, modelName: string): Promise<LoadedModelPayload> {
  const r = await fetch(`${BASE}/api/load-model?sourceFile=${encodeURIComponent(sourceFile)}&modelName=${encodeURIComponent(modelName)}`)
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function getBenchmarkReport(): Promise<unknown> {
  const r = await fetch(`${BASE}/api/benchmark-report`)
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function getCodexStatus(): Promise<CodexStatus> {
  const r = await fetch(`${BASE}/api/codex-status`)
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function validateCodexEdit(): Promise<CodexValidationResult> {
  const r = await fetch(`${BASE}/api/codex-edit/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  })
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function applyCodexSourceEdit(
  repoRoot: string,
  sourceFile: string,
  instruction: string
): Promise<CodexEditResult> {
  const r = await fetch(`${BASE}/api/codex-edit/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repoRoot, sourceFile, instruction }),
  })
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function saveToWorktree(
  worktreePath: string,
  repoRoot: string,
  sourceFile: string,
  content: string
) {
  const r = await fetch(`${BASE}/api/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ worktreePath, repoRoot, sourceFile, content }),
  })
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

export async function mergeWorktreeToMain(
  repoRoot: string,
  worktreePath: string,
  branch: string,
): Promise<{ success: boolean; diffSummary: string }> {
  const r = await fetch(`${BASE}/api/merge-worktree`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repoRoot, worktreePath, branch }),
  })
  if (!r.ok) throw new Error((await r.json()).error)
  return r.json()
}

// ── Layer data preview ──────────────────────────────────────────────────────

/**
 * A compact visual snapshot of a tensor flowing through a specific layer.
 * The `kind` is inferred purely from tensor rank at trace time — no layer-type
 * hard-coding.
 *
 *  "spatial"  — 4-D (B,C,H,W) feature map, encoded as a grayscale channel-mosaic PNG (base64)
 *  "sequence" — 3-D (B,T,D) sequence tensor, encoded as a heatmap PNG (base64)
 *  "vector"   — 2-D (B,D) activation vector, as a raw number array (first batch)
 */
export type LayerDataPreview =
  | { kind: 'spatial';  data: string; shape: number[] }
  | { kind: 'sequence'; data: string; shape: number[] }
  | { kind: 'vector';   values: number[]; shape: number[] }

export interface TraceLayerDataResult {
  /** keyed by module attr path (e.g. "conv1", "layer1.0.conv2") */
  previews: Record<string, LayerDataPreview>
  /** keyed by forward() param name (e.g. "x", "input_ids") */
  inputPreviews: Record<string, LayerDataPreview>
  error: string | null
}

/**
 * Ask the server to run trace_layer_data.py against the loaded model.
 * Payload mirrors the runtime_trace.py payload format.
 * Never throws — returns { previews: {}, inputPreviews: {}, error } on failure.
 */
export async function traceLayerData(payload: {
  repoRoot: string
  sourceFile: string
  modelName: string
  runtimeFactory?: string | null
  task?: string
  sample?: {
    resolvedPath?: string
    width?: number
    height?: number
    mimeType?: string
    source?: string
    strategy?: string
  } | null
}): Promise<TraceLayerDataResult> {
  try {
    const r = await fetch(`${BASE}/api/trace-layer-data`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!r.ok) {
      const body = await r.json().catch(() => ({}))
      return { previews: {}, inputPreviews: {}, error: body.error ?? `HTTP ${r.status}` }
    }
    return r.json()
  } catch (cause) {
    return { previews: {}, inputPreviews: {}, error: String(cause) }
  }
}
