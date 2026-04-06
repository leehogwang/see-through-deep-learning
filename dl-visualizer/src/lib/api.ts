const BASE = ''  // same origin

export interface ScannedFile {
  file: string
  relative: string
  models: string[]
}

export interface ParsedLayer {
  type: string
  nn_type: string
  params: Record<string, string | number>
  label: string
  attr: string
}

export interface ForwardTensorRef {
  token: string
  label: string
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
  params?: Record<string, string | number>
}

export interface ParsedModel {
  name: string
  layers: ParsedLayer[]
  forward_order: string[]
  forward_inputs?: ForwardTensorRef[]
  forward_graph?: ForwardOp[]
  return_outputs?: ForwardTensorRef[]
}

export interface GitInfo {
  isGit: boolean
  root?: string
  branch?: string
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
