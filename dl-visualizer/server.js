import express from 'express'
import { execSync, execFileSync } from 'child_process'
import { fileURLToPath } from 'url'
import path from 'path'
import fs from 'fs'
import os from 'os'
import YAML from 'yaml'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const WORKSPACE_ROOT = path.resolve(__dirname, '..')
const BENCHMARKS_ROOT = path.join(WORKSPACE_ROOT, 'benchmarks')
const MANIFEST_PATH = path.join(BENCHMARKS_ROOT, 'manifest.yaml')
const REPORT_PATH = path.join(BENCHMARKS_ROOT, 'reports', 'latest.json')
const RUNTIME_TRACE_PATH = path.join(__dirname, 'runtime_trace.py')
const TORCH_BLOCK_CATALOG_PATH = path.join(__dirname, 'torch_block_catalog.py')

const app = express()
const PORT = 5173

app.use(express.json())
app.use(express.static(path.join(__dirname, 'dist')))

function parsePython(command, target) {
  return JSON.parse(execFileSync('python3', [
    path.join(__dirname, 'parse_model.py'),
    command,
    target,
  ], { timeout: 30000 }).toString())
}

let cachedTorchBlockCatalog = null

function loadTorchBlockCatalog() {
  if (cachedTorchBlockCatalog) return cachedTorchBlockCatalog
  cachedTorchBlockCatalog = JSON.parse(execFileSync('python3', [
    TORCH_BLOCK_CATALOG_PATH,
  ], {
    timeout: 30000,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 8,
  }))
  return cachedTorchBlockCatalog
}

function runSandboxedPython(scriptPath, payload, cwd) {
  const input = JSON.stringify(payload)
  const bwrapPath = execSync('command -v bwrap', {
    shell: '/bin/bash',
    encoding: 'utf8',
    timeout: 5000,
  }).trim()

  const args = [
    '--ro-bind', '/', '/',
    '--dev', '/dev',
    '--proc', '/proc',
    '--tmpfs', '/tmp',
    '--chdir', cwd,
    'python3',
    scriptPath,
  ]

  return JSON.parse(execFileSync(bwrapPath, args, {
    input,
    cwd,
    encoding: 'utf8',
    timeout: 240000,
    maxBuffer: 1024 * 1024 * 16,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
    },
  }))
}

function scanDirInternal(dir) {
  return parsePython('scan', dir)
}

function parseFileInternal(file) {
  return parsePython('parse', file)
}

function buildRegistry(dir) {
  const files = scanDirInternal(dir)
  const registry = {}

  for (const file of files) {
    try {
      const parsed = parseFileInternal(file.file)
      for (const model of (parsed.models || [])) {
        if (!(model.name in registry)) {
          registry[model.name] = model
        }
      }
    } catch {
      // Skip files the static parser cannot handle.
    }
  }

  return registry
}

function gitInfoForDir(dir) {
  try {
    const root = execSync('git rev-parse --show-toplevel', {
      cwd: dir,
      timeout: 5000,
    }).toString().trim()
    const branch = execSync('git rev-parse --abbrev-ref HEAD', {
      cwd: dir,
      timeout: 5000,
    }).toString().trim()
    return { root, branch, isGit: true }
  } catch {
    return { isGit: false }
  }
}

function loadManifest() {
  if (!fs.existsSync(MANIFEST_PATH)) {
    throw new Error(`Benchmark manifest not found: ${MANIFEST_PATH}`)
  }
  return YAML.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'))
}

function resolvePathFromWorkspace(value) {
  if (!value) return ''
  return path.isAbsolute(value) ? value : path.resolve(WORKSPACE_ROOT, value)
}

function resolveRepoDir(manifest, repoId) {
  const repo = manifest.repos?.[repoId]
  if (!repo) {
    throw new Error(`Unknown benchmark repo: ${repoId}`)
  }
  return resolvePathFromWorkspace(repo.local_dir)
}

function resolveBenchmark(id) {
  const manifest = loadManifest()
  const benchmark = (manifest.benchmarks || []).find((item) => item.id === id)
  if (!benchmark) {
    throw new Error(`Unknown benchmark: ${id}`)
  }
  const repoDir = resolveRepoDir(manifest, benchmark.repo)
  return { manifest, benchmark, repoDir }
}

function inferMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase()
  if (ext === '.png') return 'image/png'
  if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg'
  if (ext === '.gif') return 'image/gif'
  if (ext === '.webp') return 'image/webp'
  return 'application/octet-stream'
}

function readImageMetadata(filePath) {
  const buffer = fs.readFileSync(filePath)
  const mimeType = inferMimeType(filePath)

  if (mimeType === 'image/png' && buffer.length >= 24) {
    return {
      mimeType,
      width: buffer.readUInt32BE(16),
      height: buffer.readUInt32BE(20),
    }
  }

  if (mimeType === 'image/gif' && buffer.length >= 10) {
    return {
      mimeType,
      width: buffer.readUInt16LE(6),
      height: buffer.readUInt16LE(8),
    }
  }

  if (mimeType === 'image/jpeg' && buffer.length >= 4) {
    let offset = 2
    while (offset + 9 < buffer.length) {
      if (buffer[offset] !== 0xff) break
      const marker = buffer[offset + 1]
      const size = buffer.readUInt16BE(offset + 2)
      const isSOF = marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)
      if (isSOF) {
        return {
          mimeType,
          height: buffer.readUInt16BE(offset + 5),
          width: buffer.readUInt16BE(offset + 7),
        }
      }
      offset += 2 + size
    }
  }

  return { mimeType }
}

function buildRuntimeTracePayload({ repoDir, sourceFile, modelName, benchmark, samplePreview }) {
  return {
    repoRoot: repoDir,
    sourceFile,
    modelName,
    runtimeFactory: benchmark?.runtime_factory || null,
    task: benchmark?.task || '',
    sample: samplePreview?.resolvedPath ? {
      resolvedPath: samplePreview.resolvedPath,
      width: samplePreview.width,
      height: samplePreview.height,
      mimeType: samplePreview.mimeType,
      source: samplePreview.source,
      strategy: samplePreview.strategy,
    } : null,
  }
}

function traceRuntimeModel(payload) {
  return runSandboxedPython(RUNTIME_TRACE_PATH, payload, payload.repoRoot)
}

function buildSamplePreview(benchmark, repoDir, candidatePath, source, strategy) {
  const sample = benchmark.sample_preview || {}
  const metadata = readImageMetadata(candidatePath)
  const relativePath = source === 'repo'
    ? path.relative(repoDir, candidatePath)
    : path.relative(WORKSPACE_ROOT, candidatePath)

  return {
    imageUrl: `/api/sample-asset?benchmarkId=${encodeURIComponent(benchmark.id)}`,
    label: path.basename(candidatePath),
    resolvedPath: candidatePath,
    relativePath,
    source,
    strategy,
    caption: sample.caption || '',
    datasetEvidence: source === 'repo'
      ? `Resolved from the benchmark repository at ${relativePath}.`
      : `Repository sample was not available, so a curated ${benchmark.task} fallback asset is shown.`,
    ...metadata,
  }
}

function resolveSamplePreview(benchmark, repoDir) {
  const sample = benchmark.sample_preview || {}
  const repoPaths = sample.repo_paths || []

  for (const relPath of repoPaths) {
    const candidate = path.join(repoDir, relPath)
    if (fs.existsSync(candidate)) {
      return buildSamplePreview(benchmark, repoDir, candidate, 'repo', 'explicit-path')
    }
  }

  if (sample.fallback_path) {
    const fallback = resolvePathFromWorkspace(sample.fallback_path)
    if (fs.existsSync(fallback)) {
      return buildSamplePreview(benchmark, repoDir, fallback, 'fallback', 'fallback')
    }
  }

  return {
    source: 'none',
    strategy: 'missing',
    caption: sample.caption || '',
    datasetEvidence: 'Neither repository sample paths nor fallback assets were resolved for this benchmark.',
  }
}

function listBenchmarks() {
  const manifest = loadManifest()
  return (manifest.benchmarks || []).map((benchmark) => {
    const repoDir = resolveRepoDir(manifest, benchmark.repo)
    return {
      ...benchmark,
      available: fs.existsSync(repoDir),
      repoDir,
    }
  })
}

function loadModelPayload(sourceFile, modelName, benchmark = null, samplePreview = null) {
  const sourceDir = path.dirname(sourceFile)
  const gitInfo = gitInfoForDir(sourceDir)
  const repoDir = benchmark ? resolveRepoDir(loadManifest(), benchmark.repo) : (gitInfo.root || sourceDir)
  const parsed = parseFileInternal(sourceFile)
  const staticModel = (parsed.models || []).find((item) => item.name === modelName)
  if (!staticModel) {
    throw new Error(`Model ${modelName} not found in ${sourceFile}`)
  }
  const registryDir = path.join(repoDir, benchmark?.registry_dir || path.relative(repoDir, sourceDir) || '.')
  let runtimeTrace
  try {
    runtimeTrace = traceRuntimeModel(buildRuntimeTracePayload({
      repoDir,
      sourceFile,
      modelName,
      benchmark,
      samplePreview,
    }))
  } catch (error) {
    runtimeTrace = {
      success: false,
      exactness: 'unsupported_runtime_behavior',
      unsupportedReason: error.message,
    }
  }

  const runtimeModel = runtimeTrace.success ? runtimeTrace.model : staticModel
  const exactness = runtimeTrace.success ? runtimeTrace.exactness : (runtimeTrace.exactness || 'unsupported_runtime_behavior')
  const traceMode = runtimeTrace.success ? runtimeTrace.traceMode : 'exploratory-static'

  return {
    model: runtimeModel,
    sourceFile,
    gitInfo: gitInfoForDir(repoDir),
    registry: {},
    registryDir,
    benchmark,
    samplePreview,
    traceMode,
    exactness,
    unsupportedReason: runtimeTrace.success ? '' : (runtimeTrace.unsupportedReason || 'runtime tracing failed'),
    constructorStrategy: runtimeTrace.success ? runtimeTrace.constructorStrategy : '',
    constructorCallable: runtimeTrace.success ? runtimeTrace.constructorCallable : '',
    inputStrategy: runtimeTrace.success ? runtimeTrace.inputStrategy : '',
    inputEvidence: runtimeTrace.success ? runtimeTrace.inputEvidence : [],
    runtimeMs: runtimeTrace.success ? runtimeTrace.runtimeMs : null,
  }
}

function loadBenchmarkPayload(id) {
  const { benchmark, repoDir } = resolveBenchmark(id)
  if (!fs.existsSync(repoDir)) {
    throw new Error(`Benchmark repository is missing locally: ${repoDir}`)
  }

  const sourceFile = path.join(repoDir, benchmark.entry_file)
  if (!fs.existsSync(sourceFile)) {
    throw new Error(`Benchmark entry file not found: ${sourceFile}`)
  }
  const samplePreview = resolveSamplePreview(benchmark, repoDir)
  return loadModelPayload(sourceFile, benchmark.model_name, {
    ...benchmark,
    available: true,
    repoDir,
  }, samplePreview)
}

function readBenchmarkReport() {
  if (!fs.existsSync(REPORT_PATH)) {
    return { generatedAt: null, results: [] }
  }
  return JSON.parse(fs.readFileSync(REPORT_PATH, 'utf8'))
}

function detectCodexStatus() {
  let binaryPath = ''
  let authMode = ''

  try {
    binaryPath = execSync('command -v codex', {
      shell: '/bin/bash',
      encoding: 'utf8',
      timeout: 5000,
    }).trim()
  } catch {
    return {
      transport: 'unavailable',
      authReady: false,
      binaryPath: '',
      authMode: '',
    }
  }

  try {
    execFileSync(binaryPath, ['exec', '--help'], { timeout: 8000 })
  } catch {
    return {
      transport: 'unavailable',
      authReady: false,
      binaryPath,
      authMode: '',
    }
  }

  const authPath = path.join(process.env.HOME || '', '.codex', 'auth.json')
  let authReady = false
  if (fs.existsSync(authPath)) {
    try {
      const auth = JSON.parse(fs.readFileSync(authPath, 'utf8'))
      authMode = auth.auth_mode || ''
      authReady = !!authMode
    } catch {
      authReady = false
    }
  }

  return {
    transport: authReady ? 'official-exec' : 'unavailable',
    authReady,
    authMode,
    binaryPath,
  }
}

function validateCodexExec() {
  const status = detectCodexStatus()
  if (status.transport !== 'official-exec' || !status.binaryPath) {
    throw new Error('Official codex exec transport is not ready. API fallback is intentionally disabled.')
  }

  const outputPath = path.join(os.tmpdir(), `codex-validate-${Date.now()}.txt`)
  const prompt = [
    'Reply with VALIDATION_OK on the first line.',
    'On the second line, state that this run used the official codex exec CLI transport.',
    `On the third line, print the current working directory: ${WORKSPACE_ROOT}.`,
  ].join('\n')

  try {
    execFileSync(status.binaryPath, [
      'exec',
      '--skip-git-repo-check',
      '--sandbox',
      'read-only',
      '-C',
      WORKSPACE_ROOT,
      '-o',
      outputPath,
      prompt,
    ], {
      timeout: 90000,
      stdio: 'ignore',
      env: {
        ...process.env,
        NO_COLOR: '1',
      },
    })

    const output = fs.existsSync(outputPath)
      ? fs.readFileSync(outputPath, 'utf8').trim()
      : ''

    if (!output.includes('VALIDATION_OK')) {
      throw new Error(output || 'codex exec completed without the expected validation marker')
    }

    return {
      success: true,
      message: 'Official codex exec validation succeeded. No API fallback path was used.',
      output,
    }
  } finally {
    if (fs.existsSync(outputPath)) {
      fs.unlinkSync(outputPath)
    }
  }
}

function extractJsonPayload(raw) {
  const trimmed = raw.trim()
  if (!trimmed) throw new Error('agent returned an empty response')
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i)
  const candidate = fenced ? fenced[1].trim() : trimmed
  const firstBrace = candidate.indexOf('{')
  const lastBrace = candidate.lastIndexOf('}')
  const jsonText = firstBrace >= 0 && lastBrace >= 0
    ? candidate.slice(firstBrace, lastBrace + 1)
    : candidate
  return JSON.parse(jsonText)
}

function normalizeAgentActions(actions) {
  if (!Array.isArray(actions)) return []
  const validTypes = new Set(['add_node', 'connect', 'update_params', 'replace_node', 'delete_node', 'move_node', 'auto_layout', 'clear_canvas'])
  return actions
    .filter((action) => action && typeof action === 'object' && validTypes.has(action.type))
    .map((action) => ({
      type: action.type,
      ...(typeof action.tempId === 'string' ? { tempId: action.tempId } : {}),
      ...(typeof action.nodeRef === 'string' ? { nodeRef: action.nodeRef } : {}),
      ...(typeof action.source === 'string' ? { source: action.source } : {}),
      ...(typeof action.target === 'string' ? { target: action.target } : {}),
      ...(typeof action.blockType === 'string' ? { blockType: action.blockType } : {}),
      ...(action.params && typeof action.params === 'object' ? { params: action.params } : {}),
      ...(action.position && Number.isFinite(action.position.x) && Number.isFinite(action.position.y)
        ? { position: { x: action.position.x, y: action.position.y } }
        : {}),
      ...(typeof action.direction === 'string' ? { direction: action.direction } : {}),
    }))
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function parsePromptParamValue(rawValue) {
  const trimmed = String(rawValue || '').trim()
  if (!trimmed) return trimmed
  if (/^(true|false)$/i.test(trimmed)) {
    return trimmed.toLowerCase() === 'true'
  }
  if (/^-?\d+(?:\.\d+)?$/.test(trimmed)) {
    return Number(trimmed)
  }

  const tupleParts = trimmed
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)

  if (tupleParts.length > 1 && tupleParts.every((part) => /^-?\d+(?:\.\d+)?$/.test(part))) {
    return tupleParts.join(', ')
  }

  return trimmed
}

function extractPromptParams(windowText, allowedKeys) {
  if (!windowText || allowedKeys.size === 0) return {}
  const params = {}
  const regex = /([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_.-]+(?:\s*,\s*[A-Za-z0-9_.-]+)*)/g

  for (const match of windowText.matchAll(regex)) {
    const key = match[1]
    if (!allowedKeys.has(key)) continue
    params[key] = parsePromptParamValue(match[2])
  }

  return params
}

function collectPromptParamSpecs(prompt, availableBlocks) {
  const text = String(prompt || '')
  const dedup = new Set()
  const mentions = []
  const uniqueBlocks = new Map()

  for (const block of Array.isArray(availableBlocks) ? availableBlocks : []) {
    if (!block?.type || uniqueBlocks.has(block.type)) continue
    uniqueBlocks.set(block.type, block)
  }

  for (const block of uniqueBlocks.values()) {
    const allowedKeys = new Set(Object.keys(block.defaultParams || {}))
    if (allowedKeys.size === 0) continue

    const variants = [...new Set([block.type, block.label].filter(Boolean))]
    for (const variant of variants) {
      const pattern = new RegExp(escapeRegExp(variant), 'ig')
      let match
      while ((match = pattern.exec(text)) !== null) {
        const start = match.index
        const key = `${block.type}:${start}`
        if (dedup.has(key)) continue
        dedup.add(key)

        mentions.push({
          blockType: block.type,
          index: start,
          allowedKeys,
        })
      }
    }
  }

  const specs = []
  const sortedMentions = mentions.sort((left, right) => left.index - right.index)

  sortedMentions.forEach((mention, index) => {
    const nextIndex = sortedMentions[index + 1]?.index ?? text.length
    const rest = text.slice(mention.index)
    const sentenceBreak = rest.search(/[.\n]/)
    const sentenceEnd = sentenceBreak >= 0 ? mention.index + sentenceBreak : text.length
    const end = Math.min(nextIndex, sentenceEnd, mention.index + 180)
    const windowText = text.slice(mention.index, end)
    const params = extractPromptParams(windowText, mention.allowedKeys)
    if (Object.keys(params).length === 0) return
    specs.push({
      blockType: mention.blockType,
      index: mention.index,
      params,
    })
  })

  return specs
}

function inferInputShapeFromTarget(target) {
  if (!target?.blockType) return null
  const blockType = String(target.blockType).toLowerCase()
  const params = target.params && typeof target.params === 'object' ? target.params : {}
  const numericParam = (key, fallback) => {
    const value = params[key]
    if (typeof value === 'number') return value
    if (typeof value === 'string' && /^-?\d+(?:\.\d+)?$/.test(value.trim())) {
      return Number(value.trim())
    }
    return fallback
  }

  if (blockType === 'conv1d') {
    return `B, ${numericParam('in_ch', 3)}, 128`
  }

  if (['conv2d', 'depthwiseconv2d', 'dilatedconv2d', 'transposedconv2d'].includes(blockType)) {
    return `B, ${numericParam('in_ch', 3)}, 224, 224`
  }

  if (blockType === 'conv3d') {
    return `B, ${numericParam('in_ch', 3)}, 16, 64, 64`
  }

  if (['linear', 'bilinear'].includes(blockType)) {
    return `B, ${numericParam('in_features', numericParam('in1_features', 128))}`
  }

  if (['embedding', 'patchembedding'].includes(blockType)) {
    return 'B, 16'
  }

  if (['multiheadattention', 'selfattention', 'crossattention', 'flashattention', 'gqa', 'mqa', 'linearattention', 'transformerencoderlayer', 'transformerdecoderlayer'].includes(blockType)) {
    return `B, 16, ${numericParam('embed_dim', 128)}`
  }

  return null
}

function enrichAgentActions({ actions, prompt, graph, availableBlocks }) {
  const promptSpecs = collectPromptParamSpecs(prompt, availableBlocks)
  if (promptSpecs.length === 0) {
    return actions
  }

  const nextActions = actions.map((action) => ({ ...action, ...(action.params ? { params: { ...action.params } } : {}) }))
  const specsByType = new Map()

  for (const spec of promptSpecs) {
    const bucket = specsByType.get(spec.blockType) || []
    bucket.push(spec)
    specsByType.set(spec.blockType, bucket)
  }

  for (const [blockType, specs] of specsByType.entries()) {
    const candidateIndexes = []
    nextActions.forEach((action, index) => {
      if (['add_node', 'replace_node'].includes(action.type) && action.blockType === blockType) {
        candidateIndexes.push(index)
      }
    })

    specs.forEach((spec, specIndex) => {
      const candidateIndex = candidateIndexes[specIndex]
      if (candidateIndex !== undefined) {
        const action = nextActions[candidateIndex]
        action.params = {
          ...(action.params || {}),
          ...spec.params,
        }
        return
      }

      const existingMatches = (graph?.nodes || []).filter((node) => node.blockType === blockType)
      const existing = existingMatches[specIndex]
      if (!existing) return

      nextActions.push({
        type: 'update_params',
        nodeRef: existing.id,
        params: spec.params,
      })
    })
  }

  const plannedNodes = new Map()
  nextActions.forEach((action) => {
    if (action.type !== 'add_node') return
    const key = action.tempId || action.blockType
    if (!key) return
    plannedNodes.set(key, action)
  })

  const explicitInputShapeCount = (specsByType.get('Input') || []).filter((spec) => Object.prototype.hasOwnProperty.call(spec.params, 'shape')).length
  let consumedExplicitInputShapes = 0

  for (const action of nextActions) {
    if (action.type !== 'add_node' || action.blockType !== 'Input') continue

    const connect = nextActions.find((candidate) => candidate.type === 'connect' && candidate.source === action.tempId)
    if (!connect?.target) continue

    const plannedTarget = plannedNodes.get(connect.target)
    const existingTarget = (graph?.nodes || []).find((node) => node.id === connect.target)
    const inferredShape = inferInputShapeFromTarget(plannedTarget || existingTarget)
    if (!inferredShape) continue

    const hasExplicitShape = consumedExplicitInputShapes < explicitInputShapeCount
    if (hasExplicitShape) {
      consumedExplicitInputShapes += 1
      continue
    }

    action.params = {
      ...(action.params || {}),
      shape: inferredShape,
    }
  }

  return nextActions
}

function buildAgentPrompt({ prompt, selectedBlockType, graph, loadedModel, availableBlocks }) {
  const blockCatalog = Array.isArray(availableBlocks)
    ? availableBlocks
    : loadTorchBlockCatalog().map((block) => ({
      type: block.type,
      label: block.label,
      category: block.category,
      defaultParams: block.defaultParams,
    }))
  const payload = {
    userPrompt: prompt,
    selectedBlockType: selectedBlockType || null,
    loadedModel: loadedModel || null,
    currentGraph: graph,
    availableBlocks: blockCatalog,
  }

  return [
    'You are the planning brain for an interactive deep-learning graph editor.',
    'Return JSON only. Do not wrap in markdown.',
    'Produce a short Korean reply and a structured action list for the canvas.',
    'If the request is only a question, answer it and return an empty actions array.',
    'If the user asks to modify the canvas, generate actions that reuse existing nodes whenever possible.',
    'Use replace_node for swaps like ReLU -> GELU when appropriate.',
    'Use exact blockType values from availableBlocks only.',
    'When setting params, preserve the exact parameter keys shown in availableBlocks[].defaultParams.',
    'Existing nodes must be referenced by their exact node id from currentGraph.nodes[].id.',
    'New nodes may define tempId, and later actions may reference that tempId.',
    'replace_node, update_params, delete_node, and move_node must include nodeRef.',
    'connect must include both source and target.',
    'Allowed action types: add_node, connect, update_params, replace_node, delete_node, move_node, auto_layout, clear_canvas.',
    'Action schema:',
    '{"reply":"short Korean response","actions":[{"type":"add_node","tempId":"gelu1","blockType":"GELU","position":{"x":640,"y":180},"params":{"approximate":"tanh"}},{"type":"connect","source":"n1","target":"gelu1"}]}',
    '{"reply":"short Korean response","actions":[{"type":"replace_node","nodeRef":"n3","blockType":"LeakyReLU","params":{"alpha":0.2}},{"type":"move_node","nodeRef":"n1","position":{"x":120,"y":60}}]}',
    'Rules:',
    '- Use auto_layout when the user asks to arrange or align the graph.',
    '- For move_node, provide absolute x/y coordinates.',
    '- Do not invent block types.',
    '- Keep the reply concise and factual.',
    '',
    JSON.stringify(payload),
  ].join('\n')
}

function findPromptMentionedNodes(prompt, graph) {
  const loweredPrompt = String(prompt || '').toLowerCase()
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes : []

  return nodes
    .map((node) => {
      const candidates = [node.id, node.blockType, node.label]
        .filter((value) => typeof value === 'string' && value.trim())
        .map((value) => String(value).toLowerCase())

      const indexes = candidates
        .map((candidate) => loweredPrompt.indexOf(candidate))
        .filter((index) => index >= 0)

      return {
        node,
        index: indexes.length > 0 ? Math.min(...indexes) : Number.POSITIVE_INFINITY,
      }
    })
    .filter((entry) => Number.isFinite(entry.index))
    .sort((left, right) => left.index - right.index)
}

function repairAgentActions(actions, graph, prompt) {
  const mentions = findPromptMentionedNodes(prompt, graph)

  return actions.map((action) => {
    if (action.nodeRef || !['replace_node', 'update_params', 'delete_node', 'move_node'].includes(action.type)) {
      return action
    }

    if (mentions.length === 0) return action

    if (action.type === 'move_node') {
      return {
        ...action,
        nodeRef: mentions[mentions.length - 1].node.id,
      }
    }

    const candidateMentions = action.type === 'replace_node' && action.blockType
      ? mentions.filter((entry) => String(entry.node.blockType || '').toLowerCase() !== String(action.blockType).toLowerCase())
      : mentions

    if (candidateMentions.length === 0) return action

    return {
      ...action,
      nodeRef: candidateMentions[0].node.id,
    }
  })
}

function buildAgentExecArgs(binaryPath, outputPath, prompt) {
  return [
    'exec',
    '--skip-git-repo-check',
    '--ephemeral',
    '--sandbox',
    'read-only',
    '-m',
    'gpt-5.4-mini',
    '-c',
    'model_reasoning_effort="low"',
    '-c',
    'model_reasoning_summary="none"',
    '-C',
    WORKSPACE_ROOT,
    '-o',
    outputPath,
    prompt,
  ]
}

function planAgentChat({ prompt, selectedBlockType, graph, loadedModel, availableBlocks }) {
  const status = detectCodexStatus()
  if (status.transport !== 'official-exec' || !status.binaryPath) {
    throw new Error('Official codex exec transport is not ready. Agent mode does not use API fallback.')
  }
  if (!prompt || !prompt.trim()) {
    throw new Error('prompt is required')
  }

  const outputPath = path.join(os.tmpdir(), `codex-agent-${Date.now()}.json`)

  try {
    execFileSync(status.binaryPath, buildAgentExecArgs(
      status.binaryPath,
      outputPath,
      buildAgentPrompt({ prompt, selectedBlockType, graph, loadedModel, availableBlocks }),
    ), {
      timeout: 180000,
      stdio: 'ignore',
      env: {
        ...process.env,
        NO_COLOR: '1',
      },
    })

    const raw = fs.existsSync(outputPath) ? fs.readFileSync(outputPath, 'utf8') : ''
    const parsed = extractJsonPayload(raw)
    const normalizedActions = normalizeAgentActions(parsed.actions)
    const repairedActions = repairAgentActions(normalizedActions, graph, prompt)
    const enrichedActions = enrichAgentActions({
      actions: repairedActions,
      prompt,
      graph,
      availableBlocks,
    })

    return {
      reply: typeof parsed.reply === 'string' && parsed.reply.trim()
        ? parsed.reply.trim()
        : '요청을 해석했지만 적용 가능한 결과를 만들지 못했습니다.',
      actions: enrichedActions,
    }
  } finally {
    if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath)
  }
}

function createDetachedWorktree(repoRoot, prefix = 'dl-viz-edit') {
  const ts = Date.now()
  const branch = `${prefix}-${ts}`
  const worktreePath = `/tmp/${prefix}-worktree-${ts}`
  execFileSync('git', ['worktree', 'add', worktreePath, '-b', branch], {
    cwd: repoRoot,
    timeout: 15000,
  })
  return { worktreePath, branch }
}

function commitFileChange(worktreePath, relPath, message) {
  execFileSync('git', ['add', relPath], {
    cwd: worktreePath,
    timeout: 10000,
  })
  execFileSync('git', ['commit', '-m', message], {
    cwd: worktreePath,
    timeout: 20000,
  })
  return execFileSync('git', ['rev-parse', '--short', 'HEAD'], {
    cwd: worktreePath,
    timeout: 5000,
    encoding: 'utf8',
  }).trim()
}

function validateSourcePath(repoRoot, sourceFile) {
  const relPath = path.relative(repoRoot, sourceFile)
  if (!relPath || relPath.startsWith('..')) {
    throw new Error('sourceFile must be inside repoRoot')
  }
  return relPath
}

function applyCodexSourceEdit(repoRoot, sourceFile, instruction) {
  const status = detectCodexStatus()
  if (status.transport !== 'official-exec' || !status.binaryPath) {
    throw new Error('Official codex exec transport is not ready. API fallback is intentionally disabled.')
  }
  if (!instruction || !instruction.trim()) {
    throw new Error('instruction is required')
  }

  const relPath = validateSourcePath(repoRoot, sourceFile)
  const { worktreePath, branch } = createDetachedWorktree(repoRoot, 'dl-viz-codex')
  const outputPath = path.join(os.tmpdir(), `codex-edit-${Date.now()}.txt`)
  const prompt = [
    'You are editing one existing source file inside a disposable git worktree.',
    `Only modify this file: ${relPath}`,
    'Stay source-aware: preserve the original structure and edit the existing file in place.',
    'Do not create API calls or introduce network dependencies.',
    'Do not edit any other file.',
    '',
    `Task: ${instruction.trim()}`,
  ].join('\n')

  try {
    execFileSync(status.binaryPath, [
      'exec',
      '--skip-git-repo-check',
      '--dangerously-bypass-approvals-and-sandbox',
      '-C',
      worktreePath,
      '-o',
      outputPath,
      prompt,
    ], {
      timeout: 240000,
      stdio: 'ignore',
      env: {
        ...process.env,
        NO_COLOR: '1',
      },
    })

    const changedFiles = execFileSync('git', ['diff', '--name-only', '--', relPath], {
      cwd: worktreePath,
      timeout: 10000,
      encoding: 'utf8',
    }).split('\n').filter(Boolean)

    if (changedFiles.length === 0) {
      const output = fs.existsSync(outputPath) ? fs.readFileSync(outputPath, 'utf8').trim() : ''
      throw new Error(output || 'codex exec completed without modifying the target file')
    }

    const commit = commitFileChange(
      worktreePath,
      relPath,
      `dl-viz: codex edited ${path.basename(sourceFile)}`,
    )
    const diffSummary = execFileSync('git', ['show', '--stat', '--patch', '--format=medium', 'HEAD', '--', relPath], {
      cwd: worktreePath,
      timeout: 15000,
      encoding: 'utf8',
    })
    const output = fs.existsSync(outputPath) ? fs.readFileSync(outputPath, 'utf8').trim() : ''

    return {
      success: true,
      message: `Codex edited ${path.basename(sourceFile)} in disposable worktree ${branch}.`,
      worktreePath,
      branch,
      commit,
      changedFiles,
      diffSummary,
      output,
    }
  } finally {
    if (fs.existsSync(outputPath)) {
      fs.unlinkSync(outputPath)
    }
  }
}

app.get('/api/scan', (req, res) => {
  const dir = req.query.dir
  if (!dir) return res.status(400).json({ error: 'dir required' })
  if (!fs.existsSync(dir)) return res.status(404).json({ error: `Directory not found: ${dir}` })

  try {
    res.json(scanDirInternal(dir))
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/parse', (req, res) => {
  const file = req.query.file
  if (!file) return res.status(400).json({ error: 'file required' })
  if (!fs.existsSync(file)) return res.status(404).json({ error: `File not found: ${file}` })

  try {
    res.json(parseFileInternal(file))
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/parse-dir', (req, res) => {
  const dir = req.query.dir
  if (!dir) return res.status(400).json({ error: 'dir required' })
  if (!fs.existsSync(dir)) return res.status(404).json({ error: `Not found: ${dir}` })

  try {
    res.json(buildRegistry(dir))
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/git-info', (req, res) => {
  const dir = req.query.dir
  if (!dir) return res.status(400).json({ error: 'dir required' })
  res.json(gitInfoForDir(dir))
})

app.get('/api/benchmarks', (_req, res) => {
  try {
    res.json(listBenchmarks())
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/block-catalog', (_req, res) => {
  try {
    res.json(loadTorchBlockCatalog())
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.post('/api/agent/chat', (req, res) => {
  const { prompt, selectedBlockType, graph, loadedModel, availableBlocks } = req.body
  if (!prompt || !graph) {
    return res.status(400).json({ error: 'prompt and graph are required' })
  }

  try {
    res.json(planAgentChat({ prompt, selectedBlockType, graph, loadedModel, availableBlocks }))
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/load-benchmark', (req, res) => {
  const id = req.query.id
  if (!id) return res.status(400).json({ error: 'benchmark id required' })

  try {
    res.json(loadBenchmarkPayload(id))
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/load-model', (req, res) => {
  const sourceFile = req.query.sourceFile
  const modelName = req.query.modelName
  if (!sourceFile || !modelName) {
    return res.status(400).json({ error: 'sourceFile and modelName required' })
  }
  if (!fs.existsSync(sourceFile)) {
    return res.status(404).json({ error: `File not found: ${sourceFile}` })
  }

  try {
    res.json(loadModelPayload(sourceFile, modelName, null, null))
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/benchmark-report', (_req, res) => {
  try {
    res.json(readBenchmarkReport())
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/sample-asset', (req, res) => {
  const benchmarkId = req.query.benchmarkId
  if (!benchmarkId) return res.status(400).json({ error: 'benchmarkId required' })

  try {
    const { benchmark, repoDir } = resolveBenchmark(benchmarkId)
    const preview = resolveSamplePreview(benchmark, repoDir)
    if (!preview.resolvedPath || !fs.existsSync(preview.resolvedPath)) {
      return res.status(404).json({ error: 'sample preview not found' })
    }
    return res.sendFile(preview.resolvedPath)
  } catch (error) {
    return res.status(500).json({ error: error.message })
  }
})

app.get('/api/codex-status', (_req, res) => {
  try {
    res.json(detectCodexStatus())
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.post('/api/codex-edit/validate', (_req, res) => {
  try {
    res.json(validateCodexExec())
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.post('/api/codex-edit/apply', (req, res) => {
  const { repoRoot, sourceFile, instruction } = req.body
  if (!repoRoot || !sourceFile || !instruction) {
    return res.status(400).json({ error: 'repoRoot, sourceFile, instruction required' })
  }

  try {
    res.json(applyCodexSourceEdit(repoRoot, sourceFile, instruction))
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.post('/api/worktree', (req, res) => {
  const { repoRoot } = req.body
  if (!repoRoot) return res.status(400).json({ error: 'repoRoot required' })

  try {
    const { worktreePath, branch } = createDetachedWorktree(repoRoot)
    res.json({ worktreePath, branch, success: true })
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.post('/api/save', (req, res) => {
  const { worktreePath, repoRoot, sourceFile, content } = req.body
  if (!worktreePath || !sourceFile || content === undefined) {
    return res.status(400).json({ error: 'worktreePath, sourceFile, content required' })
  }

  try {
    const relPath = validateSourcePath(repoRoot, sourceFile)
    const targetPath = path.join(worktreePath, relPath)

    fs.mkdirSync(path.dirname(targetPath), { recursive: true })
    fs.writeFileSync(targetPath, content, 'utf8')

    commitFileChange(worktreePath, relPath, `dl-viz: edited ${path.basename(sourceFile)}`)

    res.json({ success: true, savedTo: targetPath })
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/read', (req, res) => {
  const file = req.query.file
  if (!file || !fs.existsSync(file)) return res.status(404).json({ error: 'not found' })
  res.json({ content: fs.readFileSync(file, 'utf8') })
})

app.get('/{*path}', (_req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'))
})

app.listen(PORT, () => {
  console.log(`\n  ➜  DL Visualizer  http://localhost:${PORT}\n`)
})
