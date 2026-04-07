import type { Node, Edge } from '@xyflow/react'
import type { ParsedModel, ParsedLayer, ForwardOp, ForwardTensorRef } from './api'
import type { LayerNodeData } from '../components/FlowCanvas/nodes/LayerNode'
import { getBlockMap, type ParamValue } from '../data/blocks'
import { layoutGraph } from './layoutGraph'

let _id = 1000
const uid = () => `parsed_${_id++}`

export const CUSTOM_MODULE_TYPE = '__custom__'

const GROUP_COLORS = [
  '#818cf8', '#34d399', '#f472b6', '#fb923c',
  '#60a5fa', '#facc15', '#a78bfa', '#4ade80',
  '#f87171', '#38bdf8', '#c084fc', '#86efac',
]
const groupColorMap = new Map<string, string>()
let groupColorIdx = 0
function groupColor(name: string): string {
  if (!groupColorMap.has(name)) {
    groupColorMap.set(name, GROUP_COLORS[groupColorIdx++ % GROUP_COLORS.length])
  }
  return groupColorMap.get(name)!
}

interface ExpandedLayer {
  layer: ParsedLayer
  blockType: string
  params: Record<string, ParamValue>
  path: string[]
}

const MAX_DEPTH = 3
const X_SPACING = 260
const Y_SPACING = 150
const START_X = 60
const START_Y = 80
const LOW_SIGNAL_OPS = new Set(['merge', 'select'])
const UTILITY_OPS = new Set(['merge', 'select', 'zip', 'stack'])

export interface GraphOptions {
  expandSubmodules?: boolean
  hideUtilityOps?: boolean
}

function resolveBlockType(layerType: string): string {
  const blockMap = getBlockMap()
  if (layerType in blockMap) return layerType
  if (layerType === 'Select') return CUSTOM_MODULE_TYPE
  if (layerType === 'Placeholder') return CUSTOM_MODULE_TYPE
  const ci = Object.keys(blockMap).find(k => k.toLowerCase() === layerType.toLowerCase())
  return ci ?? CUSTOM_MODULE_TYPE
}

function buildParams(blockType: string, layer: ParsedLayer): Record<string, ParamValue> {
  const raw = layer.params
  const p: Record<string, ParamValue> = {}

  if (blockType === CUSTOM_MODULE_TYPE) {
    p.class = layer.nn_type ?? layer.type
    Object.assign(p, raw)
    return p
  }

  switch (blockType) {
    case 'Conv2D': case 'Conv1D': case 'Conv3D':
      p.in_ch = raw.arg0 ?? raw.in_channels ?? '?'
      p.out_ch = raw.arg1 ?? raw.out_channels ?? '?'
      p.kernel = raw.arg2 ?? raw.kernel_size ?? 3
      p.stride = raw.stride ?? 1
      p.padding = raw.padding ?? 0
      break
    case 'Linear':
      p.in_features = raw.arg0 ?? raw.in_features ?? '?'
      p.out_features = raw.arg1 ?? raw.out_features ?? '?'
      break
    case 'BatchNorm2D': case 'LayerNorm': case 'GroupNorm':
      p.num_features = raw.arg0 ?? raw.num_features ?? '?'
      if (raw.num_groups) p.num_groups = raw.num_groups
      break
    case 'Dropout':
      p.p = raw.p ?? raw.arg0 ?? 0.5
      break
    case 'MultiHeadAttention':
      p.embed_dim = raw.arg0 ?? raw.embed_dim ?? '?'
      p.num_heads = raw.arg1 ?? raw.num_heads ?? '?'
      break
    case 'LSTM': case 'GRU':
      p.input_size = raw.arg0 ?? raw.input_size ?? '?'
      p.hidden_size = raw.arg1 ?? raw.hidden_size ?? '?'
      break
    case 'Embedding':
      p.num_embeddings = raw.arg0 ?? raw.num_embeddings ?? '?'
      p.embedding_dim = raw.arg1 ?? raw.embedding_dim ?? '?'
      break
    default:
      Object.assign(p, raw)
  }
  return p
}

function displayLabel(label: string): string {
  return label.replace(/_/g, ' ')
}

function paramsUseSelfState(params: Record<string, ParamValue> | undefined): boolean {
  return Object.values(params ?? {}).some((value) => typeof value === 'string' && value.includes('self.'))
}

function makeNode(
  x: number,
  y: number,
  data: LayerNodeData,
): Node {
  return {
    id: uid(),
    type: 'layerNode',
    position: { x, y },
    data,
  }
}

function makeDataNode(label: string, x: number, y: number): Node {
  const display = displayLabel(label)
  return makeNode(x, y, {
    blockType: 'Input',
    params: { data: display },
    outputShape: '',
    shapeError: false,
    _attrName: display,
    _customClassName: 'Tensor',
    _groupName: undefined,
    _groupColor: '#22c55e',
    _isTopLevel: true,
    _isUtility: false,
    _expectedTerminal: false,
    _runtimeShapeLocked: false,
  } satisfies LayerNodeData)
}

function expandLayers(
  model: ParsedModel,
  registry: Record<string, ParsedModel>,
  path: string[],
  depth: number,
): ExpandedLayer[] {
  if (depth > MAX_DEPTH) return []

  const result: ExpandedLayer[] = []

  for (const layer of model.layers) {
    const blockType = resolveBlockType(layer.type)
    const subModel = registry[layer.type]

    if (subModel && subModel.layers.length > 0 && depth < MAX_DEPTH) {
      const childPath = [...path, layer.attr]
      const expanded = expandLayers(subModel, registry, childPath, depth + 1)
      result.push(...expanded)
    } else {
      result.push({
        layer,
        blockType,
        params: buildParams(blockType, layer),
        path: [...path, layer.attr],
      })
    }
  }

  return result
}

function opScore(op: ForwardOp): number {
  let score = 0
  if (op.kind === 'module') score += 100
  if (op.block_type === 'Add' || op.block_type === 'Concat') score += 10
  if (op.block_type === 'Reshape' || op.label === 'stack') score -= 10
  if (op.label === 'merge') score -= 100
  return score
}

function effectiveOutputLabel(op: ForwardOp): string {
  if (op.output.label === 'return' || op.output.label.startsWith('__tmp')) {
    return op.label
  }
  return op.output.label
}

function opKey(op: ForwardOp): string {
  return `${op.kind}::${op.output.token}`
}

function choosePreferredOps(ops: ForwardOp[]): Map<string, string> {
  const opById = new Map(ops.map(op => [op.id, op]))
  const preferred = new Map<string, string>()
  for (const op of ops) {
    const key = opKey(op)
    const current = preferred.get(key)
    if (!current || opScore(op) > opScore(opById.get(current)!)) {
      preferred.set(key, op.id)
    }
  }
  return preferred
}

function isAliasLike(op: ForwardOp, outputLabel: string, options: GraphOptions): boolean {
  if (options.hideUtilityOps !== false && !options.expandSubmodules && LOW_SIGNAL_OPS.has(op.label)) return true
  const firstInputLabel = op.inputs[0]?.label
  if (op.label === 'stack' && firstInputLabel === outputLabel) return true
  if (op.block_type === 'Reshape' && firstInputLabel === outputLabel) return true
  return false
}

function buildCanonicalForwardGraph(model: ParsedModel): {
  inputs: ForwardTensorRef[]
  ops: ForwardOp[]
  preferredOps: Map<string, string>
} | null {
  const inputs = model.forward_inputs ?? []
  const ops = model.forward_graph ?? []
  if (ops.length === 0) return null
  return {
    inputs,
    ops,
    preferredOps: choosePreferredOps(ops),
  }
}

function buildInputSignatures(inputs: ForwardTensorRef[]): Map<string, string[]> {
  const signatures = new Map<string, string[]>()
  for (const input of inputs) {
    signatures.set(input.token, [input.token])
  }
  return signatures
}

function dedupeStrings(values: string[]): string[] {
  return [...new Set(values)].sort()
}

function signatureKey(signature: string[]): string {
  return signature.join('|')
}

function average(nums: number[]): number {
  if (nums.length === 0) return 0
  return nums.reduce((sum, n) => sum + n, 0) / nums.length
}

function buildAlgorithmicForwardGraph(
  model: ParsedModel,
  registry: Record<string, ParsedModel>,
  options: GraphOptions,
): { nodes: Node[]; edges: Edge[] } | null {
  const canonical = buildCanonicalForwardGraph(model)
  if (!canonical) return null

  const { inputs, ops, preferredOps } = canonical
  const nodes: Node[] = []
  const edges: Edge[] = []

  const opById = new Map(ops.map(op => [op.id, op]))
  const tokenAlias = new Map<string, string>()
  const tokenSourceTokens = new Map<string, string[]>()
  const tokenNodeIds = new Map<string, string[]>()
  const tokenRank = new Map<string, number>()
  const tokenSignature = buildInputSignatures(inputs)
  const opPosition = new Map<string, { rank: number; lane: number }>()
  const opNodeIds = new Map<string, string>()

  const resolveToken = (token: string): string => {
    let current = token
    const seen = new Set<string>()
    while (tokenAlias.has(current) && !seen.has(current)) {
      seen.add(current)
      current = tokenAlias.get(current)!
    }
    return current
  }

  const resolveSourceTokens = (token: string, seen = new Set<string>()): string[] => {
    const current = resolveToken(token)
    if (seen.has(current)) return []
    seen.add(current)

    const directSources = tokenSourceTokens.get(current)
    if (!directSources || directSources.length === 0) return [current]

    return [...new Set(directSources.flatMap(source => resolveSourceTokens(source, seen)))]
  }

  const sourceIdsFor = (token: string): string[] => [
    ...new Set(resolveSourceTokens(token).flatMap(source => tokenNodeIds.get(source) ?? [])),
  ]
  const rankFor = (token: string): number => tokenRank.get(resolveToken(token)) ?? 0
  const signatureFor = (token: string): string[] => tokenSignature.get(resolveToken(token)) ?? []
  const positionsFor = (token: string): { rank: number; lane: number }[] =>
    resolveSourceTokens(token)
      .map(source => opPosition.get(source) ?? opPosition.get(`input:${source}`))
      .filter((pos): pos is { rank: number; lane: number } => !!pos)

  for (const input of inputs) {
    tokenRank.set(input.token, 0)
  }

  for (const op of ops) {
    const outputLabel = effectiveOutputLabel(op)
    const preferredOpId = preferredOps.get(opKey(op))
    if (preferredOpId && preferredOpId !== op.id) {
      tokenAlias.set(op.output.token, opById.get(preferredOpId)!.output.token)
      continue
    }

    const inputTokens = dedupeStrings(op.inputs.flatMap(input => resolveSourceTokens(input.token)))
    const signature = dedupeStrings(inputTokens.flatMap(signatureFor))
    const rank = inputTokens.length > 0 ? Math.max(...inputTokens.map(rankFor)) + 1 : 1
    tokenSignature.set(op.output.token, signature)
    tokenRank.set(op.output.token, rank)

    if (isAliasLike(op, outputLabel, options)) {
      tokenSourceTokens.set(op.output.token, inputTokens)
      continue
    }
  }

  const signatureOrder = new Map<string, number>()
  let signatureIdx = 0
  const laneForSignature = (signature: string[]): number => {
    const key = signatureKey(signature)
    if (!signatureOrder.has(key)) signatureOrder.set(key, signatureIdx++)
    return signatureOrder.get(key)!
  }

  const rankBuckets = new Map<number, ForwardOp[]>()
  for (const op of ops) {
    const preferredOpId = preferredOps.get(opKey(op))
    if (preferredOpId && preferredOpId !== op.id) continue
    const outputLabel = effectiveOutputLabel(op)
    if (isAliasLike(op, outputLabel, options)) continue
    const rank = tokenRank.get(op.output.token) ?? 1
    if (!rankBuckets.has(rank)) rankBuckets.set(rank, [])
    rankBuckets.get(rank)!.push(op)
  }

  for (const [index, input] of inputs.entries()) {
    const lane = laneForSignature([input.token])
    const node = makeDataNode(input.label, START_X, START_Y + lane * Y_SPACING + index * 14)
    const data = node.data as LayerNodeData
    data.outputShape = input.shape ? `(${input.shape.replace(/^\[|\]$/g, '')})` : ''
    data._runtimeShapeLocked = !!input.shape
    nodes.push(node)
    tokenNodeIds.set(input.token, [node.id])
    opPosition.set(`input:${input.token}`, { rank: 0, lane })
  }

  const sortedRanks = [...rankBuckets.keys()].sort((a, b) => a - b)
  for (const rank of sortedRanks) {
    const bucket = rankBuckets.get(rank) ?? []
    bucket.sort((a, b) => {
      const aSig = signatureFor(a.output.token)
      const bSig = signatureFor(b.output.token)
      const aLane = laneForSignature(aSig)
      const bLane = laneForSignature(bSig)
      if (aLane !== bLane) return aLane - bLane

      const aPredLanes = a.inputs.map(input => {
        return positionsFor(input.token).map(pos => pos.lane)
      }).flat()
      const bPredLanes = b.inputs.map(input => {
        return positionsFor(input.token).map(pos => pos.lane)
      }).flat()
      const baryA = average(aPredLanes)
      const baryB = average(bPredLanes)
      if (baryA !== baryB) return baryA - baryB

      const aName = a.attr_path ?? a.attr ?? a.label
      const bName = b.attr_path ?? b.attr ?? b.label
      return aName.localeCompare(bName)
    })

    for (const op of bucket) {
      const signature = signatureFor(op.output.token)
      const lane = laneForSignature(signature)
      const display = displayLabel(effectiveOutputLabel(op))
      let blockType = op.block_type ?? undefined
      const params = { ...(op.params ?? {}) }
      const moduleClass = op.module_class ?? op.label
      const subModel = registry[moduleClass]
      const isCollapsibleModule = op.kind === 'module' && !!subModel?.forward_graph?.length
      const isUtility = UTILITY_OPS.has(op.label) || UTILITY_OPS.has(display.toLowerCase())
      const usesSelfState = paramsUseSelfState(op.params) || !!op.attr_path?.includes('.')

      if (!blockType) {
        const resolved = resolveBlockType(moduleClass)
        blockType = resolved === CUSTOM_MODULE_TYPE ? CUSTOM_MODULE_TYPE : resolved
        if (blockType === CUSTOM_MODULE_TYPE) params.class = moduleClass
      }

      const groupName = signature.length <= 1
        ? `input ${displayLabel(inputs.find(input => input.token === signature[0])?.label ?? signature[0] ?? 'branch')}`
        : `fusion ${signature.length} inputs`
      const color = groupColor(groupName)

      const node = makeNode(
        START_X + rank * X_SPACING,
        START_Y + lane * Y_SPACING,
        {
          blockType,
          params,
          outputShape: op.output_shape ? `(${op.output_shape.replace(/^\[|\]$/g, '')})` : '',
          shapeError: false,
          _attrName: display,
          _customClassName: blockType === CUSTOM_MODULE_TYPE ? (op.module_class ?? op.label) : undefined,
          _groupName: groupName,
          _groupColor: color,
          _isTopLevel: false,
          _isCollapsed: isCollapsibleModule && !options.expandSubmodules,
          _subgraphSize: isCollapsibleModule ? (subModel?.forward_graph?.length ?? 0) : undefined,
          _isUtility: isUtility,
          _expectedTerminal: false,
          _attrPath: op.attr_path ?? undefined,
          _usesSelfState: usesSelfState,
          _runtimeShapeLocked: !!op.output_shape,
        } satisfies LayerNodeData,
      )
      nodes.push(node)
      opNodeIds.set(op.id, node.id)
      tokenNodeIds.set(op.output.token, [node.id])
      opPosition.set(op.output.token, { rank, lane })

      for (const input of op.inputs) {
        for (const sourceId of sourceIdsFor(input.token)) {
          edges.push({
            id: `e_${sourceId}_${node.id}_${input.token}`,
            source: sourceId,
            target: node.id,
            animated: op.block_type === 'Add' || op.block_type === 'Concat',
            style: {
              stroke: color,
              strokeWidth: op.block_type === 'Add' || op.block_type === 'Concat' ? 2.4 : 1.8,
            },
          })
        }
      }
    }
  }

  if (options.expandSubmodules) {
    const expandableOps = ops.filter(op => {
      const preferredOpId = preferredOps.get(opKey(op))
      if (preferredOpId && preferredOpId !== op.id) return false
      if (op.kind !== 'module') return false
      const moduleClass = op.module_class ?? op.label
      const sub = registry[moduleClass]
      return !!sub?.forward_graph?.length
    })

    for (const op of expandableOps) {
      const moduleClass = op.module_class ?? op.label
      const sub = registry[moduleClass]
      if (!sub?.forward_graph?.length) continue
      const parentNodeId = opNodeIds.get(op.id)
      const parentPos = opPosition.get(op.output.token)
      if (!parentNodeId || !parentPos) continue

      const sharedColor = groupColor(op.attr_path ?? op.attr ?? moduleClass)
      const subTokenMap = new Map<string, string[]>()
      const subTokenSources = new Map<string, string[]>()
      const subInputs = sub.forward_inputs ?? []
      const parentInputs = op.inputs.flatMap(input => sourceIdsFor(input.token))
      for (let i = 0; i < Math.min(subInputs.length, parentInputs.length); i++) {
        subTokenMap.set(subInputs[i].token, [parentInputs[i]])
      }

      const subSourceIdsFor = (token: string, seen = new Set<string>()): string[] => {
        if (seen.has(token)) return []
        seen.add(token)

        const directIds = subTokenMap.get(token)
        if (directIds && directIds.length > 0) return directIds

        const sourceTokens = subTokenSources.get(token)
        if (!sourceTokens || sourceTokens.length === 0) return []

        return [...new Set(sourceTokens.flatMap(source => subSourceIdsFor(source, seen)))]
      }

      let localIndex = 0
      for (const subOp of sub.forward_graph ?? []) {
        const subOutputLabel = effectiveOutputLabel(subOp)
        if (isAliasLike(subOp, subOutputLabel, options)) {
          subTokenSources.set(subOp.output.token, subOp.inputs.map(input => input.token))
          continue
        }
        const subBlockType = subOp.block_type ?? resolveBlockType(subOp.module_class ?? subOp.label)
        const isUtility = UTILITY_OPS.has(subOp.label) || UTILITY_OPS.has(subOutputLabel.toLowerCase())
        const usesSelfState = paramsUseSelfState(subOp.params) || !!subOp.attr_path?.includes('.')
        const subNodeId = uid()
        const rankOffset = localIndex + 1
        const yOffset = subOutputLabel === 'hidden'
          ? 0
          : subOutputLabel === 'calories' ? -2
          : subOutputLabel === 'mass' ? -1
          : subOutputLabel === 'fat' ? 0
          : subOutputLabel === 'carb' ? 1
          : subOutputLabel === 'protein' ? 2
          : 0

        nodes.push({
          id: subNodeId,
          type: 'layerNode',
          position: {
            x: START_X + (parentPos.rank + rankOffset) * X_SPACING,
            y: START_Y + (parentPos.lane + yOffset) * (Y_SPACING * 0.55),
          },
          data: {
            blockType: subBlockType === CUSTOM_MODULE_TYPE ? CUSTOM_MODULE_TYPE : subBlockType,
            params: subOp.params ?? {},
            outputShape: subOp.output_shape ? `(${subOp.output_shape.replace(/^\[|\]$/g, '')})` : '',
            shapeError: false,
            _attrName: displayLabel(subOutputLabel),
            _customClassName: subBlockType === CUSTOM_MODULE_TYPE ? (subOp.module_class ?? subOp.label) : undefined,
            _groupName: op.attr_path ?? op.attr ?? moduleClass,
            _groupColor: sharedColor,
            _isTopLevel: false,
            _isUtility: isUtility,
            _expectedTerminal: false,
            _attrPath: subOp.attr_path ?? undefined,
            _usesSelfState: usesSelfState,
            _runtimeShapeLocked: !!subOp.output_shape,
          } satisfies LayerNodeData,
        })
        subTokenMap.set(subOp.output.token, [subNodeId])
        localIndex += 1

        for (const input of subOp.inputs) {
          for (const sourceId of subSourceIdsFor(input.token)) {
            edges.push({
              id: `e_${sourceId}_${subNodeId}_${input.token}_${edges.length}`,
              source: sourceId,
              target: subNodeId,
              animated: true,
              style: { stroke: sharedColor, strokeWidth: 2 },
            })
          }
        }
      }

      const exitNodeIds = [...new Set((sub.return_outputs ?? []).flatMap(output => subSourceIdsFor(output.token)))]
      if (exitNodeIds.length === 0) continue

      const incomingEdges = edges.filter(edge => edge.target === parentNodeId)
      const outgoingEdges = edges.filter(edge => edge.source === parentNodeId)
      const retainedEdges = edges.filter(edge => edge.source !== parentNodeId && edge.target !== parentNodeId)
      edges.length = 0
      edges.push(...retainedEdges)

      for (const edge of outgoingEdges) {
        for (const exitNodeId of exitNodeIds) {
          edges.push({
            ...edge,
            id: `e_${exitNodeId}_${edge.target}_${edge.id}`,
            source: exitNodeId,
          })
        }
      }

      if (incomingEdges.length > 0) {
        tokenNodeIds.set(op.output.token, exitNodeIds)
      } else {
        tokenNodeIds.set(op.output.token, exitNodeIds)
      }

      const retainedNodes = nodes.filter(node => node.id !== parentNodeId)
      nodes.length = 0
      nodes.push(...retainedNodes)
    }
  }

  const protectedTerminalNodeIds = new Set(
    (model.return_outputs ?? []).flatMap(output => sourceIdsFor(output.token)),
  )

  const visibleNodes = [...nodes]
  const visibleEdges = [...edges]

  let changed = true
  while (changed) {
    changed = false
    const connectedNodeIds = new Set(visibleEdges.flatMap(edge => [edge.source, edge.target]))
    const outDegree = new Map<string, number>()
    for (const node of visibleNodes) outDegree.set(node.id, 0)
    for (const edge of visibleEdges) {
      outDegree.set(edge.source, (outDegree.get(edge.source) ?? 0) + 1)
    }

    const removableNodeIds = new Set(
      visibleNodes
        .filter(node => {
          const data = node.data as LayerNodeData
          if (data.blockType === 'Placeholder' && !connectedNodeIds.has(node.id)) return true
          if (
            data.blockType === CUSTOM_MODULE_TYPE &&
            data._attrName === 'zip' &&
            (outDegree.get(node.id) ?? 0) === 0
          ) return true
          if (
            options.hideUtilityOps !== false &&
            !!data._isUtility &&
            (outDegree.get(node.id) ?? 0) === 0 &&
            !protectedTerminalNodeIds.has(node.id)
          ) {
            return true
          }
          return false
        })
        .map(node => node.id),
    )

    if (removableNodeIds.size === 0) break
    changed = true

    for (let i = visibleNodes.length - 1; i >= 0; i--) {
      if (removableNodeIds.has(visibleNodes[i].id)) visibleNodes.splice(i, 1)
    }
    for (let i = visibleEdges.length - 1; i >= 0; i--) {
      const edge = visibleEdges[i]
      if (removableNodeIds.has(edge.source) || removableNodeIds.has(edge.target)) {
        visibleEdges.splice(i, 1)
      }
    }
  }

  const annotatedNodes = visibleNodes.map((node) => {
    const data = node.data as LayerNodeData
    return {
      ...node,
      data: {
        ...data,
        _expectedTerminal: protectedTerminalNodeIds.has(node.id),
      } satisfies LayerNodeData,
    }
  })

  return layoutGraph(annotatedNodes, visibleEdges)
}

function buildLinearGraph(
  model: ParsedModel,
  registry: Record<string, ParsedModel> = {},
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const edges: Edge[] = []

  const expanded = expandLayers(model, registry, [], 0)
  if (expanded.length === 0) return { nodes, edges }

  const spacingX = 230
  const baseY = 320

  expanded.forEach((item, i) => {
    const isTopLevel = item.path.length <= 1
    const groupName = item.path.length >= 2 ? item.path[item.path.length - 2] : undefined
    const displayAttr = item.path[item.path.length - 1]
    const gColor = groupName ? groupColor(groupName) : undefined
    const yOffset = i % 2 === 0 ? 0 : 50

    const node: Node = makeNode(START_X + i * spacingX, baseY + yOffset, {
      blockType: item.blockType,
      params: item.params,
      outputShape: '',
      shapeError: false,
      _attrName: displayAttr,
      _customClassName: item.blockType === CUSTOM_MODULE_TYPE
        ? (item.layer.nn_type ?? item.layer.type)
        : undefined,
      _groupName: groupName,
      _groupColor: gColor,
      _isTopLevel: isTopLevel,
      _isUtility: false,
      _expectedTerminal: i === expanded.length - 1,
      _attrPath: item.path.join('.'),
      _usesSelfState: false,
      _runtimeShapeLocked: false,
    } satisfies LayerNodeData)
    nodes.push(node)

    if (i > 0) {
      const prevGroupName = expanded[i - 1].path.length >= 2
        ? expanded[i - 1].path[expanded[i - 1].path.length - 2]
        : undefined
      const sameGroup = groupName === prevGroupName && groupName !== undefined
      edges.push({
        id: `e_${nodes[i - 1].id}_${node.id}`,
        source: nodes[i - 1].id,
        target: node.id,
        animated: sameGroup,
        style: {
          stroke: sameGroup ? (gColor ?? '#6366f1') : '#334155',
          strokeWidth: sameGroup ? 2 : 1.5,
          strokeDasharray: sameGroup ? undefined : '4 3',
        },
      })
    }
  })

  return layoutGraph(nodes, edges)
}

export function modelToGraph(
  model: ParsedModel,
  registry: Record<string, ParsedModel> = {},
  options: GraphOptions = {},
): { nodes: Node[]; edges: Edge[] } {
  groupColorMap.clear()
  groupColorIdx = 0

  const forwardGraph = buildAlgorithmicForwardGraph(model, registry, options)
  if (forwardGraph && forwardGraph.nodes.length > 0) return forwardGraph
  return buildLinearGraph(model, registry)
}
