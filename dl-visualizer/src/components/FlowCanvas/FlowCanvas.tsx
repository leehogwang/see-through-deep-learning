import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState, type ReactNode } from 'react'
import {
  ReactFlow,
  applyEdgeChanges,
  applyNodeChanges,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  BackgroundVariant,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import LayerNode, { type LayerNodeData } from './nodes/LayerNode'
// InspectorDrawer는 미니맵 패널로 대체됨
import { getBlockDef, type ParamValue } from '../../data/blocks'
import { calcOutputShape, parseShape, type Shape } from '../../lib/shapeCalculator'
import { generatePyTorchCode } from '../../lib/graphToCode'
import {
  type AgentCanvasAction,
  type AgentGraphSnapshot,
  createWorktree,
  saveToWorktree,
  // mergeWorktreeToMain은 App.tsx에서 호출
  getCodexStatus,
  validateCodexEdit,
  applyCodexSourceEdit,
  parseDirRegistry,
  traceLayerData,
  type LoadedModelPayload,
} from '../../lib/api'
import { modelToGraph } from '../../lib/modelToGraph'
import { buildGraphDiagnostics, decorateNodesWithDiagnostics } from '../../lib/graphDiagnostics'
import { layoutGraphWithDirection } from '../../lib/layoutGraph'
import FlowEdge from './edges/FlowEdge'

const NODE_TYPES = { layerNode: LayerNode }
const EDGE_TYPES = { flowEdge: FlowEdge }

let nodeId = 0
const uid = () => `n${++nodeId}`

function getMaxInputs(node: Node): number {
  const data = node.data as LayerNodeData
  return getBlockDef(data.blockType)?.maxInputs ?? 1
}

function dedupeEdges(edges: Edge[]): Edge[] {
  const seen = new Set<string>()
  return edges.filter((edge) => {
    const key = `${edge.source}->${edge.target}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

function makeEdge(source: string, target: string): Edge {
  return {
    id: `e_${source}_${target}`,
    source,
    target,
    type: 'flowEdge',
    animated: false,
    style: { stroke: '#6366f1', strokeWidth: 2 },
  }
}

function getNodeCanvasSize(node: Node): { width: number; height: number } {
  const measured = node as Node & {
    measured?: { width?: number; height?: number }
    width?: number
    height?: number
  }
  const data = node.data as LayerNodeData
  const fallbackWidth = data._isCollapsed ? 220 : 180
  return {
    width: measured.measured?.width ?? measured.width ?? fallbackWidth,
    height: measured.measured?.height ?? measured.height ?? 110,
  }
}

function getNodeCenter(node: Node): { x: number; y: number } {
  const { width, height } = getNodeCanvasSize(node)
  return {
    x: node.position.x + width / 2,
    y: node.position.y + height / 2,
  }
}

function distancePointToSegment(
  point: { x: number; y: number },
  start: { x: number; y: number },
  end: { x: number; y: number },
): number {
  const dx = end.x - start.x
  const dy = end.y - start.y
  if (dx === 0 && dy === 0) {
    return Math.hypot(point.x - start.x, point.y - start.y)
  }

  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)))
  const projection = {
    x: start.x + t * dx,
    y: start.y + t * dy,
  }
  return Math.hypot(point.x - projection.x, point.y - projection.y)
}

function findInsertionEdge(nodes: Node[], edges: Edge[], point: { x: number; y: number }): Edge | null {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]))
  let closest: { edge: Edge; distance: number } | null = null

  for (const edge of edges) {
    const sourceNode = nodeMap.get(edge.source)
    const targetNode = nodeMap.get(edge.target)
    if (!sourceNode || !targetNode) continue

    const sourceCenter = getNodeCenter(sourceNode)
    const targetCenter = getNodeCenter(targetNode)
    const distance = distancePointToSegment(point, sourceCenter, targetCenter)
    if (distance > 48) continue

    if (!closest || distance < closest.distance) {
      closest = { edge, distance }
    }
  }

  return closest?.edge ?? null
}

function spliceEdgeWithNode(edges: Edge[], edgeToReplace: Edge, newNodeId: string): Edge[] {
  const remainingEdges = edges.filter((edge) => edge.id !== edgeToReplace.id)
  return dedupeEdges([
    ...remainingEdges,
    makeEdge(edgeToReplace.source, newNodeId),
    makeEdge(newNodeId, edgeToReplace.target),
  ])
}

function computeFlowReachableNodeIds(nodes: Node[], edges: Edge[]): Set<string> {
  const incoming = new Map<string, number>()
  const adjacency = new Map<string, string[]>()

  for (const node of nodes) {
    incoming.set(node.id, 0)
    adjacency.set(node.id, [])
  }

  for (const edge of edges) {
    incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1)
    adjacency.set(edge.source, [...(adjacency.get(edge.source) ?? []), edge.target])
  }

  const queue = nodes
    .filter((node) => (incoming.get(node.id) ?? 0) === 0)
    .map((node) => node.id)
  const reachable = new Set<string>(queue)

  while (queue.length > 0) {
    const current = queue.shift()
    if (!current) break
    for (const next of adjacency.get(current) ?? []) {
      if (reachable.has(next)) continue
      reachable.add(next)
      queue.push(next)
    }
  }

  return reachable
}

function parseLockedShape(outputShape?: string): Shape | null {
  if (!outputShape) return null
  const normalized = outputShape.replace(/[()[\]]/g, '').trim()
  if (!normalized) return null
  try {
    return parseShape(normalized)
  } catch {
    return null
  }
}

function applyConnectionRules(nodes: Node[], edges: Edge[], sourceId: string, targetId: string): Edge[] {
  if (!sourceId || !targetId || sourceId === targetId) return edges
  const targetNode = nodes.find((node) => node.id === targetId)
  let nextEdges = edges.filter((edge) => !(edge.source === sourceId && edge.target === targetId))

  if (targetNode && getMaxInputs(targetNode) === 1) {
    nextEdges = nextEdges.filter((edge) => edge.target !== targetId)
  }

  nextEdges.push(makeEdge(sourceId, targetId))
  return dedupeEdges(nextEdges)
}

function propagateShapes(nodes: Node[], edges: Edge[]): Node[] {
  const shapeMap: Record<string, Shape | null> = {}
  const inDegree: Record<string, number> = {}
  const adjacency: Record<string, string[]> = {}
  const nodeById = new Map(nodes.map((node) => [node.id, node]))

  nodes.forEach((node) => {
    inDegree[node.id] = 0
    adjacency[node.id] = []
  })

  edges.forEach((edge) => {
    adjacency[edge.source]?.push(edge.target)
    if (inDegree[edge.target] !== undefined) inDegree[edge.target] += 1
  })

  const queue = nodes.filter((node) => inDegree[node.id] === 0)
  const order: Node[] = []
  const seen = new Set<string>()

  while (queue.length > 0) {
    const node = queue.shift()
    if (!node) break
    if (seen.has(node.id)) continue
    seen.add(node.id)
    order.push(node)
    adjacency[node.id].forEach((nextId) => {
      inDegree[nextId] -= 1
      if (inDegree[nextId] === 0) {
        const next = nodeById.get(nextId)
        if (next) queue.push(next)
      }
    })
  }

  const remainder = nodes.filter((node) => !seen.has(node.id))
  const evaluationOrder = [...order, ...remainder]
  const evaluated = new Map<string, Node>()

  for (const node of evaluationOrder) {
    const data = node.data as LayerNodeData
    if (data._runtimeShapeLocked) {
      shapeMap[node.id] = parseLockedShape(data.outputShape)
      evaluated.set(node.id, {
        ...node,
        data: {
          ...data,
          shapeError: false,
        } satisfies LayerNodeData,
      })
      continue
    }
    const incomingEdges = edges.filter((edge) => edge.target === node.id)
    const inputShapes = incomingEdges
      .map((edge) => shapeMap[edge.source])
      .filter((shape): shape is Shape => Array.isArray(shape))
    const inputShape = inputShapes[0] ?? null
    const { shape, error, str } = calcOutputShape(data.blockType, data.params, inputShape, inputShapes)
    shapeMap[node.id] = shape

    evaluated.set(node.id, {
      ...node,
      data: {
        ...data,
        outputShape: str,
        shapeError: error && data.blockType !== 'Input',
      } satisfies LayerNodeData,
    })
  }

  return nodes.map((node) => evaluated.get(node.id) ?? node)
}

function displayNodeLabel(node: Node): string {
  const data = node.data as LayerNodeData
  return data._attrName || getBlockDef(data.blockType)?.label || data.blockType
}

function buildGraphSnapshot(nodes: Node[], edges: Edge[]): AgentGraphSnapshot {
  return {
    nodes: nodes.map((node) => {
      const data = node.data as LayerNodeData
      return {
        id: node.id,
        blockType: data.blockType,
        label: displayNodeLabel(node),
        position: node.position,
        params: { ...data.params },
        outputShape: data.outputShape,
      }
    }),
    edges: edges.map((edge) => ({ source: edge.source, target: edge.target })),
  }
}

function removeNodeFromGraph(nodes: Node[], edges: Edge[], nodeId: string): { nodes: Node[]; edges: Edge[] } {
  const incomingEdges = edges.filter((edge) => edge.target === nodeId)
  const outgoingEdges = edges.filter((edge) => edge.source === nodeId)
  const nextNodes = nodes.filter((node) => node.id !== nodeId)
  let nextEdges = edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId)

  if (incomingEdges.length === 1 && outgoingEdges.length === 1) {
    const bypassSource = incomingEdges[0].source
    const bypassTarget = outgoingEdges[0].target
    const targetNode = nextNodes.find((node) => node.id === bypassTarget)
    if (targetNode && bypassSource !== bypassTarget) {
      if (getMaxInputs(targetNode) === 1) {
        nextEdges = nextEdges.filter((edge) => edge.target !== bypassTarget)
      }
      nextEdges.push(makeEdge(bypassSource, bypassTarget))
    }
  }

  return { nodes: nextNodes, edges: nextEdges }
}

function defaultAgentPosition(nodes: Node[], index: number): { x: number; y: number } {
  const maxX = nodes.reduce((current, node) => Math.max(current, node.position.x), 40)
  return {
    x: maxX + 220,
    y: 120 + index * 90,
  }
}

function parseNumericParam(value: ParamValue | undefined, fallback: number): number {
  if (typeof value === 'number') return value
  if (typeof value === 'string') {
    const parsed = Number(value.trim())
    return Number.isFinite(parsed) ? parsed : fallback
  }
  return fallback
}

function inferRecoveryInputShape(targetNode: Node | undefined): string | null {
  if (!targetNode) return null
  const data = targetNode.data as LayerNodeData
  const params = data.params || {}

  // Algorithmic approach: probe the shape calculator with decreasing-rank
  // candidate inputs and return the first shape that doesn't produce an error.
  // This avoids maintaining any list of blockType names and works for any block
  // — including custom/future ones — as long as shapeCalculator is updated.

  // Build best-guess channel/feature sizes from whatever params exist,
  // falling back to sensible defaults when the param is absent.
  const inputCh = parseNumericParam(
    (params.in_ch ?? params.in_channels) as ParamValue | undefined,
    3,
  )
  const seqDim = parseNumericParam(
    (params.embed_dim ?? params.d_model ?? params.in_features ?? params.in_ch) as ParamValue | undefined,
    128,
  )

  // Candidates ordered from highest rank to lowest.  The candidate's `shape`
  // is passed to calcOutputShape as the simulated input; if it produces no
  // error that rank is correct for this block.
  const candidates: Array<{ shape: Shape; str: string }> = [
    // 5-D: volumetric / video convolutions (B, C, D, H, W)
    { shape: [-1, inputCh, 16, 64, 64], str: `B, ${inputCh}, 16, 64, 64` },
    // 4-D: image convolutions (B, C, H, W)
    { shape: [-1, inputCh, 224, 224],   str: `B, ${inputCh}, 224, 224` },
    // 3-D: sequences / attention (B, T, D)
    { shape: [-1, 16, seqDim],          str: `B, 16, ${seqDim}` },
    // 2-D: linear / embedding lookup (B, D)
    { shape: [-1, seqDim],              str: `B, ${seqDim}` },
  ]

  for (const { shape, str } of candidates) {
    const { error } = calcOutputShape(data.blockType, params, shape, [shape])
    if (!error) return str
  }

  // All ranks failed — caller will keep the existing Input shape unchanged.
  return null
}

function normalizeInputNodeShapes(nodes: Node[], edges: Edge[]): Node[] {
  return nodes.map((node) => {
    const data = node.data as LayerNodeData
    if (data.blockType !== 'Input') return node

    const outgoing = edges.find((edge) => edge.source === node.id)
    if (!outgoing) return node
    const targetNode = nodes.find((candidate) => candidate.id === outgoing.target)
    const nextShape = inferRecoveryInputShape(targetNode)
    if (!nextShape || data.params.shape === nextShape) return node

    return {
      ...node,
      data: {
        ...data,
        params: {
          ...data.params,
          shape: nextShape,
        },
        outputShape: `(${nextShape})`,
        _runtimeShapeLocked: false,
      } satisfies LayerNodeData,
    }
  })
}

function graphHealthFromDiagnostics(diagnostics: ReturnType<typeof buildGraphDiagnostics>) {
  const errorCount = diagnostics.filter((item) => item.severity === 'error').length
  const warningCount = diagnostics.filter((item) => item.severity === 'warning').length
  const infoCount = diagnostics.filter((item) => item.severity === 'info').length

  if (errorCount > 0) {
    return {
      level: 'error' as const,
      label: 'Needs Attention',
      message: `${errorCount} graph error${errorCount > 1 ? 's' : ''} need recovery.`,
      errorCount,
      warningCount,
      infoCount,
    }
  }

  if (warningCount > 0) {
    return {
      level: 'warning' as const,
      label: 'Warnings Present',
      message: `${warningCount} warning${warningCount > 1 ? 's' : ''} remain in the current graph.`,
      errorCount,
      warningCount,
      infoCount,
    }
  }

  return {
    level: 'healthy' as const,
    label: 'Healthy',
    message: 'The current graph is consistent with no active diagnostics.',
    errorCount,
    warningCount,
    infoCount,
  }
}

function formatTraceMode(traceMode?: string) {
  if (traceMode === 'runtime-exact') return 'Runtime Exact'
  if (traceMode === 'exploratory-static') return 'Exploratory Static'
  if (traceMode === 'unsupported') return 'Unsupported'
  return traceMode || 'Unknown'
}

function formatExactness(exactness?: string) {
  if (exactness === 'runtime_exact') return 'Runtime exact'
  if (exactness === 'unsupported_import') return 'Unsupported import'
  if (exactness === 'unsupported_constructor') return 'Unsupported constructor'
  if (exactness === 'unsupported_input_resolution') return 'Unsupported input resolution'
  if (exactness === 'unsupported_runtime_behavior') return 'Unsupported runtime behavior'
  return exactness || 'Unknown'
}

function resolveActionNodeRef(
  ref: string | undefined,
  nodes: Node[],
  aliases: Map<string, string>,
): string | null {
  if (!ref) return null
  const aliasResolved = aliases.get(ref) ?? ref
  if (nodes.some((node) => node.id === aliasResolved)) return aliasResolved
  const byLabel = nodes.find((node) => displayNodeLabel(node) === ref || (node.data as LayerNodeData).blockType === ref)
  return byLabel?.id ?? null
}

function createNodeFromBlock(
  blockType: string,
  nodes: Node[],
  paramsOverride: Record<string, ParamValue> | undefined,
  positionOverride: { x: number; y: number } | undefined,
  index: number,
  options?: { editedByAgent?: boolean },
): Node | null {
  const definition = getBlockDef(blockType)
  if (!definition) return null
  const params = { ...definition.defaultParams, ...(paramsOverride ?? {}) }
  let outputShape = ''
  if (blockType === 'Input') {
    const shapeStr = String(params.shape || definition.defaultParams.shape || 'B,3,224,224')
    try {
      parseShape(shapeStr)
      outputShape = `(${shapeStr})`
    } catch {
      outputShape = shapeStr
    }
  }
  return {
    id: uid(),
    type: 'layerNode',
    position: positionOverride ?? defaultAgentPosition(nodes, index),
    data: {
      blockType,
      params,
      outputShape,
      shapeError: false,
      _isUtility: false,
      _expectedTerminal: false,
      _runtimeShapeLocked: false,
      _editedByAgent: !!options?.editedByAgent,
    } satisfies LayerNodeData,
  }
}

function markNodeAsAgentEdited(node: Node): Node {
  return {
    ...node,
    data: {
      ...(node.data as LayerNodeData),
      _editedByAgent: true,
    } satisfies LayerNodeData,
  }
}

function markNodesAsAgentEdited(nodes: Node[], nodeIds: string[]): Node[] {
  if (nodeIds.length === 0) return nodes
  const targets = new Set(nodeIds)
  return nodes.map((node) => (targets.has(node.id) ? markNodeAsAgentEdited(node) : node))
}

function applyAgentActionsToGraph(
  actions: AgentCanvasAction[],
  initialNodes: Node[],
  initialEdges: Edge[],
): { nodes: Node[]; edges: Edge[]; applied: number; warnings: string[] } {
  let currentNodes = [...initialNodes]
  let currentEdges = [...initialEdges]
  const aliases = new Map<string, string>()
  const warnings: string[] = []
  let applied = 0

  for (const action of actions) {
    switch (action.type) {
      case 'clear_canvas': {
        currentNodes = []
        currentEdges = []
        applied += 1
        break
      }
      case 'add_node': {
        if (!action.blockType) {
          warnings.push('add_node skipped: missing blockType')
          break
        }
        const node = createNodeFromBlock(
          action.blockType,
          currentNodes,
          action.params,
          action.position,
          currentNodes.length,
          { editedByAgent: true },
        )
        if (!node) {
          warnings.push(`add_node skipped: unknown block type ${action.blockType}`)
          break
        }
        currentNodes = [...currentNodes, node]
        if (action.tempId) aliases.set(action.tempId, node.id)
        applied += 1
        break
      }
      case 'connect': {
        const sourceId = resolveActionNodeRef(action.source, currentNodes, aliases)
        const targetId = resolveActionNodeRef(action.target, currentNodes, aliases)
        if (!sourceId || !targetId) {
          warnings.push(`connect skipped: unresolved refs ${action.source ?? '?'} -> ${action.target ?? '?'}`)
          break
        }
        currentEdges = applyConnectionRules(currentNodes, currentEdges, sourceId, targetId)
        currentNodes = markNodesAsAgentEdited(currentNodes, [sourceId, targetId])
        applied += 1
        break
      }
      case 'update_params': {
        const nodeId = resolveActionNodeRef(action.nodeRef, currentNodes, aliases)
        if (!nodeId || !action.params) {
          warnings.push(`update_params skipped: unresolved node ${action.nodeRef ?? '?'}`)
          break
        }
        currentNodes = currentNodes.map((node) => {
          if (node.id !== nodeId) return node
          const data = node.data as LayerNodeData
          return {
            ...node,
            data: {
              ...data,
              params: {
                ...data.params,
                ...action.params,
              },
              _runtimeShapeLocked: false,
              _editedByAgent: true,
            } satisfies LayerNodeData,
          }
        })
        applied += 1
        break
      }
      case 'replace_node': {
        const nodeId = resolveActionNodeRef(action.nodeRef, currentNodes, aliases)
        const replacementType = action.blockType
        if (!nodeId || !replacementType) {
          warnings.push(`replace_node skipped: unresolved node ${action.nodeRef ?? '?'}`)
          break
        }
        const targetNode = currentNodes.find((node) => node.id === nodeId)
        const definition = getBlockDef(replacementType)
        if (!targetNode || !definition) {
          warnings.push(`replace_node skipped: unknown block ${replacementType}`)
          break
        }
        currentNodes = currentNodes.map((node) => {
          if (node.id !== nodeId) return node
          const previousData = node.data as LayerNodeData
          const mergedParams = {
            ...definition.defaultParams,
            ...(action.params ?? {}),
          }
          let outputShape = previousData.outputShape ?? ''
          if (replacementType === 'Input') {
            const shapeStr = String(mergedParams.shape || definition.defaultParams.shape || 'B,3,224,224')
            outputShape = `(${shapeStr})`
          }
          return {
            ...node,
            position: action.position ?? node.position,
            data: {
              ...previousData,
              blockType: replacementType,
              params: mergedParams,
              outputShape,
              shapeError: false,
              _runtimeShapeLocked: false,
              _editedByAgent: true,
              // Clear model-parse metadata that belongs to the old node identity
              _attrName: undefined,
              _attrPath: undefined,
              _customClassName: undefined,
              _groupName: undefined,
              _groupColor: undefined,
              _isTopLevel: undefined,
              _isCollapsed: undefined,
              _subgraphSize: undefined,
              _isUtility: undefined,
              _expectedTerminal: undefined,
              _usesSelfState: undefined,
              _dataPreview: undefined,
            } satisfies LayerNodeData,
          }
        })
        applied += 1
        break
      }
      case 'delete_node': {
        const nodeId = resolveActionNodeRef(action.nodeRef, currentNodes, aliases)
        if (!nodeId) {
          warnings.push(`delete_node skipped: unresolved node ${action.nodeRef ?? '?'}`)
          break
        }
        const next = removeNodeFromGraph(currentNodes, currentEdges, nodeId)
        currentNodes = next.nodes
        currentEdges = next.edges
        applied += 1
        break
      }
      case 'move_node': {
        const nodeId = resolveActionNodeRef(action.nodeRef, currentNodes, aliases)
        if (!nodeId || !action.position) {
          warnings.push(`move_node skipped: unresolved node ${action.nodeRef ?? '?'}`)
          break
        }
        currentNodes = currentNodes.map((node) => node.id === nodeId
          ? markNodeAsAgentEdited({ ...node, position: action.position! })
          : node)
        applied += 1
        break
      }
      case 'auto_layout': {
        const laidOut = layoutGraphWithDirection(currentNodes, currentEdges, action.direction ?? 'LR')
        currentNodes = laidOut.nodes.map(markNodeAsAgentEdited)
        currentEdges = laidOut.edges
        applied += 1
        break
      }
      default:
        warnings.push(`unsupported action ${(action as { type?: string }).type ?? 'unknown'}`)
    }
  }

  return { nodes: currentNodes, edges: currentEdges, applied, warnings }
}

export interface FlowCanvasHandle {
  executeAgentActions: (actions: AgentCanvasAction[]) => Promise<{ applied: number; warnings: string[]; snapshot: AgentGraphSnapshot }>
}

interface Props {
  catalogVersion?: number
  onNodeSelect: (blockType: string | null) => void
  loadedModel?: LoadedModelPayload | null
  onWorktreeSaved?: (path: string, branch: string) => void
  onGraphSnapshotChange?: (snapshot: AgentGraphSnapshot) => void
  onMergeRequested?: (worktreePath: string, branch: string) => void
}

const FlowCanvas = forwardRef<FlowCanvasHandle, Props>(function FlowCanvas(
  { catalogVersion = 0, onNodeSelect, loadedModel, onWorktreeSaved, onGraphSnapshotChange, onMergeRequested }: Props,
  ref,
) {
  const [nodes, setNodes] = useNodesState<Node>([])
  const [edges, setEdges] = useEdgesState<Edge>([])
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const nodesRef = useRef<Node[]>([])
  const edgesRef = useRef<Edge[]>([])
  const [_showCode, setShowCode] = useState(false)
  void _showCode
  const [_code, setCode] = useState('')
  void _code
  const [savingWorktree, setSavingWorktree] = useState(false)
  const [worktreeInfo, setWorktreeInfo] = useState<{ path: string; branch: string } | null>(null)
  const [autoSaveStatus, setAutoSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const autoSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const retraceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const worktreeInfoRef = useRef<{ path: string; branch: string } | null>(null)
  const [expandModules, setExpandModules] = useState(false)
  const [hideUtilityOps, setHideUtilityOps] = useState(true)
  const [drawerOpen, setDrawerOpen] = useState(true)
  // drawerTab은 Diagnostics 전용으로 고정 — Sample Data는 플로팅 패널로 이동
  const [_drawerTab, setDrawerTab] = useState<'sample' | 'diagnostics'>('diagnostics')
  void _drawerTab
  const [_hideExpectedTerminals, _setHideExpectedTerminals] = useState(true)
  void _hideExpectedTerminals
  const [pendingConnectionSourceId, setPendingConnectionSourceId] = useState<string | null>(null)
  const [_codexStatus, setCodexStatus] = useState<Awaited<ReturnType<typeof getCodexStatus>> | null>(null)
  void _codexStatus
  const [_codexValidation, setCodexValidation] = useState<{
    status: 'idle' | 'running' | 'done' | 'error'
    message: string
  }>({ status: 'idle', message: '' })
  void _codexValidation
  const [codexInstruction, setCodexInstruction] = useState('')
  const [_codexEdit, setCodexEdit] = useState<{
    status: 'idle' | 'running' | 'done' | 'error'
    message: string
    diffSummary?: string
  }>({ status: 'idle', message: '' })
  void _codexEdit
  const [surfaceStatus, setSurfaceStatus] = useState<{
    level: 'idle' | 'success' | 'error'
    message: string
  }>({ level: 'idle', message: '' })
  const [resolvedRegistry, setResolvedRegistry] = useState<Record<string, LoadedModelPayload['model']>>({})
  // Track whether the agent has made edits since the last model load.
  // When true, secondary effect triggers (registry load, catalog version change)
  // must NOT overwrite the graph with the original parsed model.
  const agentDirtyRef = useRef(false)
  const loadedModelSourceRef = useRef<string | null>(null)
  const loadedModelRef = useRef<LoadedModelPayload | null | undefined>(null)
  loadedModelRef.current = loadedModel
  const rfInstanceRef = useRef<ReturnType<typeof useReactFlow> | null>(null)

  const commitGraph = useCallback((nextNodes: Node[], nextEdges: Edge[]) => {
    const dedupedEdges = dedupeEdges(nextEdges)
    const propagatedNodes = propagateShapes(nextNodes, dedupedEdges)
    nodesRef.current = propagatedNodes
    edgesRef.current = dedupedEdges
    setNodes(propagatedNodes)
    setEdges(dedupedEdges)

    // 워크트리가 있으면 디바운스(1.5초) 후 자동 저장
    if (worktreeInfoRef.current && loadedModel?.gitInfo.root && loadedModel?.sourceFile) {
      if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current)
      setAutoSaveStatus('saving')
      autoSaveTimerRef.current = setTimeout(async () => {
        const wt = worktreeInfoRef.current
        if (!wt || !loadedModel?.gitInfo.root || !loadedModel?.sourceFile) return
        try {
          const generated = generatePyTorchCode(propagatedNodes, dedupedEdges)
          await saveToWorktree(wt.path, loadedModel.gitInfo.root, loadedModel.sourceFile, generated)
          setAutoSaveStatus('saved')
        } catch (_) {
          setAutoSaveStatus('error')
        }
      }, 1500)
    }
  }, [setEdges, setNodes, loadedModel])

  const handleNodesChange = useCallback((changes: NodeChange<Node>[]) => {
    const nextNodes = applyNodeChanges(changes, nodesRef.current)
    commitGraph(nextNodes, edgesRef.current)
  }, [commitGraph])

  const handleEdgesChange = useCallback((changes: EdgeChange<Edge>[]) => {
    const nextEdges = applyEdgeChanges(changes, edgesRef.current)
    commitGraph(nodesRef.current, nextEdges)
  }, [commitGraph])

  useEffect(() => {
    nodesRef.current = nodes
  }, [nodes])

  useEffect(() => {
    edgesRef.current = edges
  }, [edges])

  useEffect(() => {
    onGraphSnapshotChange?.(buildGraphSnapshot(nodes, edges))
  }, [edges, nodes, onGraphSnapshotChange])

  useImperativeHandle(ref, () => ({
    async executeAgentActions(actions) {
      const result = applyAgentActionsToGraph(actions, nodesRef.current, edgesRef.current)
      commitGraph(result.nodes, result.edges)
      if (result.applied > 0) {
        agentDirtyRef.current = true
      }
      return {
        applied: result.applied,
        warnings: result.warnings,
        snapshot: buildGraphSnapshot(propagateShapes(result.nodes, dedupeEdges(result.edges)), dedupeEdges(result.edges)),
      }
    },
  }), [commitGraph])

  useEffect(() => {
    setResolvedRegistry(loadedModel?.registry ?? {})
  }, [loadedModel])

  // worktreeInfo ref \ub3d9\uae30\ud654 (\ucf5c\ubc31 \ud074\ub85c\uc800\uc5d0\uc11c \uc0ac\uc6a9\ud558\uae30 \uc704\ud574)\n  useEffect(() => {\n    worktreeInfoRef.current = worktreeInfo\n  }, [worktreeInfo])

  // \ubaa8\ub378\uc744 \uc5ec\ub294 \uc21c\uac04 \uc790\ub3d9\uc73c\ub85c copy \uc6cc\ud06c\ud2b8\ub9ac \uc0dd\uc131
  useEffect(() => {
    if (!loadedModel?.gitInfo.isGit || !loadedModel?.gitInfo.root || !loadedModel?.sourceFile) return
    // \uc774\uc804 \uc6cc\ud06c\ud2b8\ub9ac\uac00 \uc788\uc73c\uba74 \uc7ac\uc0ac\uc6a9
    if (worktreeInfoRef.current) return

    let cancelled = false
    createWorktree(loadedModel.gitInfo.root, loadedModel.sourceFile)
      .then((wt) => {
        if (cancelled) return
        setWorktreeInfo({ path: wt.worktreePath, branch: wt.branch })
        onWorktreeSaved?.(wt.worktreePath, wt.branch)
      })
      .catch((e) => {
        if (cancelled) return
        console.warn('[dl-viz] \uc6cc\ud06c\ud2b8\ub9ac \uc790\ub3d9 \uc0dd\uc131 \uc2e4\ud328:', e)
      })
    return () => { cancelled = true }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadedModel?.sourceFile])

  useEffect(() => {
    if (!loadedModel || !expandModules) return
    if (Object.keys(resolvedRegistry).length > 0) return
    if (!loadedModel.registryDir) return

    let cancelled = false
    parseDirRegistry(loadedModel.registryDir)
      .then((registry) => {
        if (!cancelled) setResolvedRegistry(registry)
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [expandModules, loadedModel, resolvedRegistry])

  useEffect(() => {
    if (!loadedModel) return

    // If the loaded model source file changed, reset dirty flag so the new
    // model is always rendered fresh.
    const newSource = loadedModel.sourceFile ?? null
    if (newSource !== loadedModelSourceRef.current) {
      loadedModelSourceRef.current = newSource
      agentDirtyRef.current = false
    }

    // Secondary effect triggers (registry load, catalog version change, toggle
    // changes) must not overwrite agent edits made to the current model.
    if (agentDirtyRef.current) return

    const graph = modelToGraph(
      loadedModel.model,
      resolvedRegistry,
      {
        expandSubmodules: expandModules,
        hideUtilityOps,
      },
    )
    commitGraph(graph.nodes, graph.edges)
    setDrawerTab('sample')
    setPendingConnectionSourceId(null)
    setCodexInstruction('')
    setCodexEdit({ status: 'idle', message: '', diffSummary: '' })
    setSurfaceStatus({ level: 'idle', message: '' })
  }, [catalogVersion, commitGraph, expandModules, hideUtilityOps, loadedModel, resolvedRegistry])

  // ── Data flow trace: run after model loads, update each node's _dataPreview ──
  useEffect(() => {
    if (!loadedModel?.sourceFile || !loadedModel.model?.name) return

    const gitRoot = loadedModel.gitInfo?.root
    const repoRoot = gitRoot ?? loadedModel.sourceFile.replace(/\/[^/]+$/, '')
    const payload = {
      repoRoot,
      sourceFile: loadedModel.sourceFile,
      modelName: loadedModel.model.name,
      runtimeFactory: loadedModel.constructorCallable ?? null,
      task: loadedModel.benchmark?.task ?? '',
      sample: loadedModel.samplePreview?.resolvedPath ? {
        resolvedPath: loadedModel.samplePreview.resolvedPath,
        width: loadedModel.samplePreview.width,
        height: loadedModel.samplePreview.height,
        mimeType: loadedModel.samplePreview.mimeType,
        source: loadedModel.samplePreview.source,
        strategy: loadedModel.samplePreview.strategy,
      } : null,
    }

    let cancelled = false
    traceLayerData(payload).then((result) => {
      if (cancelled) return
      if (result.error && Object.keys(result.previews).length === 0 && Object.keys(result.inputPreviews).length === 0) return

      // Pre-compute ordered input entries so multi-input models can match positionally
      const inputEntries = Object.entries(result.inputPreviews)

      setNodes((prevNodes) => {
        // Collect all input-like nodes (no incoming edges) so each gets its own
        // input tensor positionally — works for any number of inputs and any
        // block type used as the graph entry point, not just blockType === 'Input'.
        const currentEdges = edgesRef.current
        const inputNodeIds = new Set(
          prevNodes
            .filter((n) => !currentEdges.some((e) => e.target === n.id))
            .map((n) => n.id),
        )
        let inputAssignIndex = 0

        return prevNodes.map((node) => {
          const data = node.data as LayerNodeData

          // ── Input-like nodes: no incoming edges → assign input tensor previews ──
          // Detection is structural, not by blockType string, so it generalises to
          // any architecture (e.g. nn.Sequential, custom modules, multi-input models).
          if (inputNodeIds.has(node.id)) {
            const nodeKey = data._attrPath ?? data._attrName
            const namedEntry = nodeKey
              ? inputEntries.find(([k]) => k === nodeKey || k.startsWith(nodeKey) || nodeKey.startsWith(k))
              : undefined
            if (namedEntry) {
              return { ...node, data: { ...data, _dataPreview: namedEntry[1] } satisfies LayerNodeData }
            }
            // Fall back to positional assignment so each Input card gets a distinct tensor
            const positional = inputEntries[inputAssignIndex]
            if (positional) {
              inputAssignIndex += 1
              return { ...node, data: { ...data, _dataPreview: positional[1] } satisfies LayerNodeData }
            }
            return node
          }

          // ── All other nodes: match by _attrPath / _attrName ──
          const matchKey = data._attrPath ?? data._attrName
          if (matchKey) {
            // Exact match
            if (result.previews[matchKey]) {
              return { ...node, data: { ...data, _dataPreview: result.previews[matchKey] } satisfies LayerNodeData }
            }
            // Suffix / prefix match (dotted paths like "layer1.0" vs "layer1.0.conv")
            const suffixMatch = Object.entries(result.previews).find(
              ([k]) => k === matchKey || k.endsWith(`.${matchKey}`) || matchKey.endsWith(`.${k}`),
            )
            if (suffixMatch) {
              return { ...node, data: { ...data, _dataPreview: suffixMatch[1] } satisfies LayerNodeData }
            }
          }

          return node
        })
      })
    }).catch(() => { /* trace failure is silent — cards remain visible without preview */ })

    return () => { cancelled = true }
  // Re-run only when the loaded model changes (not on every graph mutation)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadedModel])

  useEffect(() => {
    let cancelled = false

    getCodexStatus()
      .then((status) => {
        if (!cancelled) setCodexStatus(status)
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setCodexStatus({
            transport: 'unavailable',
            authReady: false,
            authMode: '',
            binaryPath: cause instanceof Error ? cause.message : String(cause),
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const handleDeleteNode = useCallback((nodeId: string) => {
    const next = removeNodeFromGraph(nodesRef.current, edgesRef.current, nodeId)
    commitGraph(next.nodes, next.edges)
    setPendingConnectionSourceId((current) => (current === nodeId ? null : current))
    onNodeSelect(null)
  }, [commitGraph, onNodeSelect])

  const handleStartConnect = useCallback((nodeId: string) => {
    setPendingConnectionSourceId((current) => (current === nodeId ? null : nodeId))
    onNodeSelect(null)
  }, [onNodeSelect])

  const handleUpdateParam = useCallback((nodeId: string, key: string, value: ParamValue) => {
    const nextNodes = nodesRef.current.map((node) => {
      if (node.id !== nodeId) return node
      const data = node.data as LayerNodeData
      return {
        ...node,
        data: {
          ...data,
          params: {
            ...data.params,
            [key]: value,
          },
          _runtimeShapeLocked: false,
        } satisfies LayerNodeData,
      }
    })
    commitGraph(nextNodes, edgesRef.current)

    // 파라미터 변경 후 1.5초 debounce → trace 재실행 (미리보기 이미지 갱신)
    const lm = loadedModelRef.current
    if (lm?.sourceFile && lm?.model?.name) {
      if (retraceTimerRef.current) clearTimeout(retraceTimerRef.current)
      retraceTimerRef.current = setTimeout(() => {
        const m = loadedModelRef.current
        if (!m?.sourceFile || !m?.model?.name) return
        const gitRoot = m.gitInfo?.root
        const repoRoot = gitRoot ?? m.sourceFile.replace(/\/[^/]+$/, '')
        const payload = {
          repoRoot,
          sourceFile: m.sourceFile,
          modelName: m.model.name,
          runtimeFactory: m.constructorCallable ?? null,
          task: m.benchmark?.task ?? '',
          sample: m.samplePreview?.resolvedPath ? {
            resolvedPath: m.samplePreview.resolvedPath,
            width: m.samplePreview.width,
            height: m.samplePreview.height,
            mimeType: m.samplePreview.mimeType,
            source: m.samplePreview.source,
            strategy: m.samplePreview.strategy,
          } : null,
        }
        traceLayerData(payload).then((result) => {
          if (result.error && Object.keys(result.previews).length === 0 && Object.keys(result.inputPreviews).length === 0) return
          const inputEntries = Object.entries(result.inputPreviews)
          setNodes((prevNodes) => {
            const currentEdges = edgesRef.current
            const inputNodeIds = new Set(
              prevNodes.filter((n) => !currentEdges.some((e) => e.target === n.id)).map((n) => n.id)
            )
            let inputAssignIndex = 0
            return prevNodes.map((node) => {
              const data = node.data as LayerNodeData
              if (inputNodeIds.has(node.id)) {
                const nodeKey = data._attrPath ?? data._attrName
                const namedEntry = nodeKey
                  ? inputEntries.find(([k]) => k === nodeKey || k.startsWith(nodeKey) || nodeKey.startsWith(k))
                  : undefined
                if (namedEntry) return { ...node, data: { ...data, _dataPreview: namedEntry[1] } satisfies LayerNodeData }
                const positional = inputEntries[inputAssignIndex]
                if (positional) { inputAssignIndex += 1; return { ...node, data: { ...data, _dataPreview: positional[1] } satisfies LayerNodeData } }
                return node
              }
              const matchKey = data._attrPath ?? data._attrName
              if (matchKey) {
                if (result.previews[matchKey]) return { ...node, data: { ...data, _dataPreview: result.previews[matchKey] } satisfies LayerNodeData }
                const suffixMatch = Object.entries(result.previews).find(([k]) => k === matchKey || k.endsWith(`.${matchKey}`) || matchKey.endsWith(`.${k}`))
                if (suffixMatch) return { ...node, data: { ...data, _dataPreview: suffixMatch[1] } satisfies LayerNodeData }
              }
              return node
            })
          })
        }).catch(() => {})
      }, 1500)
    }
  }, [commitGraph])

  const diagnostics = buildGraphDiagnostics(nodes, edges, loadedModel?.model)
  const graphHealth = graphHealthFromDiagnostics(diagnostics)

  // 노드 id → outputShape / hasData / tensorSample 맵 (엣지 장식용)
  const nodeShapeMap = new Map(
    nodes.map(n => {
      const nd = n.data as LayerNodeData
      const preview = nd._dataPreview
      const tensorSample = preview?.kind === 'vector'
        ? preview.values.slice(0, 5)
        : undefined
      return [n.id, { shape: nd.outputShape, hasData: !!preview, tensorSample }]
    })
  )
  const flowReachableNodeIds = computeFlowReachableNodeIds(nodes, edges)

  const renderedNodes = decorateNodesWithDiagnostics(nodes, diagnostics).map((node) => ({
    ...node,
    data: {
      ...(node.data as LayerNodeData),
      _canStartConnect: (node.data as LayerNodeData).blockType !== 'Output',
      _onStartConnect: handleStartConnect,
      _isPendingConnectSource: pendingConnectionSourceId === node.id,
      _canDelete: true,
      _onDelete: handleDeleteNode,
      _onUpdateParam: handleUpdateParam,
    } satisfies LayerNodeData,
  }))

  // 엣지에 데이터 흐름 정보 장식 (source 노드의 shape, hasData)
  const renderedEdges = edges.map(edge => {
    const src = nodeShapeMap.get(edge.source)
    const hasReachableFlow = flowReachableNodeIds.has(edge.source) && flowReachableNodeIds.has(edge.target)
    return {
      ...edge,
      type: 'flowEdge',
      data: {
        ...((edge.data as object | undefined) ?? {}),
        hasData: (src?.hasData ?? false) || hasReachableFlow,
        shapeLabel: src?.shape ?? undefined,
        tensorSample: src?.tensorSample ?? undefined,
      },
    }
  })

  const handleSaveWorktree = async () => {
    if (!loadedModel?.gitInfo.isGit || !loadedModel?.gitInfo.root) return

    setSavingWorktree(true)
    try {
      const worktree = await createWorktree(loadedModel.gitInfo.root, loadedModel.sourceFile)
      const generatedCode = generatePyTorchCode(nodes, edges)
      await saveToWorktree(worktree.worktreePath, loadedModel.gitInfo.root, loadedModel.sourceFile, generatedCode)
      setWorktreeInfo({ path: worktree.worktreePath, branch: worktree.branch })
      setSurfaceStatus({ level: 'success', message: `Saved generated code to worktree branch ${worktree.branch}.` })
      onWorktreeSaved?.(worktree.worktreePath, worktree.branch)
    } catch (cause: unknown) {
      setSurfaceStatus({
        level: 'error',
        message: `Worktree save failed: ${cause instanceof Error ? cause.message : String(cause)}`,
      })
    } finally {
      setSavingWorktree(false)
    }
  }

  const handleAutoLayout = useCallback((direction: 'LR' | 'TB' | 'RL' | 'BT' = 'LR') => {
    const laidOut = layoutGraphWithDirection(nodesRef.current, edgesRef.current, direction)
    commitGraph(laidOut.nodes, laidOut.edges)
    setSurfaceStatus({ level: 'success', message: `Auto layout applied (${direction}).` })
  }, [commitGraph])
  void handleAutoLayout

  const handleNormalizeInputShapes = useCallback(() => {
    const normalized = normalizeInputNodeShapes(nodesRef.current, edgesRef.current)
    commitGraph(normalized, edgesRef.current)
    setSurfaceStatus({ level: 'success', message: 'Normalized Input node shapes to match connected downstream layers.' })
  }, [commitGraph])
  void handleNormalizeInputShapes

  const handleClearGraph = useCallback(() => {
    commitGraph([], [])
    setWorktreeInfo(null)
    setPendingConnectionSourceId(null)
    setSurfaceStatus({ level: 'success', message: 'Canvas cleared.' })
  }, [commitGraph])

  const handleValidateCodex = async () => {
    setCodexValidation({ status: 'running', message: 'Running official codex exec validation…' })
    try {
      const result = await validateCodexEdit()
      setCodexValidation({ status: 'done', message: result.message })
      const status = await getCodexStatus()
      setCodexStatus(status)
    } catch (cause: unknown) {
      setCodexValidation({
        status: 'error',
        message: cause instanceof Error ? cause.message : String(cause),
      })
    }
  }

  const handleApplyCodexEdit = async () => {
    if (!loadedModel?.gitInfo.isGit || !loadedModel?.gitInfo.root || !codexInstruction.trim()) return

    setCodexEdit({
      status: 'running',
      message: 'Running official codex exec against a disposable git worktree…',
      diffSummary: '',
    })

    try {
      const result = await applyCodexSourceEdit(
        loadedModel.gitInfo.root,
        loadedModel.sourceFile,
        codexInstruction,
      )
      setWorktreeInfo({ path: result.worktreePath, branch: result.branch })
      onWorktreeSaved?.(result.worktreePath, result.branch)
      setCodexEdit({
        status: 'done',
        message: result.message,
        diffSummary: result.diffSummary,
      })
    } catch (cause: unknown) {
      setCodexEdit({
        status: 'error',
        message: cause instanceof Error ? cause.message : String(cause),
        diffSummary: '',
      })
    }
  }
  void handleValidateCodex
  void handleApplyCodexEdit

  const onConnect = useCallback((params: Connection) => {
    if (!params.source || !params.target) return
    commitGraph(nodesRef.current, applyConnectionRules(nodesRef.current, edgesRef.current, params.source, params.target))
  }, [commitGraph])

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    const blockType = event.dataTransfer.getData('application/dl-block-type')
    if (!blockType) return
    if (!getBlockDef(blockType)) return

    const instance = rfInstanceRef.current as unknown as {
      screenToFlowPosition?: (position: { x: number; y: number }) => { x: number; y: number }
    } | null
    const bounds = reactFlowWrapper.current?.getBoundingClientRect()
    if (!bounds) return

    const flowPosition = instance?.screenToFlowPosition
      ? instance.screenToFlowPosition({ x: event.clientX, y: event.clientY })
      : {
          x: event.clientX - bounds.left,
          y: event.clientY - bounds.top,
        }

    const position = {
      x: flowPosition.x - 70,
      y: flowPosition.y - 30,
    }

    const newNode = createNodeFromBlock(blockType, nodesRef.current, undefined, position, nodesRef.current.length)
    if (!newNode) return

    const insertionEdge = findInsertionEdge(nodesRef.current, edgesRef.current, flowPosition)
    const nextEdges = insertionEdge
      ? spliceEdgeWithNode(edgesRef.current, insertionEdge, newNode.id)
      : edgesRef.current

    commitGraph([...nodesRef.current, newNode], nextEdges)
    if (insertionEdge) {
      setSurfaceStatus({
        level: 'success',
        message: `Inserted ${blockType} between ${insertionEdge.source} and ${insertionEdge.target}.`,
      })
    }
  }, [commitGraph])

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    const data = node.data as LayerNodeData
    if (pendingConnectionSourceId) {
      if (pendingConnectionSourceId === node.id) {
        setPendingConnectionSourceId(null)
      } else {
        commitGraph(nodesRef.current, applyConnectionRules(nodesRef.current, edgesRef.current, pendingConnectionSourceId, node.id))
        setPendingConnectionSourceId(null)
      }
    }
    onNodeSelect(data.blockType)
  }, [commitGraph, onNodeSelect, pendingConnectionSourceId])

  const onPaneClick = useCallback(() => {
    setPendingConnectionSourceId(null)
    onNodeSelect(null)
  }, [onNodeSelect])

  const handleGenerateCode = () => {
    const generated = generatePyTorchCode(nodes, edges)
    setCode(generated)
    setShowCode(true)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '8px 16px',
        background: '#161b27',
        borderBottom: '1px solid #2a3347',
        flexShrink: 0,
      }}>
        {loadedModel ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span
              data-testid="loaded-model-name"
              style={{ fontSize: 11, color: '#818cf8', fontFamily: 'monospace' }}
            >
              {loadedModel.model.name}
            </span>
            <span style={{ fontSize: 10, color: '#475569' }}>
              {loadedModel.sourceFile.split('/').slice(-2).join('/')}
            </span>
            {loadedModel.gitInfo.isGit && (
              <span style={{
                fontSize: 10,
                padding: '1px 6px',
                borderRadius: 4,
                background: '#1e2535',
                color: '#64748b',
                border: '1px solid #334155',
              }}>
                git:{loadedModel.gitInfo.branch}
              </span>
            )}
            <span style={{
              fontSize: 10,
              padding: '1px 6px',
              borderRadius: 4,
              background: loadedModel.traceMode === 'runtime-exact' ? '#052e16' : loadedModel.traceMode === 'exploratory-static' ? '#1f2937' : '#3f1d1d',
              color: loadedModel.traceMode === 'runtime-exact' ? '#86efac' : loadedModel.traceMode === 'exploratory-static' ? '#cbd5e1' : '#fca5a5',
              border: '1px solid #334155',
            }}>
              {formatTraceMode(loadedModel.traceMode)}
            </span>
            <span style={{
              fontSize: 10,
              padding: '1px 6px',
              borderRadius: 4,
              background: loadedModel.exactness === 'runtime_exact' ? '#172554' : '#3f1d1d',
              color: loadedModel.exactness === 'runtime_exact' ? '#bfdbfe' : '#fecaca',
              border: '1px solid #334155',
            }}>
              {formatExactness(loadedModel.exactness)}
            </span>
            {loadedModel.runtimeMs ? (
              <span style={{ fontSize: 10, color: '#64748b' }}>{loadedModel.runtimeMs} ms</span>
            ) : null}
          </div>
        ) : (
          <span style={{ fontSize: 12, color: '#475569' }}>
            Drag blocks here or use Open Project to load a model.
          </span>
        )}

        {pendingConnectionSourceId && (
          <span style={{
            fontSize: 10,
            padding: '2px 8px',
            borderRadius: 4,
            background: '#172554',
            color: '#bfdbfe',
            border: '1px solid #1d4ed8',
          }}>
            linking from {pendingConnectionSourceId} → click a target node
          </span>
        )}

        <div style={{ flex: 1 }} />

        {loadedModel && (
          <>
            <ToolbarButton active={!expandModules} onClick={() => setExpandModules(false)}>
              High-level View
            </ToolbarButton>
            <ToolbarButton active={expandModules} onClick={() => setExpandModules(true)}>
              Expanded View
            </ToolbarButton>
          </>
        )}

        <ToolbarButton active={hideUtilityOps} onClick={() => setHideUtilityOps((current) => !current)}>
          {hideUtilityOps ? 'Utility Ops Hidden' : 'Show Utility Ops'}
        </ToolbarButton>

        <ToolbarButton active={drawerOpen} onClick={() => setDrawerOpen((current) => !current)}>
          {drawerOpen ? 'Hide Diagnostics' : 'Show Diagnostics'}
        </ToolbarButton>

        {worktreeInfo && (
          <span style={{
            fontSize: 10,
            color: '#34d399',
            padding: '2px 8px',
            background: '#064e3b',
            borderRadius: 4,
          }}>
            saved to branch: {worktreeInfo.branch}
          </span>
        )}

        <span
          data-testid="graph-health-badge"
          style={{
            fontSize: 10,
            padding: '1px 6px',
            borderRadius: 4,
            background: graphHealth.level === 'healthy' ? '#052e16' : graphHealth.level === 'warning' ? '#3b2f12' : '#3f1d1d',
            color: graphHealth.level === 'healthy' ? '#86efac' : graphHealth.level === 'warning' ? '#fde68a' : '#fecaca',
            border: '1px solid #334155',
          }}
        >
          {graphHealth.label}
        </span>

        {surfaceStatus.message && (
          <span
            data-testid="surface-status-message"
            style={{
              fontSize: 10,
              padding: '1px 6px',
              borderRadius: 4,
              background: surfaceStatus.level === 'error' ? '#3f1d1d' : '#172554',
              color: surfaceStatus.level === 'error' ? '#fecaca' : '#bfdbfe',
              border: '1px solid #334155',
              maxWidth: 320,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
            title={surfaceStatus.message}
          >
            {surfaceStatus.message}
          </span>
        )}

        {loadedModel?.gitInfo.isGit && (
          <button
            data-testid="save-worktree-button"
            onClick={handleSaveWorktree}
            disabled={savingWorktree}
            style={{
              padding: '5px 12px',
              borderRadius: 6,
              border: 'none',
              fontSize: 12,
              background: savingWorktree ? '#374151' : '#059669',
              color: 'white',
              cursor: savingWorktree ? 'default' : 'pointer',
            }}
          >
            {savingWorktree ? 'Creating worktree…' : 'Save to Worktree'}
          </button>
        )}

        <button
          onClick={() => {
            handleClearGraph()
          }}
          style={{
            padding: '5px 10px',
            borderRadius: 6,
            border: '1px solid #334155',
            background: 'transparent',
            color: '#64748b',
            fontSize: 12,
            cursor: 'pointer',
          }}
        >
          Clear
        </button>

        <button
          data-testid="generate-code-button"
          onClick={handleGenerateCode}
          style={{
            padding: '5px 12px',
            borderRadius: 6,
            border: 'none',
            background: '#4f46e5',
            color: 'white',
            fontSize: 12,
            cursor: 'pointer',
          }}
        >
          Generate PyTorch Code
        </button>

        {/* 워크트리 자동저장 상태 + Merge 버튼 */}
        {worktreeInfo && (
          <>
            <span style={{ fontSize: 10, color: autoSaveStatus === 'saved' ? '#34d399' : autoSaveStatus === 'error' ? '#f87171' : '#94a3b8' }}>
              {autoSaveStatus === 'saved' ? `✓ auto-saved → ${worktreeInfo.branch}` :
               autoSaveStatus === 'saving' ? '⏳ saving…' :
               autoSaveStatus === 'error' ? '✗ save failed' :
               `branch: ${worktreeInfo.branch}`}
            </span>
            <button
              data-testid="merge-worktree-button"
              onClick={() => onMergeRequested?.(worktreeInfo.path, worktreeInfo.branch)}
              style={{
                padding: '5px 12px',
                borderRadius: 6,
                border: '1px solid #22c55e',
                background: 'transparent',
                color: '#22c55e',
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              ⬆ Merge to Main
            </button>
          </>
        )}
      </div>

      <div
        className="flex-1 relative"
        ref={reactFlowWrapper}
        data-testid="flow-canvas"
      >
        <ReactFlow
          nodes={renderedNodes}
          edges={renderedEdges}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onConnect={onConnect}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.18 }}
          deleteKeyCode="Delete"
          style={{ background: '#0f1117' }}
          onInit={(instance) => { rfInstanceRef.current = instance as unknown as ReturnType<typeof useReactFlow> }}
        >
          <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#1e2535" />
          <Controls className="!bg-[#161b27] !border-[#2a3347]" />
          <MiniMap
            style={{ background: '#161b27', border: '1px solid #2a3347' }}
            nodeColor="#1e2535"
            maskColor="rgba(0,0,0,0.4)"
          />
        </ReactFlow>

        {nodes.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <p className="text-slate-600 text-sm">Drag blocks here to build your architecture</p>
            <p className="text-slate-700 text-xs mt-1">Connect nodes with arrows to define data flow</p>
          </div>
        )}
      </div>

      {/* 하단 바 — Canvas Map + Sample Data + Graph Stats (같은 행) */}
      {drawerOpen && (
        <BottomBar
          nodes={nodes}
          edges={edges}
          graphHealth={graphHealth}
          diagnostics={diagnostics}
          loadedModel={loadedModel}
          onNavigate={(x, y) => {
            const inst = rfInstanceRef.current as unknown as { setCenter: (x: number, y: number, opts: object) => void; getZoom: () => number } | null
            if (inst) inst.setCenter(x, y, { zoom: inst.getZoom(), duration: 400 })
          }}
        />
      )}
    </div>
  )
})

export default FlowCanvas

// ── CanvasMapPanel ─────────────────────────────────────────────────────────────

function CanvasMapPanel({
  nodes,
  edges,
  onClickCanvas,
}: {
  nodes: import('@xyflow/react').Node[]
  edges: import('@xyflow/react').Edge[]
  onClickCanvas?: (canvasX: number, canvasY: number) => void
}) {
  if (nodes.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#1e2535', fontSize: 11 }}>
        Empty canvas
      </div>
    )
  }

  const xs = nodes.map(n => n.position.x)
  const ys = nodes.map(n => n.position.y)
  const minX = Math.min(...xs) - 40
  const minY = Math.min(...ys) - 40
  const maxX = Math.max(...xs) + 220
  const maxY = Math.max(...ys) + 80
  const vbW = maxX - minX || 400
  const vbH = maxY - minY || 200

  const handleSvgClick = onClickCanvas
    ? (e: React.MouseEvent<SVGSVGElement>) => {
        const svg = e.currentTarget
        const rect = svg.getBoundingClientRect()
        const px = (e.clientX - rect.left) / rect.width
        const py = (e.clientY - rect.top) / rect.height
        const canvasX = minX + px * vbW
        const canvasY = minY + py * vbH
        onClickCanvas(canvasX, canvasY)
      }
    : undefined

  const nodeMap = new Map(nodes.map(n => [n.id, n]))

  return (
    <svg
      viewBox={`${minX} ${minY} ${vbW} ${vbH}`}
      style={{ width: '100%', height: '100%', display: 'block', cursor: onClickCanvas ? 'crosshair' : 'default' }}
      preserveAspectRatio="xMidYMid meet"
      onClick={handleSvgClick}
    >
      {/* edges */}
      {edges.map(e => {
        const src = nodeMap.get(e.source)
        const tgt = nodeMap.get(e.target)
        if (!src || !tgt) return null
        const x1 = src.position.x + 90
        const y1 = src.position.y + 30
        const x2 = tgt.position.x + 90
        const y2 = tgt.position.y + 30
        return (
          <line key={e.id} x1={x1} y1={y1} x2={x2} y2={y2}
            stroke="#2a3347" strokeWidth={8} strokeLinecap="round" />
        )
      })}
      {/* nodes */}
      {nodes.map(n => {
        const nd = n.data as LayerNodeData
        const color = nd._groupColor ?? '#4a556a'
        const hasData = !!nd._dataPreview
        return (
          <g key={n.id}>
            <rect
              x={n.position.x} y={n.position.y}
              width={180} height={50}
              rx={8}
              fill="#161b27"
              stroke={hasData ? color : '#2a3347'}
              strokeWidth={hasData ? 3 : 1.5}
            />
            <text
              x={n.position.x + 90} y={n.position.y + 22}
              textAnchor="middle" dominantBaseline="middle"
              fill="#94a3b8" fontSize={14} fontFamily="monospace"
            >
              {nd.blockType}
            </text>
            {nd.outputShape && (
              <text
                x={n.position.x + 90} y={n.position.y + 38}
                textAnchor="middle" dominantBaseline="middle"
                fill="#334155" fontSize={10} fontFamily="monospace"
              >
                {nd.outputShape}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

// ── 하단 바: Canvas Map + Sample Data + Stats (같은 행) ──────────────────────

function BottomBar({
  nodes,
  edges,
  graphHealth,
  diagnostics,
  loadedModel,
  onNavigate,
}: {
  nodes: import('@xyflow/react').Node[]
  edges: import('@xyflow/react').Edge[]
  graphHealth: { level: string; label: string }
  diagnostics: { severity: string }[]
  loadedModel?: import('../../lib/api').LoadedModelPayload | null
  onNavigate?: (canvasX: number, canvasY: number) => void
}) {

  return (
    <div style={{
      height: 140,
      background: '#0d1120',
      borderTop: '1px solid #1e2535',
      display: 'flex',
      alignItems: 'stretch',
      flexShrink: 0,
    }}>
      {/* 미니맵 — 클릭하면 해당 위치로 이동 */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', top: 8, left: 12,
          fontSize: 9, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600, zIndex: 1,
        }}>Canvas Map</div>
        <div style={{ width: '100%', height: '100%', paddingTop: 24 }}>
          <CanvasMapPanel nodes={nodes} edges={edges} onClickCanvas={onNavigate} />
        </div>
      </div>

      {/* Sample Data — 모델 로딩 시에만 표시 */}
      {loadedModel?.samplePreview && (
        <div style={{
          width: 180,
          borderLeft: '1px solid #1e2535',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}>
          <div style={{
            padding: '4px 10px',
            borderBottom: '1px solid #1e2535',
            fontSize: 9,
            fontWeight: 600,
            color: '#64748b',
            textTransform: 'uppercase',
            letterSpacing: '0.07em',
            background: '#0d1120',
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            flexShrink: 0,
          }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#34d399', display: 'inline-block' }} />
            Sample Data
            {loadedModel.benchmark?.label && (
              <span style={{ marginLeft: 'auto', fontSize: 8, color: '#334155' }}>
                {loadedModel.benchmark.label}
              </span>
            )}
          </div>
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {loadedModel.samplePreview.imageUrl ? (
              <img
                data-testid="sample-preview-image"
                src={loadedModel.samplePreview.imageUrl}
                alt={loadedModel.samplePreview.caption ?? loadedModel.samplePreview.label ?? 'Sample preview'}
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
              />
            ) : (
              <div style={{ padding: 10, fontSize: 10, color: '#334155' }}>No preview</div>
            )}
          </div>
          {(loadedModel.samplePreview.label || loadedModel.samplePreview.caption) && (
            <div style={{ padding: '3px 8px', fontSize: 9, color: '#475569', borderTop: '1px solid #1e2535', flexShrink: 0, background: '#0a0f1c' }}>
              {loadedModel.samplePreview.caption ?? loadedModel.samplePreview.label}
            </div>
          )}
        </div>
      )}

      {/* Graph Stats */}
      <div style={{
        width: 200,
        borderLeft: '1px solid #1e2535',
        padding: '10px 14px',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        justifyContent: 'center',
      }}>
        <div style={{ fontSize: 9, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600 }}>Graph</div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <StatChip label="Nodes" value={nodes.length} />
          <StatChip label="Edges" value={edges.length} />
          <StatChip label="Health" value={graphHealth.label} color={
            graphHealth.level === 'healthy' ? '#34d399' : graphHealth.level === 'warning' ? '#fbbf24' : '#f87171'
          } />
        </div>
        {loadedModel?.model?.name && (
          <div style={{ fontSize: 10, color: '#475569', marginTop: 2 }}>
            {loadedModel.model.name}
          </div>
        )}
        {diagnostics.filter(d => d.severity === 'error').length > 0 && (
          <div style={{ fontSize: 10, color: '#f87171' }}>
            {diagnostics.filter(d => d.severity === 'error').length} error(s)
          </div>
        )}
      </div>
    </div>
  )
}

// ── ToolbarButton ─────────────────────────────────────────────────────────────

function ToolbarButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '5px 12px',
        borderRadius: 6,
        border: `1px solid ${active ? '#1d4ed8' : '#334155'}`,
        background: active ? '#172554' : '#111827',
        color: active ? '#bfdbfe' : '#94a3b8',
        fontSize: 12,
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  )
}

// ── StatChip ─────────────────────────────────────────────────────────────────

function StatChip({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '4px 10px',
      background: '#131927',
      borderRadius: 6,
      border: '1px solid #1e2535',
      minWidth: 50,
    }}>
      <span style={{ fontSize: 14, fontWeight: 700, color: color ?? '#94a3b8' }}>{value}</span>
      <span style={{ fontSize: 8, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.07em' }}>{label}</span>
    </div>
  )
}
