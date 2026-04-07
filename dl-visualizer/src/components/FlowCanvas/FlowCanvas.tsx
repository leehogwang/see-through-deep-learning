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
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  BackgroundVariant,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import LayerNode, { type LayerNodeData } from './nodes/LayerNode'
import InspectorDrawer from './InspectorDrawer'
import { getBlockDef, type ParamValue } from '../../data/blocks'
import { calcOutputShape, parseShape, type Shape } from '../../lib/shapeCalculator'
import { generatePyTorchCode } from '../../lib/graphToCode'
import {
  type AgentCanvasAction,
  type AgentGraphSnapshot,
  createWorktree,
  saveToWorktree,
  getCodexStatus,
  validateCodexEdit,
  applyCodexSourceEdit,
  parseDirRegistry,
  type LoadedModelPayload,
} from '../../lib/api'
import { modelToGraph } from '../../lib/modelToGraph'
import { buildGraphDiagnostics, decorateNodesWithDiagnostics } from '../../lib/graphDiagnostics'
import { layoutGraphWithDirection } from '../../lib/layoutGraph'

const NODE_TYPES = { layerNode: LayerNode }

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
    animated: true,
    style: { stroke: '#6366f1', strokeWidth: 2 },
  }
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
  const type = data.blockType.toLowerCase()
  const params = data.params || {}

  if (type === 'conv1d') {
    return `B, ${parseNumericParam(params.in_ch, 3)}, 128`
  }
  if (['conv2d', 'depthwiseconv2d', 'dilatedconv2d', 'transposedconv2d'].includes(type)) {
    return `B, ${parseNumericParam(params.in_ch, 3)}, 224, 224`
  }
  if (type === 'conv3d') {
    return `B, ${parseNumericParam(params.in_ch, 3)}, 16, 64, 64`
  }
  if (['multiheadattention', 'selfattention', 'crossattention', 'flashattention', 'gqa', 'mqa', 'linearattention', 'transformerencoderlayer', 'transformerdecoderlayer'].includes(type)) {
    return `B, 16, ${parseNumericParam(params.embed_dim ?? params.d_model, 128)}`
  }
  if (type === 'embedding') {
    return 'B, 16'
  }
  if (['linear', 'bilinear'].includes(type)) {
    return `B, ${parseNumericParam(params.in_features ?? params.in1_features, 128)}`
  }
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
}

const FlowCanvas = forwardRef<FlowCanvasHandle, Props>(function FlowCanvas(
  { catalogVersion = 0, onNodeSelect, loadedModel, onWorktreeSaved, onGraphSnapshotChange }: Props,
  ref,
) {
  const [nodes, setNodes] = useNodesState<Node>([])
  const [edges, setEdges] = useEdgesState<Edge>([])
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const nodesRef = useRef<Node[]>([])
  const edgesRef = useRef<Edge[]>([])
  const [showCode, setShowCode] = useState(false)
  const [code, setCode] = useState('')
  const [savingWorktree, setSavingWorktree] = useState(false)
  const [worktreeInfo, setWorktreeInfo] = useState<{ path: string; branch: string } | null>(null)
  const [expandModules, setExpandModules] = useState(false)
  const [hideUtilityOps, setHideUtilityOps] = useState(true)
  const [drawerOpen, setDrawerOpen] = useState(true)
  const [drawerTab, setDrawerTab] = useState<'sample' | 'diagnostics'>('sample')
  const [hideExpectedTerminals, setHideExpectedTerminals] = useState(true)
  const [pendingConnectionSourceId, setPendingConnectionSourceId] = useState<string | null>(null)
  const [codexStatus, setCodexStatus] = useState<Awaited<ReturnType<typeof getCodexStatus>> | null>(null)
  const [codexValidation, setCodexValidation] = useState<{
    status: 'idle' | 'running' | 'done' | 'error'
    message: string
  }>({ status: 'idle', message: '' })
  const [codexInstruction, setCodexInstruction] = useState('')
  const [codexEdit, setCodexEdit] = useState<{
    status: 'idle' | 'running' | 'done' | 'error'
    message: string
    diffSummary?: string
  }>({ status: 'idle', message: '' })
  const [surfaceStatus, setSurfaceStatus] = useState<{
    level: 'idle' | 'success' | 'error'
    message: string
  }>({ level: 'idle', message: '' })
  const [resolvedRegistry, setResolvedRegistry] = useState<Record<string, LoadedModelPayload['model']>>({})

  const commitGraph = useCallback((nextNodes: Node[], nextEdges: Edge[]) => {
    const dedupedEdges = dedupeEdges(nextEdges)
    const propagatedNodes = propagateShapes(nextNodes, dedupedEdges)
    nodesRef.current = propagatedNodes
    edgesRef.current = dedupedEdges
    setNodes(propagatedNodes)
    setEdges(dedupedEdges)
  }, [setEdges, setNodes])

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
  }, [commitGraph])

  const diagnostics = buildGraphDiagnostics(nodes, edges, loadedModel?.model)
  const graphHealth = graphHealthFromDiagnostics(diagnostics)
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

  const handleNormalizeInputShapes = useCallback(() => {
    const normalized = normalizeInputNodeShapes(nodesRef.current, edgesRef.current)
    commitGraph(normalized, edgesRef.current)
    setSurfaceStatus({ level: 'success', message: 'Normalized Input node shapes to match connected downstream layers.' })
  }, [commitGraph])

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

    const bounds = reactFlowWrapper.current?.getBoundingClientRect()
    if (!bounds) return

    const position = {
      x: event.clientX - bounds.left - 70,
      y: event.clientY - bounds.top - 30,
    }

    const newNode = createNodeFromBlock(blockType, nodesRef.current, undefined, position, nodesRef.current.length)
    if (!newNode) return
    commitGraph([...nodesRef.current, newNode], edgesRef.current)
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
      </div>

      <div
        className="flex-1 relative"
        ref={reactFlowWrapper}
        data-testid="flow-canvas"
      >
        <ReactFlow
          nodes={renderedNodes}
          edges={edges}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onConnect={onConnect}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          nodeTypes={NODE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.18 }}
          deleteKeyCode="Delete"
          style={{ background: '#0f1117' }}
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

      {drawerOpen && (
        <InspectorDrawer
          activeTab={drawerTab}
          onTabChange={setDrawerTab}
          samplePreview={loadedModel?.samplePreview}
          benchmark={loadedModel?.benchmark}
          diagnostics={diagnostics}
          graphHealth={graphHealth}
          hideExpectedTerminals={hideExpectedTerminals}
          onToggleHideExpectedTerminals={() => setHideExpectedTerminals((current) => !current)}
          onAutoLayout={() => handleAutoLayout('LR')}
          onNormalizeInputShapes={handleNormalizeInputShapes}
          onClearGraph={handleClearGraph}
          codexStatus={codexStatus}
          codexValidation={codexValidation}
          onValidateCodex={handleValidateCodex}
          codexInstruction={codexInstruction}
          onCodexInstructionChange={setCodexInstruction}
          codexEdit={codexEdit}
          onApplyCodexEdit={handleApplyCodexEdit}
          traceMode={formatTraceMode(loadedModel?.traceMode)}
          exactness={formatExactness(loadedModel?.exactness)}
          unsupportedReason={loadedModel?.unsupportedReason}
          constructorStrategy={loadedModel?.constructorStrategy}
          constructorCallable={loadedModel?.constructorCallable}
          inputStrategy={loadedModel?.inputStrategy}
          runtimeMs={loadedModel?.runtimeMs}
        />
      )}

      {showCode && (
        <div className="absolute inset-0 z-50 bg-black/70 flex items-center justify-center p-6">
          <div className="bg-[#161b27] border border-[#2a3347] rounded-xl w-full max-w-2xl max-h-[80vh] flex flex-col shadow-2xl">
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a3347]">
              <span className="text-sm font-semibold text-slate-200">Generated PyTorch Code</span>
              <div className="flex gap-2">
                <button
                  onClick={() => navigator.clipboard.writeText(code)}
                  className="text-xs px-2 py-1 rounded border border-[#2a3347] text-slate-400 hover:text-slate-200 transition-colors"
                >
                  Copy
                </button>
                <button
                  onClick={() => setShowCode(false)}
                  className="text-xs px-2 py-1 rounded border border-[#2a3347] text-slate-400 hover:text-red-400 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
            <pre
              data-testid="generated-code-output"
              className="flex-1 overflow-auto p-4 text-xs font-mono text-emerald-300 leading-relaxed"
            >
              {code}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
})

export default FlowCanvas

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
