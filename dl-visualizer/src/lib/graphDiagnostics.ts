import type { Edge, Node } from '@xyflow/react'
import type { ParsedModel } from './api'
import type { LayerNodeData } from '../components/FlowCanvas/nodes/LayerNode'

export interface GraphDiagnostic {
  nodeId: string
  nodeLabel: string
  severity: 'error' | 'warning' | 'info'
  code: string
  title: string
  detail: string
  recoveryHint?: string
  expectedTerminal?: boolean
  utility?: boolean
}

const UTILITY_NODE_NAMES = new Set(['zip', 'merge', 'select', 'stack'])

function normalize(value: string | undefined): string {
  return (value ?? '').toLowerCase().replace(/_/g, ' ').trim()
}

function labelFor(data: LayerNodeData): string {
  return String(data._attrName ?? data._customClassName ?? data.blockType)
}

export function buildGraphDiagnostics(
  nodes: Node[],
  edges: Edge[],
  model: ParsedModel | null | undefined,
): GraphDiagnostic[] {
  const inDegree = new Map<string, number>()
  const outDegree = new Map<string, number>()
  const returnLabels = new Set((model?.return_outputs ?? []).map(output => normalize(output.label)))

  for (const node of nodes) {
    inDegree.set(node.id, 0)
    outDegree.set(node.id, 0)
  }
  for (const edge of edges) {
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1)
    outDegree.set(edge.source, (outDegree.get(edge.source) ?? 0) + 1)
  }

  const diagnostics: GraphDiagnostic[] = []
  const labelNodes = new Map<string, string[]>()

  for (const node of nodes) {
    const data = node.data as LayerNodeData
    const normalized = normalize(labelFor(data))
    if (!labelNodes.has(normalized)) labelNodes.set(normalized, [])
    labelNodes.get(normalized)!.push(node.id)
  }

  for (const node of nodes) {
    const data = node.data as LayerNodeData
    const label = labelFor(data)
    const normalizedLabel = normalize(label)
    const isInput = data.blockType === 'Input'
    const isOutput = data.blockType === 'Output'
    const isCustom = data.blockType === '__custom__'
    const isPlaceholder = data.blockType === 'Placeholder'
    const usesSelfState = !!data._usesSelfState
    const isUtility = !!data._isUtility || UTILITY_NODE_NAMES.has(normalizedLabel) || isPlaceholder
    const expectedTerminal = !!data._expectedTerminal || isOutput || returnLabels.has(normalizedLabel)
    const hasIn = (inDegree.get(node.id) ?? 0) > 0
    const hasOut = (outDegree.get(node.id) ?? 0) > 0
    const duplicateBranchLabel = (labelNodes.get(normalizedLabel) ?? []).some((candidateId) => {
      if (candidateId === node.id) return false
      return (outDegree.get(candidateId) ?? 0) > 0
    })

    if (data.shapeError) {
      diagnostics.push({
        nodeId: node.id,
        nodeLabel: label,
        severity: 'warning',
        code: 'shape_uncertain',
        title: 'Shape inference uncertain',
        detail: 'The node is connected, but shape propagation could not confirm the output shape.',
        recoveryHint: 'Check the upstream shape and this node’s parameters, then try Auto layout or Normalize input shapes.',
        expectedTerminal,
        utility: isUtility,
      })
    }

    if (!hasIn && !isInput) {
      const helperSource = isUtility || isCustom || usesSelfState
      diagnostics.push({
        nodeId: node.id,
        nodeLabel: label,
        severity: helperSource ? 'info' : 'error',
        code: helperSource ? (usesSelfState ? 'state_derived_root' : 'utility_source') : 'unexpected_disconnected_input',
        title: helperSource
          ? (usesSelfState ? 'State-derived helper root' : 'Helper branch root')
          : 'Unexpected disconnected input',
        detail: helperSource
          ? (usesSelfState
              ? 'This node is generated from model state or a nested helper call, so a data input edge is intentionally omitted in high-level view.'
              : 'This node behaves like a derived source or helper root, so the upstream edge is intentionally omitted in high-level view.')
          : 'This node has no incoming edge even though it is not a declared graph input.',
        recoveryHint: helperSource
          ? 'This is informational. Inspect the expanded view if you need to follow helper branches.'
          : 'Reconnect an upstream node or remove this node if it is no longer needed.',
        expectedTerminal,
        utility: helperSource,
      })
    }

    if (!hasOut) {
      if (expectedTerminal) {
        diagnostics.push({
          nodeId: node.id,
          nodeLabel: label,
          severity: 'info',
          code: 'expected_terminal',
          title: 'Expected terminal',
          detail: 'This node is a declared output or protected return terminal for the current model.',
          recoveryHint: 'No action required.',
          expectedTerminal: true,
          utility: isUtility,
        })
      } else if (data._isCollapsed) {
        diagnostics.push({
          nodeId: node.id,
          nodeLabel: label,
          severity: 'info',
          code: 'collapsed_boundary',
          title: 'Collapsed module boundary',
          detail: 'The module is collapsed in high-level view, so its internal exits are intentionally hidden.',
          recoveryHint: 'Switch to Expanded View if you need to inspect the internal path.',
          expectedTerminal,
          utility: isUtility,
        })
      } else if (isUtility || isCustom) {
        diagnostics.push({
          nodeId: node.id,
          nodeLabel: label,
          severity: 'info',
          code: 'utility_terminal',
          title: 'Helper terminal',
          detail: 'This helper or custom-module boundary ends a derived branch and is not treated as a graph failure.',
          recoveryHint: 'No action required unless you want to inspect low-level helper ops.',
          expectedTerminal,
          utility: true,
        })
      } else if (duplicateBranchLabel) {
        diagnostics.push({
          nodeId: node.id,
          nodeLabel: label,
          severity: 'info',
          code: 'branch_local_projection',
          title: 'Branch-local intermediate',
          detail: 'Another node with the same semantic label continues the branch downstream, so this dead-end is treated as an internal branch projection rather than a graph failure.',
          recoveryHint: 'No action required unless the downstream branch looks incorrect.',
          expectedTerminal: false,
          utility: true,
        })
      } else {
        diagnostics.push({
          nodeId: node.id,
          nodeLabel: label,
          severity: 'error',
          code: 'unexpected_dead_end',
          title: 'Unexpected dead-end node',
          detail: 'This node has no outgoing edge and is not classified as an expected model output.',
          recoveryHint: 'Reconnect this node to the next layer, mark it as an output, or delete it.',
          expectedTerminal: false,
          utility: false,
        })
      }
    }
  }

  return diagnostics.sort((a, b) => {
    const rank = { error: 0, warning: 1, info: 2 }
    return rank[a.severity] - rank[b.severity] || a.nodeLabel.localeCompare(b.nodeLabel)
  })
}

export function decorateNodesWithDiagnostics(nodes: Node[], diagnostics: GraphDiagnostic[]): Node[] {
  const strongest = new Map<string, GraphDiagnostic>()
  const rank = { error: 3, warning: 2, info: 1 }

  for (const diagnostic of diagnostics) {
    const current = strongest.get(diagnostic.nodeId)
    if (!current || rank[diagnostic.severity] > rank[current.severity]) {
      strongest.set(diagnostic.nodeId, diagnostic)
    }
  }

  return nodes.map((node) => {
    const diagnostic = strongest.get(node.id)
    const data = node.data as LayerNodeData
    const visualSeverity = diagnostic?.severity === 'info' ? undefined : diagnostic?.severity
    return {
      ...node,
      data: {
        ...data,
        diagnosticSeverity: visualSeverity,
        diagnosticReason: visualSeverity ? diagnostic?.detail : undefined,
        diagnosticCode: diagnostic?.code,
      } satisfies LayerNodeData,
    }
  })
}
