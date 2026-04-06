import type { Edge, Node } from '@xyflow/react'
import * as dagre from 'dagre'
import type { LayerNodeData } from '../components/FlowCanvas/nodes/LayerNode'

const DEFAULT_WIDTH = 220
const DEFAULT_HEIGHT = 112

function estimateNodeSize(node: Node): { width: number; height: number } {
  const data = node.data as LayerNodeData
  const paramCount = Object.keys(data.params ?? {}).filter((key) => key !== 'class').length
  const width = data._isCollapsed ? 260 : data.blockType === 'Input' ? 200 : DEFAULT_WIDTH
  let height = DEFAULT_HEIGHT + Math.min(paramCount, 4) * 16
  if (data._groupName) height += 18
  if (data.outputShape) height += 18
  if (data._isCollapsed) height += 18
  return { width, height }
}

export function layoutGraph(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const graph = new dagre.graphlib.Graph()
  graph.setDefaultEdgeLabel(() => ({}))
  graph.setGraph({
    rankdir: 'LR',
    ranksep: 140,
    nodesep: 70,
    edgesep: 30,
    marginx: 48,
    marginy: 40,
    ranker: 'network-simplex',
  })

  const sizes = new Map<string, { width: number; height: number }>()

  for (const node of nodes) {
    const size = estimateNodeSize(node)
    sizes.set(node.id, size)
    graph.setNode(node.id, size)
  }

  for (const edge of edges) {
    graph.setEdge(edge.source, edge.target)
  }

  dagre.layout(graph)

  const laidOutNodes = nodes.map((node) => {
    const size = sizes.get(node.id) ?? { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT }
    const position = graph.node(node.id)
    return {
      ...node,
      position: {
        x: position.x - size.width / 2,
        y: position.y - size.height / 2,
      },
      style: {
        ...(node.style ?? {}),
        width: size.width,
      },
    }
  })

  const laidOutEdges = edges.map((edge) => ({
    ...edge,
    type: edge.type ?? 'smoothstep',
  }))

  return { nodes: laidOutNodes, edges: laidOutEdges }
}
