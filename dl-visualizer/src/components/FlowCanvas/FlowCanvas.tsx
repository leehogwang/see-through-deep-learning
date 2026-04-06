import { useCallback, useRef, useState, useEffect } from 'react'
import {
  ReactFlow,
  addEdge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Connection,
  type Edge,
  type Node,
  BackgroundVariant,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import LayerNode, { type LayerNodeData } from './nodes/LayerNode'
import { BLOCK_MAP } from '../../data/blocks'
import { calcOutputShape, parseShape, type Shape } from '../../lib/shapeCalculator'
import { generatePyTorchCode } from '../../lib/graphToCode'
import { createWorktree, saveToWorktree, type GitInfo } from '../../lib/api'

const NODE_TYPES = { layerNode: LayerNode }

let nodeId = 0
const uid = () => `n${++nodeId}`

function propagateShapes(nodes: Node[], edges: Edge[]): Node[] {
  const shapeMap: Record<string, Shape | null> = {}

  // Topological sort
  const inDeg: Record<string, number> = {}
  const adj: Record<string, string[]> = {}
  nodes.forEach(n => { inDeg[n.id] = 0; adj[n.id] = [] })
  edges.forEach(e => { adj[e.source]?.push(e.target); if (inDeg[e.target] !== undefined) inDeg[e.target]++ })

  const queue = nodes.filter(n => inDeg[n.id] === 0)
  const order: Node[] = []
  while (queue.length) {
    const n = queue.shift()!
    order.push(n)
    adj[n.id].forEach(nid => {
      if (inDeg[nid] !== undefined) { inDeg[nid]--; if (inDeg[nid] === 0) queue.push(nodes.find(x => x.id === nid)!) }
    })
  }

  return order.map(node => {
    const data = node.data as LayerNodeData
    const parentEdge = edges.find(e => e.target === node.id)
    const inputShape = parentEdge ? shapeMap[parentEdge.source] ?? null : null

    const { shape, error, str } = calcOutputShape(data.blockType, data.params, inputShape)
    shapeMap[node.id] = shape

    return {
      ...node,
      data: {
        ...data,
        outputShape: data.blockType === 'Input' ? str : str,
        shapeError: error && data.blockType !== 'Input',
      },
    }
  })
}

import { modelToGraph } from '../../lib/modelToGraph'
import type { ParsedModel } from '../../lib/api'

interface Props {
  onNodeSelect: (blockType: string | null) => void
  loadedModel?: {
    model: ParsedModel
    sourceFile: string
    gitInfo: GitInfo
    registry: Record<string, ParsedModel>
  } | null
  onWorktreeSaved?: (path: string, branch: string) => void
}

export default function FlowCanvas({ onNodeSelect, loadedModel, onWorktreeSaved }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const [showCode, setShowCode] = useState(false)
  const [code, setCode] = useState('')
  const [savingWorktree, setSavingWorktree] = useState(false)
  const [worktreeInfo, setWorktreeInfo] = useState<{ path: string; branch: string } | null>(null)
  const [expandModules, setExpandModules] = useState(false)

  // Load model from scan whenever loadedModel changes — pass registry for recursive expansion
  useEffect(() => {
    if (!loadedModel) return
    const { nodes: n, edges: e } = modelToGraph(
      loadedModel.model,
      loadedModel.registry,
      { expandSubmodules: expandModules },
    )
    const withShapes = propagateShapes(n, e)
    setNodes(withShapes)
    setEdges(e)
  }, [expandModules, loadedModel, setNodes, setEdges])

  const handleSaveWorktree = async () => {
    if (!loadedModel?.gitInfo.isGit || !loadedModel?.gitInfo.root) return
    setSavingWorktree(true)
    try {
      const wt = await createWorktree(loadedModel.gitInfo.root, loadedModel.sourceFile)
      const generatedCode = generatePyTorchCode(nodes, edges)
      await saveToWorktree(wt.worktreePath, loadedModel.gitInfo.root, loadedModel.sourceFile, generatedCode)
      setWorktreeInfo({ path: wt.worktreePath, branch: wt.branch })
      onWorktreeSaved?.(wt.worktreePath, wt.branch)
    } catch (e: unknown) {
      alert('Worktree save failed: ' + (e instanceof Error ? e.message : String(e)))
    } finally {
      setSavingWorktree(false)
    }
  }

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges(eds => {
        const newEdges = addEdge({
          ...params,
          animated: true,
          style: { stroke: '#6366f1', strokeWidth: 2 },
        }, eds)
        setNodes(nds => propagateShapes(nds, newEdges))
        return newEdges
      })
    },
    [setEdges, setNodes]
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const blockType = e.dataTransfer.getData('application/dl-block-type')
      if (!blockType) return

      const def = BLOCK_MAP[blockType]
      if (!def) return

      const bounds = reactFlowWrapper.current?.getBoundingClientRect()
      if (!bounds) return

      const position = {
        x: e.clientX - bounds.left - 70,
        y: e.clientY - bounds.top - 30,
      }

      // Auto-calc input shape for Input nodes
      let outputShape = ''
      let inputShapeCalc: Shape | null = null
      if (blockType === 'Input') {
        const shapeStr = String(def.defaultParams.shape || 'B,3,224,224')
        inputShapeCalc = parseShape(shapeStr)
        outputShape = '(' + shapeStr + ')'
      }

      const newNode: Node = {
        id: uid(),
        type: 'layerNode',
        position,
        data: {
          blockType,
          params: { ...def.defaultParams },
          outputShape,
          shapeError: false,
        } satisfies LayerNodeData,
      }
      void inputShapeCalc // suppress unused warning

      setNodes(nds => {
        const updated = [...nds, newNode]
        return propagateShapes(updated, edges)
      })
    },
    [edges, setNodes]
  )

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const data = node.data as LayerNodeData
      onNodeSelect(data.blockType)
    },
    [onNodeSelect]
  )

  const onPaneClick = useCallback(() => onNodeSelect(null), [onNodeSelect])

  const handleGenerateCode = () => {
    const c = generatePyTorchCode(nodes, edges)
    setCode(c)
    setShowCode(true)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Toolbar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '8px 16px', background: '#161b27',
        borderBottom: '1px solid #2a3347', flexShrink: 0,
      }}>
        {loadedModel ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 11, color: '#818cf8', fontFamily: 'monospace' }}>
              {loadedModel.model.name}
            </span>
            <span style={{ fontSize: 10, color: '#475569' }}>
              {loadedModel.sourceFile.split('/').slice(-2).join('/')}
            </span>
            {loadedModel.gitInfo.isGit && (
              <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: '#1e2535', color: '#64748b', border: '1px solid #334155' }}>
                git:{loadedModel.gitInfo.branch}
              </span>
            )}
          </div>
        ) : (
          <span style={{ fontSize: 12, color: '#475569' }}>Drag blocks → or use "Open Project" to load a model</span>
        )}
        <div style={{ flex: 1 }} />
        {loadedModel && (
          <button
            onClick={() => setExpandModules((current) => !current)}
            style={{
              padding: '5px 12px',
              borderRadius: 6,
              border: `1px solid ${expandModules ? '#1d4ed8' : '#334155'}`,
              background: expandModules ? '#172554' : '#111827',
              color: expandModules ? '#bfdbfe' : '#94a3b8',
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            {expandModules ? 'Expanded View' : 'High-level View'}
          </button>
        )}
        {worktreeInfo && (
          <span style={{ fontSize: 10, color: '#34d399', padding: '2px 8px', background: '#064e3b', borderRadius: 4 }}>
            ✓ saved to branch: {worktreeInfo.branch}
          </span>
        )}
        {loadedModel?.gitInfo.isGit && (
          <button
            onClick={handleSaveWorktree}
            disabled={savingWorktree}
            style={{
              padding: '5px 12px', borderRadius: 6, border: 'none', fontSize: 12,
              background: savingWorktree ? '#374151' : '#059669',
              color: 'white', cursor: savingWorktree ? 'default' : 'pointer',
            }}
          >
            {savingWorktree ? 'Creating worktree…' : '⑂ Save to Worktree'}
          </button>
        )}
        <button
          onClick={() => { setNodes([]); setEdges([]); setWorktreeInfo(null) }}
          style={{ padding: '5px 10px', borderRadius: 6, border: '1px solid #334155', background: 'transparent', color: '#64748b', fontSize: 12, cursor: 'pointer' }}
        >
          Clear
        </button>
        <button
          onClick={handleGenerateCode}
          style={{ padding: '5px 12px', borderRadius: 6, border: 'none', background: '#4f46e5', color: 'white', fontSize: 12, cursor: 'pointer' }}
        >
          Generate PyTorch Code
        </button>
      </div>

      <div className="flex-1 relative" ref={reactFlowWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
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

      {/* Code Modal */}
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
            <pre className="flex-1 overflow-auto p-4 text-xs font-mono text-emerald-300 leading-relaxed">
              {code}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}
