import { useEffect, useRef, useState } from 'react'
import AgentPanel from './components/AgentPanel/AgentPanel'
import FlowCanvas, { type FlowCanvasHandle } from './components/FlowCanvas/FlowCanvas'
import BlockPalette from './components/BlockPalette/BlockPalette'
import ScanModal from './components/ScanModal/ScanModal'
import { getBlockCatalog, mergeWorktreeToMain, type AgentCanvasAction, type AgentGraphSnapshot, type LoadedModelPayload } from './lib/api'
import { setDynamicBlocks } from './data/blocks'

export default function App() {
  const [selectedBlockType, setSelectedBlockType] = useState<string | null>(null)
  const [showScan, setShowScan] = useState(false)
  const [loadedModel, setLoadedModel] = useState<LoadedModelPayload | null>(null)
  const [worktreeNotice, setWorktreeNotice] = useState('')
  const [merging, setMerging] = useState(false)
  const [catalogVersion, setCatalogVersion] = useState(0)
  const [graphSnapshot, setGraphSnapshot] = useState<AgentGraphSnapshot>({ nodes: [], edges: [] })
  const flowCanvasRef = useRef<FlowCanvasHandle | null>(null)

  useEffect(() => {
    let cancelled = false
    getBlockCatalog()
      .then((blocks) => {
        if (cancelled) return
        setDynamicBlocks(blocks)
        setCatalogVersion((current) => current + 1)
      })
      .catch((error) => {
        console.error('Failed to load dynamic torch.nn block catalog', error)
      })

    return () => {
      cancelled = true
    }
  }, [])

  const handleLoad = (payload: LoadedModelPayload) => {
    setLoadedModel(payload)
    setShowScan(false)
    setWorktreeNotice('')
  }

  const handleMergeRequested = async (worktreePath: string, branch: string) => {
    if (!loadedModel?.gitInfo.root) return
    if (!confirm(`"${branch}" 브랜치의 변경사항을 main에 병합할까요?\n\n병합 전 diff를 에이전트가 검토합니다.`)) return
    setMerging(true)
    setWorktreeNotice('⏳ 병합 중…')
    try {
      const result = await mergeWorktreeToMain(loadedModel.gitInfo.root, worktreePath, branch)
      setWorktreeNotice(`✓ 병합 완료 (${branch} → main)`)
      console.info('[dl-viz] merge diff:\n', result.diffSummary)
    } catch (e) {
      setWorktreeNotice(`✗ 병합 실패: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setMerging(false)
    }
  }

  const registrySize = loadedModel
    ? Object.keys(loadedModel.registry).length
    : 0

  const executeAgentActions = async (actions: AgentCanvasAction[]) => {
    if (!flowCanvasRef.current) {
      return {
        applied: 0,
        warnings: ['Canvas is not ready yet.'],
        snapshot: graphSnapshot,
      }
    }
    return flowCanvasRef.current.executeAgentActions(actions)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '6px 16px', background: '#0f1117',
        borderBottom: '1px solid #1e2535', flexShrink: 0,
      }}>
        <span style={{ fontSize: 13, fontWeight: 700, color: '#818cf8' }}>DL Visualizer</span>
        <button
          onClick={() => setShowScan(true)}
          style={{
            padding: '4px 12px', borderRadius: 6, border: '1px solid #334155',
            background: '#1e2535', color: '#94a3b8', fontSize: 12, cursor: 'pointer',
          }}
        >
          📂 Open Project
        </button>
        {worktreeNotice && (
          <span style={{ fontSize: 11, color: '#34d399' }}>{worktreeNotice}</span>
        )}
        <div style={{ flex: 1 }} />
        {loadedModel && (
          <span style={{ fontSize: 10, color: '#475569' }}>
            {loadedModel.benchmark?.label ? `${loadedModel.benchmark.label} · ` : ''}
            {registrySize > 0 ? `registry: ${registrySize} classes` : 'runtime-first load'}
          </span>
        )}
      </div>

      {/* Main 3-panel */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <div style={{ width: 280, flexShrink: 0, height: '100%' }}>
          <AgentPanel
            selectedBlockType={selectedBlockType}
            graphSnapshot={graphSnapshot}
            loadedModel={loadedModel}
            onExecuteActions={executeAgentActions}
          />
        </div>

        <div style={{ flex: 1, height: '100%', position: 'relative' }}>
          <FlowCanvas
            ref={flowCanvasRef}
            catalogVersion={catalogVersion}
            onNodeSelect={setSelectedBlockType}
            loadedModel={loadedModel}
            onGraphSnapshotChange={setGraphSnapshot}
            onWorktreeSaved={(_path, branch) =>
              setWorktreeNotice(`✓ worktree 생성: ${branch}`)
            }
            onMergeRequested={merging ? undefined : handleMergeRequested}
          />
        </div>

        <div style={{ width: 224, flexShrink: 0, height: '100%' }}>
          <BlockPalette />
        </div>
      </div>

      {showScan && (
        <ScanModal onLoad={handleLoad} onClose={() => setShowScan(false)} />
      )}
    </div>
  )
}
