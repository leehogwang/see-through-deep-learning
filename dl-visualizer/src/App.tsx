import { useState } from 'react'
import AgentPanel from './components/AgentPanel/AgentPanel'
import FlowCanvas from './components/FlowCanvas/FlowCanvas'
import BlockPalette from './components/BlockPalette/BlockPalette'
import ScanModal from './components/ScanModal/ScanModal'
import type { ParsedModel, GitInfo } from './lib/api'

interface LoadedModel {
  model: ParsedModel
  sourceFile: string
  gitInfo: GitInfo
  registry: Record<string, ParsedModel>
}

export default function App() {
  const [selectedBlockType, setSelectedBlockType] = useState<string | null>(null)
  const [showScan, setShowScan] = useState(false)
  const [loadedModel, setLoadedModel] = useState<LoadedModel | null>(null)
  const [worktreeNotice, setWorktreeNotice] = useState('')

  const handleLoad = (
    model: ParsedModel,
    sourceFile: string,
    gitInfo: GitInfo,
    registry: Record<string, ParsedModel>,
  ) => {
    setLoadedModel({ model, sourceFile, gitInfo, registry })
    setShowScan(false)
    setWorktreeNotice('')
  }

  const registrySize = loadedModel
    ? Object.keys(loadedModel.registry).length
    : 0

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
            registry: {registrySize} classes
          </span>
        )}
      </div>

      {/* Main 3-panel */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <div style={{ width: 280, flexShrink: 0, height: '100%' }}>
          <AgentPanel selectedBlockType={selectedBlockType} />
        </div>

        <div style={{ flex: 1, height: '100%', position: 'relative' }}>
          <FlowCanvas
            onNodeSelect={setSelectedBlockType}
            loadedModel={loadedModel}
            onWorktreeSaved={(path, branch) =>
              setWorktreeNotice(`✓ Saved to worktree: ${branch} (${path})`)
            }
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
