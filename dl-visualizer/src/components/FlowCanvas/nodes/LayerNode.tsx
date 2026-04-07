import { useEffect, useState, type CSSProperties } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { getBlockDef, inferParamMeta, type ParamValue } from '../../../data/blocks'
import { CUSTOM_MODULE_TYPE } from '../../../lib/modelToGraph'
import type { LayerDataPreview } from '../../../lib/api'

export interface LayerNodeData {
  blockType: string
  params: Record<string, ParamValue>
  outputShape?: string
  shapeError?: boolean
  diagnosticSeverity?: 'error' | 'warning' | 'info'
  diagnosticReason?: string
  diagnosticCode?: string
  _customClassName?: string
  _attrName?: string
  _groupName?: string
  _groupColor?: string
  _isTopLevel?: boolean
  _isCollapsed?: boolean
  _subgraphSize?: number
  _isUtility?: boolean
  _expectedTerminal?: boolean
  _attrPath?: string
  _usesSelfState?: boolean
  _runtimeShapeLocked?: boolean
  _editedByAgent?: boolean
  _canDelete?: boolean
  _onDelete?: (nodeId: string) => void
  _canStartConnect?: boolean
  _onStartConnect?: (nodeId: string) => void
  _onUpdateParam?: (nodeId: string, key: string, value: ParamValue) => void
  _isPendingConnectSource?: boolean
  _dataPreview?: LayerDataPreview
  [key: string]: unknown
}

export default function LayerNode({ id, data, selected }: NodeProps) {
  const d = data as LayerNodeData
  const isCustom = d.blockType === CUSTOM_MODULE_TYPE
  const def = isCustom ? null : getBlockDef(d.blockType)

  const accentColor = d._groupColor
    ?? (isCustom ? colorForName(d._customClassName ?? d.blockType) : (def?.color ?? '#6366f1'))

  const headerLabel = isCustom
    ? (d._customClassName ?? d.blockType)
    : (def?.label ?? d.blockType)

  const shapeError = !!d.shapeError
  const hasShape = !!d.outputShape
  const isIO = !isCustom && def?.category === 'io'
  const diagnosticSeverity = d.diagnosticSeverity
  const borderColor = diagnosticSeverity === 'error'
    ? '#ef4444'
    : diagnosticSeverity === 'warning'
      ? '#f59e0b'
      : diagnosticSeverity === 'info'
        ? '#38bdf8'
        : selected
          ? '#818cf8'
          : '#2a3347'
  const isCollapsedModule = isCustom && !!d._isCollapsed
  const isAgentEdited = !!d._editedByAgent
  const agentRingStyle: CSSProperties = isAgentEdited
    ? {
        padding: 1.5,
        borderRadius: 12,
        background: 'linear-gradient(135deg, rgba(255,255,255,0.82) 0%, rgba(255,255,255,0.16) 36%, rgba(255,255,255,0.58) 72%, rgba(255,255,255,0.1) 100%)',
        boxShadow: selected
          ? '0 0 0 2px #818cf844, 0 0 18px rgba(255,255,255,0.22)'
          : '0 0 18px rgba(255,255,255,0.12)',
        transition: 'box-shadow 0.15s',
      }
    : {
        borderRadius: 10,
        boxShadow: selected ? '0 0 0 2px #818cf844' : 'none',
        transition: 'box-shadow 0.15s',
      }

  return (
    <div
      data-testid={`node-${id}`}
      data-diagnostic-severity={diagnosticSeverity ?? 'none'}
      data-diagnostic-code={d.diagnosticCode ?? 'none'}
      data-agent-edited={isAgentEdited ? 'true' : 'false'}
      style={{
        minWidth: isCollapsedModule ? 220 : 180,
        position: 'relative',
        ...agentRingStyle,
      }}
      title={d.diagnosticReason ?? undefined}
    >
      {/* 데이터 프리뷰: 카드 위에 floating */}
      {d._dataPreview && (
        <div style={{
          position: 'absolute',
          bottom: '100%',
          left: 0,
          right: 0,
          marginBottom: 6,
          zIndex: 10,
          background: '#0f1117ee',
          borderRadius: 8,
          border: `1px solid ${accentColor}44`,
          padding: (d._dataPreview.kind === 'spatial' && d.blockType === 'Input') ? '0' : '4px 6px',
          backdropFilter: 'blur(4px)',
          boxShadow: `0 -4px 16px ${accentColor}22`,
          overflow: 'hidden',
        }}>
          {/* Input 노드 + spatial(이미지) → 샘플 이미지를 크게 표시 */}
          {d._dataPreview.kind === 'spatial' && d.blockType === 'Input' ? (
            <div>
              <img
                src={`data:image/png;base64,${d._dataPreview.data}`}
                alt="sample input"
                style={{
                  width: '100%',
                  maxHeight: 100,
                  objectFit: 'cover',
                  display: 'block',
                  imageRendering: 'pixelated',
                  borderRadius: '8px 8px 0 0',
                }}
              />
              <div style={{
                padding: '3px 6px',
                fontSize: 9,
                fontFamily: 'monospace',
                color: '#94a3b8',
                background: '#0b0f1a',
                borderTop: `1px solid ${accentColor}22`,
              }}>
                sample&nbsp;<span style={{ color: '#475569' }}>[{d._dataPreview.shape.join(', ')}]</span>
              </div>
            </div>
          ) : (
            <DataPreviewPanel preview={d._dataPreview} accentColor={accentColor} />
          )}
        </div>
      )}
      <div
        style={{
          borderRadius: 10,
          border: `1.5px solid ${borderColor}`,
          background: '#1a2035',
          transition: 'border-color 0.15s',
          overflow: 'hidden',
        }}
      >
        {d._groupName && (
          <div style={{
            padding: '2px 10px',
            background: `${d._groupColor ?? '#818cf8'}18`,
            borderBottom: `1px solid ${d._groupColor ?? '#818cf8'}33`,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}>
            <div style={{ width: 5, height: 5, borderRadius: '50%', background: d._groupColor ?? '#818cf8', flexShrink: 0 }} />
            <span style={{ fontSize: 9, color: d._groupColor ?? '#94a3b8', fontFamily: 'monospace', opacity: 0.9 }}>
              {d._groupName}
            </span>
          </div>
        )}

        <div style={{
          padding: '6px 10px',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          background: `${accentColor}22`,
          borderBottom: `1px solid ${accentColor}44`,
        }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: accentColor, flexShrink: 0 }} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <span style={{
              fontSize: 12,
              fontWeight: 600,
              color: '#f1f5f9',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              display: 'block',
            }}>
              {d._attrName && d._attrName !== headerLabel ? d._attrName : headerLabel}
            </span>
            {d._attrName && d._attrName !== headerLabel && (
              <span style={{ fontSize: 9, color: '#64748b', display: 'block' }}>{headerLabel}</span>
            )}
          </div>
          {isCustom && (
            <span style={{
              fontSize: 8,
              padding: '1px 4px',
              borderRadius: 3,
              background: `${accentColor}28`,
              color: accentColor,
              border: `1px solid ${accentColor}44`,
              flexShrink: 0,
            }}>mod</span>
          )}
          {d._canStartConnect && (
            <button
              type="button"
              className="nodrag nopan"
              data-testid={`connect-node-${id}`}
              aria-label={`Connect from ${d._attrName ?? headerLabel}`}
              onMouseDown={(event) => {
                event.preventDefault()
                event.stopPropagation()
              }}
              onClick={(event) => {
                event.preventDefault()
                event.stopPropagation()
                d._onStartConnect?.(id)
              }}
              style={iconButtonStyle(
                d._isPendingConnectSource ? '#93c5fd' : `${accentColor}44`,
                d._isPendingConnectSource ? '#172554' : '#0f1117',
                d._isPendingConnectSource ? '#bfdbfe' : '#cbd5e1',
              )}
              title={d._isPendingConnectSource ? 'Connecting from this node' : 'Start connection from this node'}
            >
              ↗
            </button>
          )}
          {d._canDelete && (
            <button
              type="button"
              className="nodrag nopan"
              data-testid={`delete-node-${id}`}
              aria-label={`Delete ${d._attrName ?? headerLabel}`}
              onMouseDown={(event) => {
                event.preventDefault()
                event.stopPropagation()
              }}
              onClick={(event) => {
                event.preventDefault()
                event.stopPropagation()
                d._onDelete?.(id)
              }}
              style={iconButtonStyle(`${accentColor}44`, '#0f1117', '#cbd5e1')}
              title="Delete node"
            >
              ×
            </button>
          )}
        </div>

        <div style={{ padding: '6px 10px', display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 240, overflowY: 'auto' }}>
          {d.diagnosticReason && (
            <div style={{
              marginBottom: 4,
              padding: '5px 7px',
              borderRadius: 8,
              background: diagnosticSeverity === 'error'
                ? '#2b1212'
                : diagnosticSeverity === 'warning'
                  ? '#2d2411'
                  : '#102133',
              border: `1px solid ${borderColor}33`,
            }}>
              <div style={{ fontSize: 9, color: borderColor, fontWeight: 700, textTransform: 'uppercase' }}>
                {diagnosticSeverity ?? 'info'}
              </div>
              <div style={{ fontSize: 10, color: '#cbd5e1', marginTop: 2, lineHeight: 1.4 }}>
                {d.diagnosticReason}
              </div>
            </div>
          )}
          {Object.entries(d.params)
            .filter(([key]) => key !== 'class')
            .map(([key, value]) => (
              <div key={key} style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
                <span style={{ fontSize: 10, color: '#64748b', flexShrink: 0 }}>{key}</span>
                <ParamValueEditor
                  nodeId={id}
                  blockType={d.blockType}
                  paramKey={key}
                  value={value}
                  onCommit={d._onUpdateParam}
                />
              </div>
            ))}
          {isCollapsedModule && (
            <div style={{
              marginTop: 4,
              padding: '6px 8px',
              borderRadius: 8,
              background: `${accentColor}14`,
              border: `1px solid ${accentColor}33`,
            }}>
              <div style={{ fontSize: 10, color: '#cbd5e1', fontWeight: 600 }}>
                internals hidden
              </div>
              <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                {typeof d._subgraphSize === 'number' && d._subgraphSize > 0
                  ? `${d._subgraphSize} traced ops available in expanded view`
                  : 'switch to expanded view to inspect sub-ops'}
              </div>
            </div>
          )}
          {hasShape && (
            <div style={{ marginTop: 4, paddingTop: 4, borderTop: '1px solid #1e2535', textAlign: 'center' }}>
              <span style={{ fontSize: 10, fontFamily: 'monospace', color: shapeError ? '#fbbf24' : '#34d399' }}>
                {shapeError ? `⚠ ${d.outputShape}` : `→ ${d.outputShape}`}
              </span>
            </div>
          )}
        </div>
      </div>

      {(!isIO || def?.type === 'Output') && (
        <Handle
          type="target"
          position={Position.Left}
          data-testid={`target-handle-${id}`}
          style={{ width: 10, height: 10, border: '2px solid #475569', background: '#1a2035', borderRadius: '50%' }}
        />
      )}
      {(!isIO || def?.type === 'Input') && (
        <Handle
          type="source"
          position={Position.Right}
          data-testid={`source-handle-${id}`}
          style={{ width: 10, height: 10, border: '2px solid #475569', background: '#1a2035', borderRadius: '50%' }}
        />
      )}
    </div>
  )
}

interface ParamValueEditorProps {
  nodeId: string
  blockType: string
  paramKey: string
  value: ParamValue
  onCommit?: (nodeId: string, key: string, value: ParamValue) => void
}

function ParamValueEditor({ nodeId, blockType, paramKey, value, onCommit }: ParamValueEditorProps) {
  const meta = inferParamMeta(blockType, paramKey, value)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(String(value))

  useEffect(() => {
    if (!editing) setDraft(String(value))
  }, [editing, value])

  const commit = () => {
    onCommit?.(nodeId, paramKey, parseDraftValue(draft, meta.kind))
    setEditing(false)
  }

  if (editing && meta.kind === 'bool') {
    return (
      <input
        className="nodrag nopan"
        type="checkbox"
        checked={draft === 'true'}
        onChange={(event) => {
          const next = String(event.target.checked)
          setDraft(next)
          onCommit?.(nodeId, paramKey, parseDraftValue(next, meta.kind))
        }}
        onBlur={() => setEditing(false)}
      />
    )
  }

  if (editing) {
    return (
      <input
        className="nodrag nopan"
        data-testid={`param-input-${nodeId}-${paramKey}`}
        autoFocus
        type={meta.kind === 'int' || meta.kind === 'float' ? 'number' : 'text'}
        step={meta.step}
        min={meta.min}
        max={meta.max}
        placeholder={meta.placeholder}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === 'Enter') commit()
          if (event.key === 'Escape') {
            setDraft(String(value))
            setEditing(false)
          }
        }}
        style={inputStyle}
      />
    )
  }

  return (
    <button
      type="button"
      className="nodrag nopan"
      data-testid={`param-value-${nodeId}-${paramKey}`}
      onMouseDown={(event) => {
        event.preventDefault()
        event.stopPropagation()
      }}
      onClick={(event) => {
        event.preventDefault()
        event.stopPropagation()
        setEditing(true)
      }}
      style={{
        ...inputStyle,
        width: 'auto',
        minWidth: 72,
        maxWidth: 120,
        cursor: 'pointer',
      }}
      title="Click to edit"
    >
      {String(value)}
    </button>
  )
}

function parseDraftValue(draft: string, kind: ReturnType<typeof inferParamMeta>['kind']): ParamValue {
  const trimmed = draft.trim()
  if (kind === 'bool') return trimmed === 'true'
  if (kind === 'int') {
    const parsed = Number.parseInt(trimmed || '0', 10)
    return Number.isFinite(parsed) ? parsed : 0
  }
  if (kind === 'float') {
    const parsed = Number.parseFloat(trimmed || '0')
    return Number.isFinite(parsed) ? parsed : 0
  }
  return trimmed
}

function iconButtonStyle(border: string, background: string, color: string) {
  return {
    width: 18,
    height: 18,
    borderRadius: 999,
    border: `1px solid ${border}`,
    background,
    color,
    fontSize: 11,
    lineHeight: 1,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    flexShrink: 0,
  } satisfies CSSProperties
}

const inputStyle: CSSProperties = {
  width: 96,
  padding: '2px 6px',
  borderRadius: 6,
  border: '1px solid #475569',
  background: '#0f172a',
  color: '#e2e8f0',
  fontSize: 11,
  fontFamily: 'monospace',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
}

function colorForName(name: string): string {
  const colors = ['#c084fc', '#fb923c', '#34d399', '#f472b6', '#60a5fa', '#facc15', '#a78bfa', '#4ade80']
  let hash = 0
  for (let i = 0; i < name.length; i += 1) hash = (hash * 31 + name.charCodeAt(i)) & 0xffff
  return colors[hash % colors.length]
}

// ── Data Preview Panel ───────────────────────────────────────────────────────

interface DataPreviewPanelProps {
  preview: LayerDataPreview
  accentColor: string
}

function VectorBarChart({ values }: { values: number[] }) {
  if (values.length === 0) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  // 노드 폭(180px)에 고정 — 값이 많으면 line chart로
  const W = 164
  const H = 36
  const n = values.length

  if (n <= 32) {
    // bar chart
    const barW = W / n
    return (
      <svg width={W} height={H} style={{ display: 'block', width: '100%', height: H }}>
        {values.map((v, i) => {
          const barH = Math.max(1, Math.round(((v - min) / range) * (H - 4)))
          return (
            <rect
              key={i}
              x={i * barW + 0.5}
              y={H - barH - 2}
              width={Math.max(1, barW - 1)}
              height={barH}
              fill="#6366f1"
              opacity={0.8}
              rx={1}
            />
          )
        })}
      </svg>
    )
  }

  // line chart (값이 많을 때)
  const pts = values.map((v, i) => {
    const x = (i / (n - 1)) * W
    const y = H - 2 - ((v - min) / range) * (H - 4)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return (
    <svg width={W} height={H} style={{ display: 'block', width: '100%', height: H }}>
      <polyline points={pts} fill="none" stroke="#818cf8" strokeWidth={1.2} strokeLinejoin="round" />
      {/* zero line */}
      {min < 0 && max > 0 && (
        <line
          x1={0} y1={H - 2 - ((-min) / range) * (H - 4)}
          x2={W} y2={H - 2 - ((-min) / range) * (H - 4)}
          stroke="#334155" strokeWidth={0.8} strokeDasharray="3 3"
        />
      )}
    </svg>
  )
}

function DataPreviewPanel({ preview, accentColor }: DataPreviewPanelProps) {
  const shapeLabel = `[${preview.shape.join(', ')}]`
  return (
    <div style={{
      marginTop: 4,
      borderTop: `1px solid ${accentColor}22`,
      paddingTop: 4,
    }}>
      <div style={{ fontSize: 9, color: '#475569', marginBottom: 3, fontFamily: 'monospace' }}>
        ▸ data&nbsp;
        <span style={{ color: '#334155' }}>{shapeLabel}</span>
      </div>
      {(preview.kind === 'spatial' || preview.kind === 'sequence') && (
        <img
          src={`data:image/png;base64,${preview.data}`}
          alt={`tensor preview ${shapeLabel}`}
          style={{
            width: '100%',
            maxHeight: 72,
            objectFit: 'cover',
            borderRadius: 4,
            imageRendering: 'pixelated',
            display: 'block',
          }}
        />
      )}
      {preview.kind === 'vector' && (
        <VectorBarChart values={preview.values} />
      )}
    </div>
  )
}
