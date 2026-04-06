import { Handle, Position, NodeProps } from '@xyflow/react'
import { BLOCK_MAP } from '../../../data/blocks'
import { CUSTOM_MODULE_TYPE } from '../../../lib/modelToGraph'

export interface LayerNodeData {
  blockType: string
  params: Record<string, string | number>
  outputShape?: string
  shapeError?: boolean
  _customClassName?: string
  _attrName?: string
  _groupName?: string    // parent sub-module attr name (e.g. 'nutrition_head')
  _groupColor?: string   // accent color for this group
  _isTopLevel?: boolean
  _isCollapsed?: boolean
  _subgraphSize?: number
  [key: string]: unknown
}

export default function LayerNode({ data, selected }: NodeProps) {
  const d = data as LayerNodeData
  const isCustom = d.blockType === CUSTOM_MODULE_TYPE
  const def = isCustom ? null : BLOCK_MAP[d.blockType]

  const accentColor = d._groupColor
    ?? (isCustom ? colorForName(d._customClassName ?? d.blockType) : (def?.color ?? '#6366f1'))

  const headerLabel = isCustom
    ? (d._customClassName ?? d.blockType)
    : (def?.label ?? d.blockType)

  const shapeError = !!d.shapeError
  const hasShape = !!d.outputShape
  const isIO = !isCustom && def?.category === 'io'
  const borderColor = shapeError ? '#ef4444' : selected ? '#818cf8' : '#2a3347'
  const isCollapsedModule = isCustom && !!d._isCollapsed

  return (
    <div style={{
      minWidth: isCollapsedModule ? 220 : 160,
      borderRadius: 10,
      border: `1.5px solid ${borderColor}`,
      background: '#1a2035',
      boxShadow: selected ? '0 0 0 2px #818cf844' : 'none',
      transition: 'border-color 0.15s',
      overflow: 'hidden',
    }}>
      {/* Group label banner — shown when inside a sub-module */}
      {d._groupName && (
        <div style={{
          padding: '2px 10px',
          background: `${d._groupColor ?? '#818cf8'}18`,
          borderBottom: `1px solid ${d._groupColor ?? '#818cf8'}33`,
          display: 'flex', alignItems: 'center', gap: 4,
        }}>
          <div style={{ width: 5, height: 5, borderRadius: '50%', background: d._groupColor ?? '#818cf8', flexShrink: 0 }} />
          <span style={{ fontSize: 9, color: d._groupColor ?? '#94a3b8', fontFamily: 'monospace', opacity: 0.9 }}>
            {d._groupName}
          </span>
        </div>
      )}

      {/* Header */}
      <div style={{
        padding: '6px 10px',
        display: 'flex', alignItems: 'center', gap: 6,
        background: `${accentColor}22`,
        borderBottom: `1px solid ${accentColor}44`,
      }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: accentColor, flexShrink: 0 }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <span style={{
            fontSize: 12, fontWeight: 600, color: '#f1f5f9',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block',
          }}>
            {d._attrName && d._attrName !== headerLabel ? d._attrName : headerLabel}
          </span>
          {/* Show class type as subtitle when attr name differs */}
          {d._attrName && d._attrName !== headerLabel && (
            <span style={{ fontSize: 9, color: '#64748b', display: 'block' }}>{headerLabel}</span>
          )}
        </div>
        {isCustom && (
          <span style={{
            fontSize: 8, padding: '1px 4px', borderRadius: 3,
            background: `${accentColor}28`, color: accentColor, border: `1px solid ${accentColor}44`,
            flexShrink: 0,
          }}>mod</span>
        )}
      </div>

      {/* Body */}
      <div style={{ padding: '6px 10px', display: 'flex', flexDirection: 'column', gap: 3 }}>
        {Object.entries(d.params)
          .filter(([k]) => k !== 'class')
          .map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
              <span style={{ fontSize: 10, color: '#64748b' }}>{k}</span>
              <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#cbd5e1', maxWidth: 90, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {String(v)}
              </span>
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
            <span style={{ fontSize: 10, fontFamily: 'monospace', color: shapeError ? '#f87171' : '#34d399' }}>
              {shapeError ? '⚠ check connection' : `→ ${d.outputShape}`}
            </span>
          </div>
        )}
      </div>

      {/* Handles */}
      {(!isIO || def?.type === 'Output') && (
        <Handle type="target" position={Position.Left}
          style={{ width: 10, height: 10, border: '2px solid #475569', background: '#1a2035', borderRadius: '50%' }}
        />
      )}
      {(!isIO || def?.type === 'Input') && (
        <Handle type="source" position={Position.Right}
          style={{ width: 10, height: 10, border: '2px solid #475569', background: '#1a2035', borderRadius: '50%' }}
        />
      )}
    </div>
  )
}

function colorForName(name: string): string {
  const COLORS = ['#c084fc','#fb923c','#34d399','#f472b6','#60a5fa','#facc15','#a78bfa','#4ade80']
  let h = 0
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffff
  return COLORS[h % COLORS.length]
}
