import { useEffect, useState } from 'react'
import { getBlockCatalog, getCategoryCatalog } from '../../data/blocks'

type PointerDragState = {
  blockType: string
  label: string
  x: number
  y: number
}

function normalizeSearchToken(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '')
}

export default function BlockPalette() {
  const [search, setSearch] = useState('')
  const [open, setOpen] = useState<Record<string, boolean>>({ io: true, conv: true, activation: true })
  const [renderNonce, setRenderNonce] = useState(0)
  const [dragState, setDragState] = useState<PointerDragState | null>(null)
  const blocks = getBlockCatalog()
  const categories = getCategoryCatalog()

  useEffect(() => {
    const resetPalette = () => {
      setRenderNonce((current) => current + 1)
      setDragState(null)
    }
    window.addEventListener('dl-viz-reset-palette', resetPalette)
    document.addEventListener('mouseup', resetPalette, true)
    return () => {
      window.removeEventListener('dl-viz-reset-palette', resetPalette)
      document.removeEventListener('mouseup', resetPalette, true)
    }
  }, [])

  const query = search.toLowerCase()
  const normalizedQuery = normalizeSearchToken(search)
  const filtered = query
    ? blocks.filter((b) => {
      const normalizedLabel = normalizeSearchToken(b.label)
      const normalizedCategory = normalizeSearchToken(b.category)
      const normalizedType = normalizeSearchToken(b.type)
      return b.label.toLowerCase().includes(query)
        || b.category.includes(query)
        || b.type.toLowerCase().includes(query)
        || normalizedLabel.includes(normalizedQuery)
        || normalizedCategory.includes(normalizedQuery)
        || normalizedType.includes(normalizedQuery)
    })
    : null

  const handlePointerStart = (event: React.PointerEvent, block: { type: string; label: string }) => {
    event.preventDefault()
    const startX = event.clientX
    const startY = event.clientY
    setDragState({ blockType: block.type, label: block.label, x: startX, y: startY })

    const handlePointerMove = (moveEvent: PointerEvent) => {
      setDragState((current) => current
        ? { ...current, x: moveEvent.clientX, y: moveEvent.clientY }
        : current)
    }

    const handlePointerEnd = (endEvent: PointerEvent) => {
      const canvas = document.elementFromPoint(endEvent.clientX, endEvent.clientY)?.closest('[data-testid="flow-canvas"]')
      if (canvas) {
        window.dispatchEvent(new CustomEvent('dl-viz-pointer-drop-block', {
          detail: {
            blockType: block.type,
            clientX: endEvent.clientX,
            clientY: endEvent.clientY,
          },
        }))
      }
      setDragState(null)
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerEnd)
      window.removeEventListener('pointercancel', handlePointerEnd)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerEnd)
    window.addEventListener('pointercancel', handlePointerEnd)
  }

  return (
    <div key={renderNonce} style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, background: '#161b27', borderLeft: '1px solid #2a3347' }}>
      <div style={{ padding: '10px 12px', borderBottom: '1px solid #2a3347', flexShrink: 0 }}>
        <p style={{ fontSize: 11, fontWeight: 700, color: '#64748b', letterSpacing: '0.08em', marginBottom: 6 }}>BLOCKS</p>
        <input
          data-testid="block-search-input"
          type="text"
          placeholder="Search blocks…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{
            width: '100%', background: '#0f1117', border: '1px solid #334155',
            borderRadius: 6, padding: '5px 8px', fontSize: 12,
            color: '#e2e8f0', outline: 'none', boxSizing: 'border-box',
          }}
        />
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', overflowX: 'hidden' }}>
        {filtered ? (
          <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 2 }}>
            {filtered.length === 0 && (
              <p style={{ fontSize: 12, color: '#475569', textAlign: 'center', padding: '16px 0' }}>No blocks found</p>
            )}
            {filtered.map(b => <DraggableBlock key={b.type} block={b} onPointerStart={handlePointerStart} isDragging={dragState?.blockType === b.type} />)}
          </div>
        ) : (
          categories.map(cat => {
            const categoryBlocks = blocks.filter(b => b.category === cat.id)
            const isOpen = open[cat.id] ?? false
            return (
              <div key={cat.id} style={{ borderBottom: '1px solid #1e2535' }}>
                <button
                  data-testid={`category-toggle-${cat.id}`}
                  onClick={() => setOpen(o => ({ ...o, [cat.id]: !isOpen }))}
                  style={{
                    width: '100%', display: 'flex', alignItems: 'center', gap: 6,
                    padding: '7px 12px', background: 'transparent', border: 'none',
                    cursor: 'pointer', textAlign: 'left',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = '#1e2535')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                >
                  <span style={{ fontSize: 14, lineHeight: 1 }}>{cat.icon}</span>
                  <span style={{ fontSize: 12, fontWeight: 500, color: '#94a3b8', flex: 1 }}>{cat.label}</span>
                  <span style={{ fontSize: 10, color: '#475569' }}>{categoryBlocks.length}</span>
                  <span style={{ fontSize: 10, color: '#475569', marginLeft: 2 }}>{isOpen ? '▲' : '▼'}</span>
                </button>
                {isOpen && (
                  <div style={{ padding: '2px 8px 8px', display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {categoryBlocks.map(b => <DraggableBlock key={b.type} block={b} onPointerStart={handlePointerStart} isDragging={dragState?.blockType === b.type} />)}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>

      {dragState && (
        <div
          style={{
            position: 'fixed',
            left: dragState.x + 14,
            top: dragState.y + 14,
            zIndex: 1000,
            pointerEvents: 'none',
            padding: '6px 10px',
            borderRadius: 8,
            background: '#111827',
            border: '1px solid #334155',
            color: '#e2e8f0',
            fontSize: 12,
            boxShadow: '0 8px 30px rgba(0, 0, 0, 0.35)',
            whiteSpace: 'nowrap',
          }}
        >
          {dragState.label}
        </div>
      )}
    </div>
  )
}

function DraggableBlock({
  block,
  onPointerStart,
  isDragging,
}: {
  block: { type: string; label: string; color: string; formula?: string }
  onPointerStart: (event: React.PointerEvent, block: { type: string; label: string }) => void
  isDragging?: boolean
}) {
  return (
    <div
      data-testid={`block-${block.type}`}
      onPointerDown={(event) => onPointerStart(event, { type: block.type, label: block.label })}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '5px 8px', borderRadius: 6, cursor: isDragging ? 'grabbing' : 'grab',
        userSelect: 'none', touchAction: 'none',
        opacity: isDragging ? 0.55 : 1,
      }}
      onMouseEnter={e => (e.currentTarget.style.background = '#1e2535')}
      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
    >
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: block.color, flexShrink: 0 }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ fontSize: 12, fontWeight: 500, color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {block.label}
        </p>
        {block.formula && (
          <p style={{ fontSize: 10, color: '#64748b', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {block.formula}
          </p>
        )}
      </div>
    </div>
  )
}
