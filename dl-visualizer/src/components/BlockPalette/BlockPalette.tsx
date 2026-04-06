import { useState } from 'react'
import { BLOCKS, CATEGORIES } from '../../data/blocks'

export default function BlockPalette() {
  const [search, setSearch] = useState('')
  const [open, setOpen] = useState<Record<string, boolean>>({ io: true, conv: true, activation: true })

  const query = search.toLowerCase()
  const filtered = query
    ? BLOCKS.filter(b => b.label.toLowerCase().includes(query) || b.category.includes(query))
    : null

  const onDragStart = (e: React.DragEvent, type: string) => {
    e.dataTransfer.setData('application/dl-block-type', type)
    e.dataTransfer.effectAllowed = 'move'
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#161b27', borderLeft: '1px solid #2a3347' }}>
      <div style={{ padding: '10px 12px', borderBottom: '1px solid #2a3347', flexShrink: 0 }}>
        <p style={{ fontSize: 11, fontWeight: 700, color: '#64748b', letterSpacing: '0.08em', marginBottom: 6 }}>BLOCKS</p>
        <input
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

      <div style={{ flex: 1, overflowY: 'auto' }}>
        {filtered ? (
          <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 2 }}>
            {filtered.length === 0 && (
              <p style={{ fontSize: 12, color: '#475569', textAlign: 'center', padding: '16px 0' }}>No blocks found</p>
            )}
            {filtered.map(b => <DraggableBlock key={b.type} block={b} onDragStart={onDragStart} />)}
          </div>
        ) : (
          CATEGORIES.map(cat => {
            const blocks = BLOCKS.filter(b => b.category === cat.id)
            const isOpen = open[cat.id] ?? false
            return (
              <div key={cat.id} style={{ borderBottom: '1px solid #1e2535' }}>
                <button
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
                  <span style={{ fontSize: 10, color: '#475569' }}>{blocks.length}</span>
                  <span style={{ fontSize: 10, color: '#475569', marginLeft: 2 }}>{isOpen ? '▲' : '▼'}</span>
                </button>
                {isOpen && (
                  <div style={{ padding: '2px 8px 8px', display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {blocks.map(b => <DraggableBlock key={b.type} block={b} onDragStart={onDragStart} />)}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

function DraggableBlock({
  block,
  onDragStart,
}: {
  block: { type: string; label: string; color: string; formula?: string }
  onDragStart: (e: React.DragEvent, type: string) => void
}) {
  return (
    <div
      draggable
      onDragStart={e => onDragStart(e, block.type)}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '5px 8px', borderRadius: 6, cursor: 'grab',
        userSelect: 'none',
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
