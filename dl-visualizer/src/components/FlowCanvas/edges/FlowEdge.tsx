import { getBezierPath } from '@xyflow/react'
import type { EdgeProps } from '@xyflow/react'

export interface FlowEdgeData {
  shapeLabel?: string      // e.g. "(B, 16, 64, 64)"
  hasData?: boolean        // trace 데이터가 있을 때 true
  tensorSample?: number[]  // 실제 텐서 샘플 값 (최대 5개)
  [key: string]: unknown
}

/**
 * 데이터 흐름 커스텀 엣지.
 * - hasData=true 이면 대시 흐르는 strokeDashoffset 애니메이션 + shape 라벨
 * - 기본은 보라색 스타일
 */
export default function FlowEdge({
  id,
  sourceX, sourceY,
  targetX, targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
  markerEnd,
}: EdgeProps) {
  const d = (data ?? {}) as FlowEdgeData
  const hasData = !!d.hasData

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX, sourceY, sourcePosition,
    targetX, targetY, targetPosition,
  })

  const strokeColor = selected ? '#a5b4fc' : hasData ? '#818cf8' : '#4a556a'
  const strokeWidth = hasData ? 2.5 : 1.8

  // 대시 애니리: 가상의 길이를 주기적으로 이동
  const dashLen = 8
  const gapLen = 16

  return (
    <g>
      {/* 기본 선 */}
      <path
        d={edgePath}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeOpacity={hasData ? 0.4 : 0.9}
        markerEnd={markerEnd}
        style={{ filter: hasData ? `drop-shadow(0 0 2px ${strokeColor}66)` : 'none' }}
      />

      {/* 데이터 흐름 대시 선 (CSS 애니메이션) */}
      {hasData && (
        <>
          <style>{`
            @keyframes flow-dash-${id} {
              from { stroke-dashoffset: ${(dashLen + gapLen) * 3}; }
              to   { stroke-dashoffset: 0; }
            }
          `}</style>
          <path
            d={edgePath}
            fill="none"
            stroke="#a5b4fc"
            strokeWidth={2.5}
            strokeDasharray={`${dashLen} ${gapLen}`}
            strokeLinecap="round"
            style={{
              animation: `flow-dash-${id} 1.4s linear infinite`,
              filter: `drop-shadow(0 0 3px #818cf8aa)`,
            }}
          />
        </>
      )}

      {/* 실제 텐서 샘플 값 라벨 (우선) 또는 shape 라벨 */}
      {hasData && (d.tensorSample || d.shapeLabel) && (
        <foreignObject
          x={labelX - 64}
          y={labelY - 18}
          width={128}
          height={d.tensorSample ? 32 : 22}
          style={{ pointerEvents: 'none', overflow: 'visible' }}
        >
          {/* @ts-expect-error xmlns needed for foreignObject in SVG */}
          <div xmlns="http://www.w3.org/1999/xhtml" style={{
            background: '#0d1120ee',
            border: '1px solid #818cf866',
            borderRadius: 5,
            padding: '3px 7px',
            fontSize: 9,
            fontFamily: 'monospace',
            color: '#a5b4fc',
            textAlign: 'center',
            whiteSpace: 'nowrap',
            backdropFilter: 'blur(6px)',
            boxShadow: '0 2px 8px #00000088',
          }}>
            {d.tensorSample ? (
              <>
                <div style={{ fontSize: 8, color: '#64748b', marginBottom: 1 }}>{d.shapeLabel ?? ''}</div>
                <div style={{ color: '#c7d2fe', letterSpacing: '0.03em' }}>
                  {d.tensorSample.map(v => v.toFixed(2)).join(', ')}&hellip;
                </div>
              </>
            ) : d.shapeLabel}
          </div>
        </foreignObject>
      )}
    </g>
  )
}
