import { useState, useEffect, useRef } from 'react'
import { BLOCK_MAP } from '../../data/blocks'

interface Message {
  role: 'agent' | 'user'
  text: string
}

interface Props {
  selectedBlockType: string | null
}

export default function AgentPanel({ selectedBlockType }: Props) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'agent',
      text: '안녕하세요! DL 아키텍처 시각화 툴입니다.\n\n오른쪽 팔레트에서 블록을 캔버스로 드래그하세요. 블록을 연결하면 shape이 자동 계산됩니다.\n\n블록을 클릭하면 설명해드릴게요.',
    },
  ])
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!selectedBlockType) return
    const def = BLOCK_MAP[selectedBlockType]
    if (!def) return
    setMessages(prev => [...prev, {
      role: 'agent',
      text: `**${def.label}**\n\n${def.description}${def.formula ? `\n\n수식: ${def.formula}` : ''}`,
    }])
  }, [selectedBlockType])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = () => {
    if (!input.trim()) return
    setMessages(prev => [...prev, { role: 'user', text: input.trim() }])
    const q = input.toLowerCase()
    setInput('')

    let reply = '블록을 클릭하면 해당 레이어에 대해 자세히 설명드릴 수 있습니다.'
    if (q.includes('attention'))
      reply = 'Attention은 토큰 간 관계를 학습합니다.\n\n수식: softmax(QKᵀ/√d)·V\n\n• Self-Attention: Q=K=V\n• Cross-Attention: Q≠K,V\n• Multi-Head: 여러 attention 병렬 계산'
    else if (q.includes('conv'))
      reply = 'Convolution은 작은 필터를 슬라이딩해 특징을 추출합니다.\n\n출력 크기: H\' = (H + 2p - k) / s + 1\n\n• kernel: 필터 크기\n• stride: 이동 간격\n• padding: 경계 처리'
    else if (q.includes('pool'))
      reply = 'Pooling은 공간 크기를 줄입니다.\n\n• MaxPool: 최댓값 → 강한 특징 보존\n• AvgPool: 평균값\n• GlobalAvgPool: (B,C,H,W)→(B,C)'
    else if (q.includes('norm'))
      reply = '• BatchNorm: 배치 단위 정규화\n• LayerNorm: 특징 차원 정규화 (Transformer 표준)\n• GroupNorm: 채널 그룹 단위\n• RMSNorm: LLaMA 계열 사용'
    else if (q.includes('relu') || q.includes('gelu') || q.includes('활성'))
      reply = '활성화 함수는 비선형성을 추가합니다.\n\n• ReLU: max(0,x) — 가장 기본\n• GELU: x·Φ(x) — Transformer 표준\n• SiLU: x·σ(x) — LLaMA 사용\n• LeakyReLU: Dying ReLU 방지'
    else if (q.includes('코드') || q.includes('pytorch'))
      reply = '상단의 "Generate PyTorch Code" 버튼을 누르면 현재 그래프를 PyTorch 코드로 변환합니다!'
    else if (q.includes('lstm') || q.includes('gru'))
      reply = '순환 신경망 계열입니다.\n\n• RNN: 기본 구조, 기울기 소실 문제\n• LSTM: 게이트로 장거리 의존성 해결\n• GRU: LSTM보다 가볍고 빠름\n• Bidirectional: 양방향 처리'

    setTimeout(() => setMessages(prev => [...prev, { role: 'agent', text: reply }]), 400)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#161b27', borderRight: '1px solid #2a3347' }}>
      {/* Header */}
      <div style={{ padding: '10px 12px', borderBottom: '1px solid #2a3347', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#34d399' }} />
          <span style={{ fontSize: 12, fontWeight: 600, color: '#cbd5e1' }}>AI Agent</span>
        </div>
        <p style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>Quick presets:</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {['Simple CNN', 'Transformer', 'ResNet'].map(label => (
            <span key={label} style={{
              fontSize: 11, padding: '2px 8px', borderRadius: 4,
              border: '1px solid #334155', color: '#94a3b8', cursor: 'pointer',
              background: '#1e2535'
            }}>{label}</span>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div style={{
              maxWidth: '88%',
              padding: '8px 12px',
              borderRadius: m.role === 'user' ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
              background: m.role === 'user' ? '#4f46e5' : '#1e2535',
              border: m.role === 'agent' ? '1px solid #2a3347' : 'none',
              fontSize: 12,
              lineHeight: 1.6,
              color: '#e2e8f0',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}>
              {m.text}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{ padding: '10px 12px', borderTop: '1px solid #2a3347', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 6 }}>
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
            placeholder="Ask about any layer…"
            style={{
              flex: 1, background: '#0f1117', border: '1px solid #334155',
              borderRadius: 6, padding: '6px 10px', fontSize: 12,
              color: '#e2e8f0', outline: 'none',
            }}
          />
          <button onClick={send} style={{
            padding: '6px 12px', borderRadius: 6, background: '#4f46e5',
            color: 'white', fontSize: 13, border: 'none', cursor: 'pointer',
          }}>↑</button>
        </div>
      </div>
    </div>
  )
}
