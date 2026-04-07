import { useEffect, useRef, useState } from 'react'
import { getBlockCatalog, getBlockDef } from '../../data/blocks'
import {
  runAgentChat,
  type AgentCanvasAction,
  type AgentGraphSnapshot,
  type LoadedModelPayload,
} from '../../lib/api'

interface Message {
  role: 'agent' | 'user' | 'system'
  text: string
}

interface Props {
  selectedBlockType: string | null
  graphSnapshot: AgentGraphSnapshot
  loadedModel?: LoadedModelPayload | null
  onExecuteActions: (actions: AgentCanvasAction[]) => Promise<{
    applied: number
    warnings: string[]
    snapshot: AgentGraphSnapshot
  }>
}

const QUICK_PROMPTS = [
  {
    label: 'Simple CNN',
    prompt: '빈 캔버스에 Input, Conv2D, ReLU, Conv2D, Output을 순서대로 배치하고 연결해줘. 첫 Conv2D는 in_ch=3 out_ch=32 kernel=3 padding=1, 둘째 Conv2D는 in_ch=32 out_ch=64 kernel=3 padding=1로 설정하고 마지막에 자동 정렬해줘.',
  },
  {
    label: 'Activation Swap',
    prompt: '현재 그래프에서 중간 활성화 노드를 찾아 GELU로 바꾸고, 앞뒤 연결은 유지한 채 자동 정렬해줘.',
  },
  {
    label: 'Top Align Input',
    prompt: '현재 그래프에서 Input 노드를 가장 위쪽으로 옮기고 전체 배치를 정리해줘.',
  },
]

const EDIT_INTENT_PATTERN = /(배치|추가|바꿔|교체|삭제|연결|옮겨|정렬|쌓|변경|layout|move|replace|add|delete|connect|insert|swap|align|change|stack|more|modify)/i

// 자연어 별칭 → blockType 매핑 (대소문자 무관)
const BLOCK_ALIASES: Record<string, string> = {
  'conv':         'Conv2D',
  'convolution':  'Conv2D',
  'conv2d':       'Conv2D',
  'conv1d':       'Conv1D',
  'conv3d':       'Conv3D',
  'leaky':        'LeakyReLU',
  'leakyrelu':    'LeakyReLU',
  'leaky relu':   'LeakyReLU',
  'elu':          'ELU',
  'gelu':         'GELU',
  'selu':         'SELU',
  'relu':         'ReLU',
  'sigmoid':      'Sigmoid',
  'tanh':         'Tanh',
  'bn':           'BatchNorm2D',
  'batchnorm':    'BatchNorm2D',
  'batch norm':   'BatchNorm2D',
  'linear':       'Linear',
  'fc':           'Linear',
  'dropout':      'Dropout',
  'maxpool':      'MaxPool2D',
  'avgpool':      'AvgPool2D',
  'flatten':      'Flatten',
  'lstm':         'LSTM',
  'gru':          'GRU',
  'transformer':  'TransformerEncoderLayer',
  'attention':    'MultiheadAttention',
  'embedding':    'Embedding',
  'layernorm':    'LayerNorm',
  'groupnorm':    'GroupNorm',
  'upsample':     'Upsample',
  'gap':          'GlobalAvgPool',
}

function normalizeBlockIdentity(value: string) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '')
}

function selectAgentAvailableBlocks(
  prompt: string,
  graphSnapshot: AgentGraphSnapshot,
  selectedBlockType: string | null,
) {
  const blockCatalog = getBlockCatalog()
  const loweredPrompt = prompt.toLowerCase()
  const graphTypes = new Set(graphSnapshot.nodes.map((node) => node.blockType))
  if (selectedBlockType) graphTypes.add(selectedBlockType)

  // alias 매핑으로 프롬프트에서 언급된 blockType 수집
  const aliasMatchedTypes = new Set<string>()
  for (const [alias, blockType] of Object.entries(BLOCK_ALIASES)) {
    if (loweredPrompt.includes(alias.toLowerCase())) {
      aliasMatchedTypes.add(blockType)
    }
  }

  // 수정 의도가 있거나 그래프가 비어 있지 않으면 관련 블록 전달
  const hasEditIntent = EDIT_INTENT_PATTERN.test(prompt)

  const mentionedBlocks = blockCatalog.filter((block) => {
    const typeMatch = loweredPrompt.includes(block.type.toLowerCase())
    const labelMatch = loweredPrompt.includes(block.label.toLowerCase())
    const aliasMatch = aliasMatchedTypes.has(block.type)
    const graphMatch = graphTypes.has(block.type)
    return typeMatch || labelMatch || aliasMatch || (hasEditIntent && graphMatch)
  })

  // 아무것도 매칭 안 됐고 수정 의도가 있으면 → 전체 카탈로그 전달 (에이전트가 직접 판단)
  const candidates = mentionedBlocks.length > 0
    ? mentionedBlocks
    : hasEditIntent
      ? blockCatalog
      : []

  const deduped = new Map<string, (typeof candidates)[number]>()
  for (const block of candidates) {
    const key = normalizeBlockIdentity(block.type)
    if (!deduped.has(key)) {
      deduped.set(key, block)
    }
  }

  return [...deduped.values()].map((block) => ({
    type: block.type,
    label: block.label,
    category: block.category,
    defaultParams: block.defaultParams,
  }))
}

export default function AgentPanel({ selectedBlockType, graphSnapshot, loadedModel, onExecuteActions }: Props) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'agent',
      text: 'AI 에이전트가 질문 응답과 캔버스 조작을 같이 처리합니다.\n\n예: "Conv2D 두 개와 GELU를 배치해줘", "가운데 ReLU를 LeakyReLU로 바꿔줘", "입력 노드를 맨 위로 옮겨줘".',
    },
  ])
  const [input, setInput] = useState('')
  const [running, setRunning] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!selectedBlockType) return
    const def = getBlockDef(selectedBlockType)
    if (!def) return
    setMessages((prev) => {
      const next = [...prev]
      const summary = `선택된 노드: **${def.label}**\n\n${def.description}${def.formula ? `\n\n수식: ${def.formula}` : ''}`
      if (next[next.length - 1]?.text === summary) return prev
      return [...prev, { role: 'system', text: summary }]
    })
  }, [selectedBlockType])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, running])

  const submitPrompt = async (prompt: string) => {
    const trimmed = prompt.trim()
    if (!trimmed || running) return

    setMessages((prev) => [...prev, { role: 'user', text: trimmed }])
    setInput('')
    setRunning(true)

    try {
      const availableBlocks = selectAgentAvailableBlocks(trimmed, graphSnapshot, selectedBlockType)
      const response = await runAgentChat({
        prompt: trimmed,
        selectedBlockType,
        graph: graphSnapshot,
        availableBlocks,
        loadedModel: loadedModel ? {
          name: loadedModel.model.name,
          sourceFile: loadedModel.sourceFile,
          traceMode: loadedModel.traceMode,
          exactness: loadedModel.exactness,
        } : null,
      })

      setMessages((prev) => [...prev, { role: 'agent', text: response.reply }])

      if (response.actions.length > 0) {
        setMessages((prev) => [...prev, {
          role: 'system',
          text: `실행 계획: ${formatActionSummary(response.actions)}`,
        }])
        const result = await onExecuteActions(response.actions)
        const warningText = result.warnings.length > 0
          ? `\n경고: ${result.warnings.join(' | ')}`
          : ''
        setMessages((prev) => [...prev, {
          role: 'system',
          text: `적용 완료: ${result.applied}개 액션${warningText}`,
        }])
      }
    } catch (cause: unknown) {
      setMessages((prev) => [...prev, {
        role: 'system',
        text: `에이전트 실행 실패: ${cause instanceof Error ? cause.message : String(cause)}`,
      }])
    } finally {
      setRunning(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#161b27', borderRight: '1px solid #2a3347' }}>
      <div style={{ padding: '10px 12px', borderBottom: '1px solid #2a3347', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: running ? '#f59e0b' : '#34d399' }} />
          <span style={{ fontSize: 12, fontWeight: 600, color: '#cbd5e1' }}>AI Agent</span>
          <span style={{ fontSize: 10, color: '#64748b' }}>
            {running ? 'planning / applying…' : `${graphSnapshot.nodes.length} nodes · ${graphSnapshot.edges.length} edges`}
          </span>
        </div>
        <p style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>Quick prompts:</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {QUICK_PROMPTS.map((item) => (
            <button
              key={item.label}
              type="button"
              onClick={() => submitPrompt(item.prompt)}
              disabled={running}
              style={{
                fontSize: 11,
                padding: '2px 8px',
                borderRadius: 4,
                border: '1px solid #334155',
                color: '#94a3b8',
                cursor: running ? 'default' : 'pointer',
                background: '#1e2535',
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} style={{ display: 'flex', justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div
              data-testid={message.role === 'agent' ? 'agent-message' : message.role === 'system' ? 'agent-system-message' : 'agent-user-message'}
              style={{
                maxWidth: '92%',
                padding: '8px 12px',
                borderRadius: message.role === 'user' ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
                background: message.role === 'user' ? '#4f46e5' : message.role === 'system' ? '#0f172a' : '#1e2535',
                border: message.role !== 'user' ? '1px solid #2a3347' : 'none',
                fontSize: 12,
                lineHeight: 1.6,
                color: '#e2e8f0',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {message.text}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div style={{ padding: '10px 12px', borderTop: '1px solid #2a3347', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 6 }}>
          <input
            data-testid="agent-chat-input"
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') void submitPrompt(input)
            }}
            placeholder="Ask or command the agent…"
            disabled={running}
            style={{
              flex: 1,
              background: '#0f1117',
              border: '1px solid #334155',
              borderRadius: 6,
              padding: '6px 10px',
              fontSize: 12,
              color: '#e2e8f0',
              outline: 'none',
            }}
          />
          <button
            data-testid="agent-chat-send"
            onClick={() => void submitPrompt(input)}
            disabled={running}
            style={{
              padding: '6px 12px',
              borderRadius: 6,
              background: running ? '#334155' : '#4f46e5',
              color: 'white',
              fontSize: 13,
              border: 'none',
              cursor: running ? 'default' : 'pointer',
            }}
          >
            {running ? '…' : '↑'}
          </button>
        </div>
      </div>
    </div>
  )
}

function formatActionSummary(actions: AgentCanvasAction[]): string {
  return actions
    .map((action) => {
      switch (action.type) {
        case 'add_node':
          return `add ${action.blockType}`
        case 'replace_node':
          return `replace ${action.nodeRef} -> ${action.blockType}`
        case 'update_params':
          return `update ${action.nodeRef}`
        case 'connect':
          return `connect ${action.source} -> ${action.target}`
        case 'move_node':
          return `move ${action.nodeRef}`
        case 'delete_node':
          return `delete ${action.nodeRef}`
        case 'auto_layout':
          return `layout ${action.direction ?? 'LR'}`
        case 'clear_canvas':
          return 'clear canvas'
        default:
          return action.type
      }
    })
    .join(', ')
}
