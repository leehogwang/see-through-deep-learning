import type { Node, Edge } from '@xyflow/react'
import type { LayerNodeData } from '../components/FlowCanvas/nodes/LayerNode'
import { getBlockDef, type ParamValue } from '../data/blocks'

function getTopologicalOrder(nodes: Node[], edges: Edge[]): Node[] {
  const inDegree: Record<string, number> = {}
  const adj: Record<string, string[]> = {}
  nodes.forEach(n => { inDegree[n.id] = 0; adj[n.id] = [] })
  edges.forEach(e => {
    adj[e.source].push(e.target)
    inDegree[e.target]++
  })
  const queue = nodes.filter(n => inDegree[n.id] === 0)
  const result: Node[] = []
  while (queue.length) {
    const node = queue.shift()!
    result.push(node)
    adj[node.id].forEach(nid => {
      inDegree[nid]--
      if (inDegree[nid] === 0) queue.push(nodes.find(n => n.id === nid)!)
    })
  }
  return result
}

export function generatePyTorchCode(nodes: Node[], edges: Edge[]): string {
  if (nodes.length === 0) return '# No layers defined yet'

  const ordered = getTopologicalOrder(nodes, edges)
  const lines: string[] = [
    'import torch',
    'import torch.nn as nn',
    '',
    '',
    'class GeneratedModel(nn.Module):',
    '    def __init__(self):',
    '        super().__init__()',
  ]

  const varNames: Record<string, string> = {}
  const initLines: string[] = []
  const forwardLines: string[] = []
  const counters: Record<string, number> = {}

  const getVar = (type: string, id: string) => {
    if (varNames[id]) return varNames[id]
    const base = type.toLowerCase().replace(/[^a-z0-9]/g, '_')
    counters[base] = (counters[base] || 0) + 1
    const name = `${base}${counters[base] > 1 ? counters[base] : ''}`
    varNames[id] = name
    return name
  }

  const p = (data: LayerNodeData) => data.params
  const py = (value: ParamValue, tupleish = false) => formatPythonValue(value, tupleish)

  ordered.forEach(node => {
    const data = node.data as LayerNodeData
    const t = data.blockType
    const vn = getVar(t, node.id)
    const params = p(data)

    const layer = (() => {
      switch (t) {
        case 'Input':  return null
        case 'Output': return null
        case 'Conv1D': return `nn.Conv1d(${py(params.in_ch)}, ${py(params.out_ch)}, kernel_size=${py(params.kernel, true)}, stride=${py(params.stride, true)}, padding=${py(params.padding, true)})`
        case 'Conv2D': return `nn.Conv2d(${py(params.in_ch)}, ${py(params.out_ch)}, kernel_size=${py(params.kernel, true)}, stride=${py(params.stride, true)}, padding=${py(params.padding, true)})`
        case 'Conv3D': return `nn.Conv3d(${py(params.in_ch)}, ${py(params.out_ch)}, kernel_size=${py(params.kernel, true)}, stride=${py(params.stride, true)}, padding=${py(params.padding, true)})`
        case 'DepthwiseConv2D': return `nn.Conv2d(${py(params.in_ch)}, ${py(params.in_ch)}, kernel_size=${py(params.kernel, true)}, groups=${py(params.in_ch)}, padding=${py(params.padding, true)})`
        case 'TransposedConv2D': return `nn.ConvTranspose2d(${py(params.in_ch)}, ${py(params.out_ch)}, kernel_size=${py(params.kernel, true)}, stride=${py(params.stride, true)})`
        case 'DilatedConv2D': return `nn.Conv2d(${py(params.in_ch)}, ${py(params.out_ch)}, kernel_size=${py(params.kernel, true)}, dilation=${py(params.dilation, true)}, padding=${py(params.dilation, true)})`
        case 'MaxPool2D': return `nn.MaxPool2d(kernel_size=${py(params.kernel, true)}, stride=${py(params.stride, true)})`
        case 'AvgPool2D': return `nn.AvgPool2d(kernel_size=${py(params.kernel, true)}, stride=${py(params.stride, true)})`
        case 'GlobalAvgPool': return `nn.AdaptiveAvgPool2d(1)`
        case 'GlobalMaxPool': return `nn.AdaptiveMaxPool2d(1)`
        case 'AdaptiveAvgPool': return `nn.AdaptiveAvgPool2d(${py(params.output_size, true)})`
        case 'ReLU': return `nn.ReLU()`
        case 'LeakyReLU': return `nn.LeakyReLU(${py(params.alpha)})`
        case 'PReLU': return `nn.PReLU()`
        case 'ELU': return `nn.ELU(alpha=${py(params.alpha)})`
        case 'SELU': return `nn.SELU()`
        case 'GELU': return `nn.GELU()`
        case 'SiLU': return `nn.SiLU()`
        case 'Mish': return `nn.Mish()`
        case 'Sigmoid': return `nn.Sigmoid()`
        case 'Tanh': return `nn.Tanh()`
        case 'Softmax': return `nn.Softmax(dim=${py(params.dim)})`
        case 'HardSwish': return `nn.Hardswish()`
        case 'BatchNorm2D': return `nn.BatchNorm2d(${py(params.num_features)})`
        case 'LayerNorm': return `nn.LayerNorm(${py(params.normalized_shape, true)})`
        case 'GroupNorm': return `nn.GroupNorm(${py(params.num_groups)}, ${py(params.num_channels)})`
        case 'InstanceNorm': return `nn.InstanceNorm2d(${py(params.num_features)})`
        case 'RMSNorm': return `nn.RMSNorm(${py(params.dim)})`
        case 'Dropout': return `nn.Dropout(p=${py(params.p)})`
        case 'Dropout2D': return `nn.Dropout2d(p=${py(params.p)})`
        case 'Linear': return `nn.Linear(${py(params.in_features)}, ${py(params.out_features)})`
        case 'Bilinear': return `nn.Bilinear(${py(params.in1)}, ${py(params.in2)}, ${py(params.out)})`
        case 'MultiHeadAttention': return `nn.MultiheadAttention(${py(params.embed_dim)}, ${py(params.num_heads)}, batch_first=True)`
        case 'LSTM': return `nn.LSTM(${py(params.input_size)}, ${py(params.hidden_size)}, num_layers=${py(params.num_layers)}, batch_first=True)`
        case 'GRU': return `nn.GRU(${py(params.input_size)}, ${py(params.hidden_size)}, batch_first=True)`
        case 'RNN': return `nn.RNN(${py(params.input_size)}, ${py(params.hidden_size)}, batch_first=True)`
        case 'BiLSTM': return `nn.LSTM(${py(params.input_size)}, ${py(params.hidden_size)}, bidirectional=True, batch_first=True)`
        case 'BiGRU': return `nn.GRU(${py(params.input_size)}, ${py(params.hidden_size)}, bidirectional=True, batch_first=True)`
        case 'Embedding': return `nn.Embedding(${py(params.num_embeddings)}, ${py(params.embedding_dim)})`
        case 'TransformerEncoderLayer': return `nn.TransformerEncoderLayer(d_model=${py(params.d_model)}, nhead=${py(params.nhead)}, dim_feedforward=${py(params.dim_feedforward)}, batch_first=True)`
        case 'TransformerDecoderLayer': return `nn.TransformerDecoderLayer(d_model=${py(params.d_model)}, nhead=${py(params.nhead)}, dim_feedforward=${py(params.dim_feedforward)}, batch_first=True)`
        case 'Flatten': return `nn.Flatten(start_dim=${py(params.start_dim ?? 1)})`
        default: return buildGenericLayerCall(t, params)
      }
    })()

    if (layer) {
      initLines.push(`        self.${vn} = ${layer}`)
    }

    const fwd = (() => {
      if (t === 'Input') return `        x = inputs`
      if (t === 'Output') return `        return x`
      if (!layer) return null
      if (t === 'LSTM' || t === 'GRU' || t === 'RNN' || t === 'BiLSTM' || t === 'BiGRU')
        return `        x, _ = self.${vn}(x)`
      if (t === 'MultiHeadAttention')
        return `        x, _ = self.${vn}(x, x, x)`
      if (t === 'GlobalAvgPool' || t === 'GlobalMaxPool')
        return `        x = self.${vn}(x).squeeze(-1).squeeze(-1)`
      if (/attention/i.test(t))
        return `        x, _ = self.${vn}(x, x, x)`
      if (/^(LSTM|GRU|RNN|BiLSTM|BiGRU)$/i.test(t))
        return `        x, _ = self.${vn}(x)`
      const genericInputs = getBlockDef(t)?.maxInputs ?? 1
      if (genericInputs > 1) {
        const args = Array.from({ length: genericInputs }, () => 'x').join(', ')
        return `        x = self.${vn}(${args})`
      }
      return `        x = self.${vn}(x)`
    })()
    if (fwd) forwardLines.push(fwd)
  })

  lines.push(...initLines)
  lines.push('', '    def forward(self, inputs):')
  lines.push(...forwardLines)
  if (!forwardLines.some(l => l.includes('return'))) {
    lines.push('        return x')
  }
  lines.push('', '', 'model = GeneratedModel()')
  lines.push('print(model)')

  return lines.join('\n')
}

function formatPythonValue(value: ParamValue, tupleish = false): string {
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'number') return String(value)

  const trimmed = value.trim()
  if (trimmed.length === 0) return "''"
  if (trimmed === 'None' || trimmed === 'True' || trimmed === 'False') return trimmed
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) return trimmed
  if (tupleish && trimmed.includes(',')) return `(${trimmed})`
  if (/^['"].*['"]$/.test(trimmed)) return trimmed
  if (/^[A-Za-z_][A-Za-z0-9_.]*$/.test(trimmed) && trimmed.includes('.')) return trimmed
  return `'${trimmed.replace(/'/g, "\\'")}'`
}

function buildGenericLayerCall(type: string, params: Record<string, ParamValue>): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(type)) {
    return `nn.Identity()  # unsupported block type: ${type}`
  }

  const definition = getBlockDef(type)
  const entries = Object.entries(params).filter(([key]) => key !== 'class')
  const argList = entries.map(([key, value]) => {
    const tupleish = typeof value === 'string' && value.includes(',')
      || /(?:shape|size|dims|kernel|stride|padding|dilation)$/i.test(key)
    return `${key}=${formatPythonValue(value, tupleish)}`
  }).join(', ')

  if (definition) {
    return `nn.${type}(${argList})`
  }

  return `nn.Identity()  # ${type}`
}
