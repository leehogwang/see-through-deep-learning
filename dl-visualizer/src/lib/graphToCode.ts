import type { Node, Edge } from '@xyflow/react'
import type { LayerNodeData } from '../components/FlowCanvas/nodes/LayerNode'

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

  ordered.forEach(node => {
    const data = node.data as LayerNodeData
    const t = data.blockType
    const vn = getVar(t, node.id)
    const params = p(data)

    const layer = (() => {
      switch (t) {
        case 'Input':  return null
        case 'Output': return null
        case 'Conv1D': return `nn.Conv1d(${params.in_ch}, ${params.out_ch}, kernel_size=${params.kernel}, stride=${params.stride}, padding=${params.padding})`
        case 'Conv2D': return `nn.Conv2d(${params.in_ch}, ${params.out_ch}, kernel_size=${params.kernel}, stride=${params.stride}, padding=${params.padding})`
        case 'Conv3D': return `nn.Conv3d(${params.in_ch}, ${params.out_ch}, kernel_size=${params.kernel}, stride=${params.stride}, padding=${params.padding})`
        case 'DepthwiseConv2D': return `nn.Conv2d(${params.in_ch}, ${params.in_ch}, kernel_size=${params.kernel}, groups=${params.in_ch}, padding=${params.padding})`
        case 'TransposedConv2D': return `nn.ConvTranspose2d(${params.in_ch}, ${params.out_ch}, kernel_size=${params.kernel}, stride=${params.stride})`
        case 'DilatedConv2D': return `nn.Conv2d(${params.in_ch}, ${params.out_ch}, kernel_size=${params.kernel}, dilation=${params.dilation}, padding=${params.dilation})`
        case 'MaxPool2D': return `nn.MaxPool2d(kernel_size=${params.kernel}, stride=${params.stride})`
        case 'AvgPool2D': return `nn.AvgPool2d(kernel_size=${params.kernel}, stride=${params.stride})`
        case 'GlobalAvgPool': return `nn.AdaptiveAvgPool2d(1)`
        case 'GlobalMaxPool': return `nn.AdaptiveMaxPool2d(1)`
        case 'AdaptiveAvgPool': return `nn.AdaptiveAvgPool2d((${params.output_size}))`
        case 'ReLU': return `nn.ReLU()`
        case 'LeakyReLU': return `nn.LeakyReLU(${params.alpha})`
        case 'PReLU': return `nn.PReLU()`
        case 'ELU': return `nn.ELU(alpha=${params.alpha})`
        case 'SELU': return `nn.SELU()`
        case 'GELU': return `nn.GELU()`
        case 'SiLU': return `nn.SiLU()`
        case 'Mish': return `nn.Mish()`
        case 'Sigmoid': return `nn.Sigmoid()`
        case 'Tanh': return `nn.Tanh()`
        case 'Softmax': return `nn.Softmax(dim=${params.dim})`
        case 'HardSwish': return `nn.Hardswish()`
        case 'BatchNorm2D': return `nn.BatchNorm2d(${params.num_features})`
        case 'LayerNorm': return `nn.LayerNorm(${params.normalized_shape})`
        case 'GroupNorm': return `nn.GroupNorm(${params.num_groups}, ${params.num_channels})`
        case 'InstanceNorm': return `nn.InstanceNorm2d(${params.num_features})`
        case 'RMSNorm': return `nn.RMSNorm(${params.dim})`
        case 'Dropout': return `nn.Dropout(p=${params.p})`
        case 'Dropout2D': return `nn.Dropout2d(p=${params.p})`
        case 'Linear': return `nn.Linear(${params.in_features}, ${params.out_features})`
        case 'Bilinear': return `nn.Bilinear(${params.in1}, ${params.in2}, ${params.out})`
        case 'MultiHeadAttention': return `nn.MultiheadAttention(${params.embed_dim}, ${params.num_heads}, batch_first=True)`
        case 'LSTM': return `nn.LSTM(${params.input_size}, ${params.hidden_size}, num_layers=${params.num_layers}, batch_first=True)`
        case 'GRU': return `nn.GRU(${params.input_size}, ${params.hidden_size}, batch_first=True)`
        case 'RNN': return `nn.RNN(${params.input_size}, ${params.hidden_size}, batch_first=True)`
        case 'BiLSTM': return `nn.LSTM(${params.input_size}, ${params.hidden_size}, bidirectional=True, batch_first=True)`
        case 'BiGRU': return `nn.GRU(${params.input_size}, ${params.hidden_size}, bidirectional=True, batch_first=True)`
        case 'Embedding': return `nn.Embedding(${params.num_embeddings}, ${params.embedding_dim})`
        case 'TransformerEncoderLayer': return `nn.TransformerEncoderLayer(d_model=${params.d_model}, nhead=${params.nhead}, dim_feedforward=${params.dim_feedforward}, batch_first=True)`
        case 'TransformerDecoderLayer': return `nn.TransformerDecoderLayer(d_model=${params.d_model}, nhead=${params.nhead}, dim_feedforward=${params.dim_feedforward}, batch_first=True)`
        case 'Flatten': return `nn.Flatten(start_dim=${params.start_dim ?? 1})`
        default: return `nn.Identity()  # ${t}`
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
