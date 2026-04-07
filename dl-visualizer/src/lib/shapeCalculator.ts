import type { ParamValue } from '../data/blocks'

export type Shape = number[]

function canonicalBlockType(blockType: string): string {
  const lower = blockType.toLowerCase()
  const aliasMap: Record<string, string> = {
    conv1d: 'Conv1D',
    conv2d: 'Conv2D',
    conv3d: 'Conv3D',
    convtranspose2d: 'TransposedConv2D',
    maxpool2d: 'MaxPool2D',
    avgpool2d: 'AvgPool2D',
    adaptiveavgpool1d: 'AdaptiveAvgPool',
    adaptiveavgpool2d: 'AdaptiveAvgPool',
    adaptiveavgpool3d: 'AdaptiveAvgPool',
    relu: 'ReLU',
    leakyrelu: 'LeakyReLU',
    prelu: 'PReLU',
    elu: 'ELU',
    selu: 'SELU',
    gelu: 'GELU',
    silu: 'SiLU',
    mish: 'Mish',
    sigmoid: 'Sigmoid',
    tanh: 'Tanh',
    softmax: 'Softmax',
    hardswish: 'HardSwish',
    batchnorm1d: 'BatchNorm2D',
    batchnorm2d: 'BatchNorm2D',
    batchnorm3d: 'BatchNorm2D',
    layernorm: 'LayerNorm',
    groupnorm: 'GroupNorm',
    instancenorm1d: 'InstanceNorm',
    instancenorm2d: 'InstanceNorm',
    instancenorm3d: 'InstanceNorm',
    rmsnorm: 'RMSNorm',
    dropout: 'Dropout',
    dropout1d: 'Dropout',
    dropout2d: 'Dropout2D',
    dropout3d: 'Dropout2D',
    linear: 'Linear',
    bilinear: 'Bilinear',
    multiheadattention: 'MultiHeadAttention',
    lstm: 'LSTM',
    gru: 'GRU',
    rnn: 'RNN',
    embedding: 'Embedding',
    flatten: 'Flatten',
    transformerencoderlayer: 'TransformerEncoderLayer',
    transformerdecoderlayer: 'TransformerDecoderLayer',
  }
  return aliasMap[lower] ?? blockType
}

function parseShape(s: string): Shape {
  return s.split(',').map((token) => token.trim() === 'B' ? -1 : Number.parseInt(token.trim(), 10))
}

function shapeStr(s: Shape): string {
  return `(${s.map((value) => value === -1 ? 'B' : String(value)).join(', ')})`
}

function convOut(size: number, kernel: number, stride: number, padding: number, dilation = 1): number {
  return Math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1
}

function normalizeInputShape(inputShape: Shape | null, inputShapes: Shape[]): Shape | null {
  return inputShape ?? inputShapes.find((shape) => Array.isArray(shape)) ?? null
}

function parseAxisValues(value: ParamValue | undefined, dims: number, fallback: number): number[] {
  if (typeof value === 'number') return Array(dims).fill(value)
  if (typeof value === 'string') {
    const values = value
      .split(',')
      .map((token) => Number.parseInt(token.trim(), 10))
      .filter((item) => Number.isFinite(item))
    if (values.length === 0) return Array(dims).fill(fallback)
    if (values.length === 1) return Array(dims).fill(values[0])
    return Array.from({ length: dims }, (_, index) => values[index] ?? values[values.length - 1])
  }
  return Array(dims).fill(fallback)
}

function parseNumber(value: ParamValue | undefined, fallback: number): number {
  if (typeof value === 'number') return value
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value.trim())
    return Number.isFinite(parsed) ? parsed : fallback
  }
  return fallback
}

function sumConcatShapes(shapes: Shape[], dim: number): Shape | null {
  if (shapes.length === 0) return null
  const merged = [...shapes[0]]
  for (const shape of shapes.slice(1)) {
    if (shape.length !== merged.length) return null
    for (let index = 0; index < shape.length; index += 1) {
      if (index === dim) continue
      if (merged[index] !== -1 && shape[index] !== -1 && merged[index] !== shape[index]) {
        return null
      }
    }
    if (merged[dim] === -1 || shape[dim] === -1) {
      merged[dim] = -1
    } else {
      merged[dim] += shape[dim]
    }
  }
  return merged
}

export function calcOutputShape(
  blockType: string,
  params: Record<string, ParamValue>,
  inputShape: Shape | null,
  inputShapes: Shape[] = [],
): { shape: Shape | null; error: boolean; str: string } {
  blockType = canonicalBlockType(blockType)
  const err = (msg: string) => ({ shape: null, error: true, str: msg })
  const ok = (shape: Shape) => ({ shape, error: false, str: shapeStr(shape) })

  if (blockType === 'Input') {
    return ok(parseShape(String(params.shape || 'B,3,224,224')))
  }

  const resolvedInputShape = normalizeInputShape(inputShape, inputShapes)
  if (blockType === 'Output') return resolvedInputShape ? ok([...resolvedInputShape]) : err('no input')
  if (!resolvedInputShape) return err('no input')

  switch (blockType) {
    case 'Conv1D': {
      if (resolvedInputShape.length !== 3) return err('needs (B,C,L)')
      const inChannels = parseNumber(params.in_ch, resolvedInputShape[1])
      if (resolvedInputShape[1] !== -1 && inChannels !== resolvedInputShape[1]) return err(`expected C=${inChannels}, got ${resolvedInputShape[1]}`)
      const kernel = parseAxisValues(params.kernel, 1, 3)[0]
      const stride = parseAxisValues(params.stride, 1, 1)[0]
      const padding = parseAxisValues(params.padding, 1, 0)[0]
      const length = convOut(resolvedInputShape[2], kernel, stride, padding)
      return ok([-1, parseNumber(params.out_ch, 32), length])
    }
    case 'Conv2D':
    case 'DepthwiseConv2D':
    case 'DilatedConv2D': {
      if (resolvedInputShape.length !== 4) return err('needs (B,C,H,W)')
      const inChannels = parseNumber(params.in_ch, resolvedInputShape[1])
      if (resolvedInputShape[1] !== -1 && inChannels !== resolvedInputShape[1]) return err(`expected C=${inChannels}, got ${resolvedInputShape[1]}`)
      const kernel = parseAxisValues(params.kernel, 2, 3)
      const stride = parseAxisValues(params.stride, 2, 1)
      const padding = parseAxisValues(params.padding, 2, 0)
      const dilation = parseAxisValues(params.dilation, 2, 1)
      const height = convOut(resolvedInputShape[2], kernel[0], stride[0], padding[0], dilation[0])
      const width = convOut(resolvedInputShape[3], kernel[1], stride[1], padding[1], dilation[1])
      const outChannels = blockType === 'DepthwiseConv2D'
        ? parseNumber(params.in_ch, resolvedInputShape[1])
        : parseNumber(params.out_ch, 64)
      return ok([-1, outChannels, height, width])
    }
    case 'SeparableConv2D': {
      if (resolvedInputShape.length !== 4) return err('needs (B,C,H,W)')
      const kernel = parseAxisValues(params.kernel, 2, 3)
      const height = convOut(resolvedInputShape[2], kernel[0], 1, Math.floor(kernel[0] / 2))
      const width = convOut(resolvedInputShape[3], kernel[1], 1, Math.floor(kernel[1] / 2))
      return ok([-1, parseNumber(params.out_ch, 128), height, width])
    }
    case 'Conv3D': {
      if (resolvedInputShape.length !== 5) return err('needs (B,C,D,H,W)')
      const inChannels = parseNumber(params.in_ch, resolvedInputShape[1])
      if (resolvedInputShape[1] !== -1 && inChannels !== resolvedInputShape[1]) return err(`expected C=${inChannels}, got ${resolvedInputShape[1]}`)
      const kernel = parseAxisValues(params.kernel, 3, 3)
      const stride = parseAxisValues(params.stride, 3, 1)
      const padding = parseAxisValues(params.padding, 3, 0)
      return ok([
        -1,
        parseNumber(params.out_ch, 32),
        convOut(resolvedInputShape[2], kernel[0], stride[0], padding[0]),
        convOut(resolvedInputShape[3], kernel[1], stride[1], padding[1]),
        convOut(resolvedInputShape[4], kernel[2], stride[2], padding[2]),
      ])
    }
    case 'TransposedConv2D': {
      if (resolvedInputShape.length !== 4) return err('needs (B,C,H,W)')
      const kernel = parseAxisValues(params.kernel, 2, 2)
      const stride = parseAxisValues(params.stride, 2, 2)
      return ok([
        -1,
        parseNumber(params.out_ch, 32),
        (resolvedInputShape[2] - 1) * stride[0] + kernel[0],
        (resolvedInputShape[3] - 1) * stride[1] + kernel[1],
      ])
    }
    case 'MaxPool2D':
    case 'AvgPool2D': {
      if (resolvedInputShape.length !== 4) return err('needs (B,C,H,W)')
      const kernel = parseAxisValues(params.kernel, 2, 2)
      const stride = parseAxisValues(params.stride, 2, kernel[0])
      return ok([
        -1,
        resolvedInputShape[1],
        convOut(resolvedInputShape[2], kernel[0], stride[0], 0),
        convOut(resolvedInputShape[3], kernel[1], stride[1], 0),
      ])
    }
    case 'GlobalAvgPool':
    case 'GlobalMaxPool':
      if (resolvedInputShape.length < 3) return err('needs (B,C,...)')
      return ok([-1, resolvedInputShape[1]])
    case 'AdaptiveAvgPool': {
      if (resolvedInputShape.length !== 4) return err('needs (B,C,H,W)')
      const size = parseAxisValues(params.output_size, 2, 1)
      return ok([-1, resolvedInputShape[1], size[0], size[1]])
    }
    case 'ReLU':
    case 'LeakyReLU':
    case 'PReLU':
    case 'ELU':
    case 'SELU':
    case 'GELU':
    case 'SiLU':
    case 'Mish':
    case 'Sigmoid':
    case 'Tanh':
    case 'Softmax':
    case 'HardSwish':
    case 'BatchNorm2D':
    case 'LayerNorm':
    case 'GroupNorm':
    case 'InstanceNorm':
    case 'RMSNorm':
    case 'Dropout':
    case 'Dropout2D':
    case 'DropPath':
    case 'SpatialDropout':
    case 'SinPE':
    case 'LearnedPE':
    case 'RoPE':
    case 'ALiBi':
      return ok([...resolvedInputShape])
    case 'Flatten': {
      const total = resolvedInputShape.slice(1).reduce((acc, value) => acc * (value === -1 ? 1 : value), 1)
      return ok([-1, total])
    }
    case 'Reshape':
      return ok(parseShape(String(params.shape || 'B,-1')))
    case 'Squeeze': {
      const dim = parseNumber(params.dim, 1)
      return ok(resolvedInputShape.filter((_, index) => index !== dim))
    }
    case 'Unsqueeze': {
      const dim = parseNumber(params.dim, 1)
      return ok([...resolvedInputShape.slice(0, dim), 1, ...resolvedInputShape.slice(dim)])
    }
    case 'Permute': {
      const dims = parseAxisValues(params.dims, resolvedInputShape.length, 0)
      return ok(dims.map((index) => resolvedInputShape[index] ?? -1))
    }
    case 'Linear':
      return ok([...resolvedInputShape.slice(0, -1), parseNumber(params.out_features, 10)])
    case 'MultiHeadAttention':
    case 'SelfAttention':
    case 'CrossAttention':
    case 'FlashAttention':
    case 'GQA':
    case 'MQA':
    case 'LinearAttention': {
      const embedDim = parseNumber(params.embed_dim, resolvedInputShape[resolvedInputShape.length - 1] ?? 0)
      const numHeads = parseNumber(params.num_heads, 1)
      if (numHeads > 0 && embedDim > 0 && embedDim % numHeads !== 0) return err(`embed_dim ${embedDim} not divisible by num_heads ${numHeads}`)
      return ok([...resolvedInputShape])
    }
    case 'LSTM':
    case 'GRU':
    case 'RNN':
      return ok([-1, resolvedInputShape[1] ?? -1, parseNumber(params.hidden_size, 256)])
    case 'BiLSTM':
    case 'BiGRU':
      return ok([-1, resolvedInputShape[1] ?? -1, parseNumber(params.hidden_size, 256) * 2])
    case 'Embedding':
      return ok([-1, resolvedInputShape[resolvedInputShape.length - 1], parseNumber(params.embedding_dim, 512)])
    case 'PatchEmbedding':
      return ok([-1, -1, parseNumber(params.embed_dim, 768)])
    case 'TransformerEncoderLayer':
    case 'TransformerDecoderLayer':
    case 'FFN':
    case 'MoE':
      return ok([...resolvedInputShape])
    case 'Concat': {
      const dim = parseNumber(params.dim, 1)
      const merged = sumConcatShapes(inputShapes.length > 0 ? inputShapes : [resolvedInputShape], dim)
      return merged ? ok(merged) : err('concat mismatch')
    }
    case 'Add':
      return ok([...(inputShapes[0] ?? resolvedInputShape)])
    case 'Split': {
      const nextShape = [...resolvedInputShape]
      const dim = parseNumber(params.dim, 1)
      if (nextShape[dim] !== undefined) nextShape[dim] = -1
      return ok(nextShape)
    }
    default:
      return ok([...resolvedInputShape])
  }
}

export { parseShape, shapeStr }
