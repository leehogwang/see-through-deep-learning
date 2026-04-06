export type Shape = number[] // e.g. [B, C, H, W]

function parseShape(s: string): Shape {
  return s.split(',').map(x => x.trim() === 'B' ? -1 : parseInt(x.trim(), 10))
}

function shapeStr(s: Shape): string {
  return '(' + s.map(v => v === -1 ? 'B' : String(v)).join(', ') + ')'
}

function convOut(size: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((size + 2 * padding - kernel) / stride) + 1
}

export function calcOutputShape(
  blockType: string,
  params: Record<string, string | number>,
  inputShape: Shape | null
): { shape: Shape | null; error: boolean; str: string } {
  const err = (msg: string) => ({ shape: null, error: true, str: msg })
  const ok = (s: Shape) => ({ shape: s, error: false, str: shapeStr(s) })

  // Input/Output don't need an upstream shape
  if (blockType === 'Input') {
    const parts = String(params.shape || 'B,3,224,224').split(',')
    const s = parts.map(x => x.trim() === 'B' ? -1 : parseInt(x.trim(), 10))
    return ok(s)
  }
  if (blockType === 'Output') return inputShape ? ok([...inputShape]) : err('no input')

  if (!inputShape) return err('no input')

  const p = params as Record<string, number>

  switch (blockType) {

    case 'Conv1D': {
      if (inputShape.length !== 3) return err('needs (B,C,L)')
      const L2 = convOut(inputShape[2], p.kernel || 3, p.stride || 1, p.padding || 0)
      return ok([-1, p.out_ch || 32, L2])
    }
    case 'Conv2D':
    case 'DepthwiseConv2D':
    case 'DilatedConv2D': {
      if (inputShape.length !== 4) return err('needs (B,C,H,W)')
      const k = p.kernel || 3, s = p.stride || 1, pad = p.padding || 0
      const H2 = convOut(inputShape[2], k, s, pad)
      const W2 = convOut(inputShape[3], k, s, pad)
      const ch = blockType === 'DepthwiseConv2D' ? (p.in_ch || inputShape[1]) : (p.out_ch || 64)
      return ok([-1, ch, H2, W2])
    }
    case 'SeparableConv2D': {
      if (inputShape.length !== 4) return err('needs (B,C,H,W)')
      const k = p.kernel || 3
      const H2 = convOut(inputShape[2], k, 1, Math.floor(k / 2))
      const W2 = convOut(inputShape[3], k, 1, Math.floor(k / 2))
      return ok([-1, p.out_ch || 128, H2, W2])
    }
    case 'Conv3D': {
      if (inputShape.length !== 5) return err('needs (B,C,D,H,W)')
      const k = p.kernel || 3, s = p.stride || 1, pad = p.padding || 0
      return ok([-1, p.out_ch || 32,
        convOut(inputShape[2], k, s, pad),
        convOut(inputShape[3], k, s, pad),
        convOut(inputShape[4], k, s, pad)])
    }
    case 'TransposedConv2D': {
      if (inputShape.length !== 4) return err('needs (B,C,H,W)')
      const k = p.kernel || 2, s = p.stride || 2
      const H2 = (inputShape[2] - 1) * s + k
      const W2 = (inputShape[3] - 1) * s + k
      return ok([-1, p.out_ch || 32, H2, W2])
    }

    case 'MaxPool2D':
    case 'AvgPool2D': {
      if (inputShape.length !== 4) return err('needs (B,C,H,W)')
      const k = p.kernel || 2, s = p.stride || k
      return ok([-1, inputShape[1], convOut(inputShape[2], k, s, 0), convOut(inputShape[3], k, s, 0)])
    }
    case 'GlobalAvgPool':
    case 'GlobalMaxPool':
      if (inputShape.length < 3) return err('needs (B,C,...)')
      return ok([-1, inputShape[1]])
    case 'AdaptiveAvgPool': {
      if (inputShape.length !== 4) return err('needs (B,C,H,W)')
      const sz = String(params.output_size || '1,1').split(',').map(Number)
      return ok([-1, inputShape[1], sz[0] || 1, sz[1] || 1])
    }

    // Activations & Norms — shape preserved
    case 'ReLU': case 'LeakyReLU': case 'PReLU': case 'ELU': case 'SELU':
    case 'GELU': case 'SiLU': case 'Mish': case 'Sigmoid': case 'Tanh':
    case 'Softmax': case 'HardSwish':
    case 'BatchNorm2D': case 'LayerNorm': case 'GroupNorm': case 'InstanceNorm': case 'RMSNorm':
    case 'Dropout': case 'Dropout2D': case 'DropPath': case 'SpatialDropout':
    case 'SinPE': case 'LearnedPE': case 'RoPE': case 'ALiBi':
      return ok([...inputShape])

    case 'Flatten': {
      const total = inputShape.slice(1).reduce((a, b) => a * (b === -1 ? 1 : b), 1)
      return ok([-1, total])
    }
    case 'Reshape': {
      const parts = String(params.shape || 'B,-1').split(',')
      return ok(parts.map(x => x.trim() === 'B' ? -1 : parseInt(x.trim(), 10)))
    }
    case 'Squeeze':
      return ok(inputShape.filter((_, i) => i !== (p.dim || 1)))
    case 'Unsqueeze': {
      const dim = p.dim || 1
      return ok([...inputShape.slice(0, dim), 1, ...inputShape.slice(dim)])
    }
    case 'Permute': {
      const dims = String(params.dims || '0,1,2,3').split(',').map(Number)
      return ok(dims.map(i => inputShape[i] ?? -1))
    }

    case 'Linear': {
      const outFeats = p.out_features || 10
      return ok([...inputShape.slice(0, -1), outFeats])
    }

    case 'MultiHeadAttention': case 'SelfAttention': case 'CrossAttention':
    case 'FlashAttention': case 'GQA': case 'MQA': case 'LinearAttention':
      return ok([...inputShape])

    case 'LSTM': case 'GRU': case 'RNN': {
      const hidden = p.hidden_size || 256
      return ok([-1, inputShape[1] ?? -1, hidden])
    }
    case 'BiLSTM': case 'BiGRU': {
      const hidden = p.hidden_size || 256
      return ok([-1, inputShape[1] ?? -1, hidden * 2])
    }

    case 'Embedding': {
      const dim = p.embedding_dim || 512
      return ok([-1, inputShape[inputShape.length - 1], dim])
    }
    case 'PatchEmbedding': {
      const dim = p.embed_dim || 768
      return ok([-1, -1, dim])
    }

    case 'TransformerEncoderLayer': case 'TransformerDecoderLayer':
    case 'FFN':
      return ok([...inputShape])

    case 'MoE':
      return ok([...inputShape])

    case 'Concat': {
      const dim = p.dim || 1
      const newShape = [...inputShape]
      if (newShape[dim] !== undefined) newShape[dim] = -1
      return ok(newShape)
    }
    case 'Add': return ok([...inputShape])
    case 'Split': {
      const newShape = [...inputShape]
      if (newShape[p.dim || 1] !== undefined) newShape[p.dim || 1] = -1
      return ok(newShape)
    }

    case 'Output': return ok([...inputShape])

    default: return ok([...inputShape])
  }
}

export { parseShape, shapeStr }
