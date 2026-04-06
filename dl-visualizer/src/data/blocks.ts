export interface BlockDef {
  type: string
  label: string
  category: string
  color: string
  formula?: string
  description: string
  defaultParams: Record<string, string | number>
}

export const CATEGORIES = [
  { id: 'io',           icon: '📥', label: 'Input / Output' },
  { id: 'conv',         icon: '🔲', label: 'Convolution' },
  { id: 'pooling',      icon: '🏊', label: 'Pooling' },
  { id: 'activation',   icon: '⚡', label: 'Activation' },
  { id: 'norm',         icon: '📏', label: 'Normalization' },
  { id: 'attention',    icon: '🎯', label: 'Attention' },
  { id: 'recurrent',    icon: '🔄', label: 'Recurrent' },
  { id: 'embedding',    icon: '🔤', label: 'Embedding' },
  { id: 'transformer',  icon: '🧩', label: 'Transformer' },
  { id: 'shape',        icon: '🔗', label: 'Shape Ops' },
  { id: 'regularize',   icon: '🛡️', label: 'Regularization' },
  { id: 'linear',       icon: '🔵', label: 'Linear' },
]

export const BLOCKS: BlockDef[] = [
  // ── IO ──────────────────────────────────────────────────────
  {
    type: 'Input', label: 'Input', category: 'io', color: '#10b981',
    description: '데이터 입력 노드. 텐서의 초기 shape을 정의합니다.',
    defaultParams: { shape: 'B,3,224,224' },
  },
  {
    type: 'Output', label: 'Output', category: 'io', color: '#10b981',
    description: '모델의 최종 출력 노드.',
    defaultParams: {},
  },

  // ── Convolution ─────────────────────────────────────────────
  {
    type: 'Conv1D', label: 'Conv1D', category: 'conv', color: '#6366f1',
    formula: 'L\' = (L+2p-k)/s+1',
    description: '1D 합성곱. 시계열/텍스트 데이터에서 지역 패턴을 추출합니다.',
    defaultParams: { in_ch: 1, out_ch: 32, kernel: 3, stride: 1, padding: 1 },
  },
  {
    type: 'Conv2D', label: 'Conv2D', category: 'conv', color: '#6366f1',
    formula: 'H\' = (H+2p-k)/s+1',
    description: '2D 합성곱. 이미지에서 공간적 특징(엣지, 텍스처)을 추출합니다.',
    defaultParams: { in_ch: 3, out_ch: 64, kernel: 3, stride: 1, padding: 1 },
  },
  {
    type: 'Conv3D', label: 'Conv3D', category: 'conv', color: '#6366f1',
    formula: 'D\' = (D+2p-k)/s+1',
    description: '3D 합성곱. 비디오/의료 영상 등 시공간 특징을 추출합니다.',
    defaultParams: { in_ch: 1, out_ch: 32, kernel: 3, stride: 1, padding: 1 },
  },
  {
    type: 'DepthwiseConv2D', label: 'DepthwiseConv2D', category: 'conv', color: '#6366f1',
    description: '채널별 독립 합성곱. 파라미터가 적어 경량화 모델(MobileNet)에 사용됩니다.',
    defaultParams: { in_ch: 64, kernel: 3, stride: 1, padding: 1 },
  },
  {
    type: 'SeparableConv2D', label: 'SeparableConv2D', category: 'conv', color: '#6366f1',
    description: 'Depthwise + Pointwise 합성곱 결합. 연산량을 크게 줄입니다.',
    defaultParams: { in_ch: 64, out_ch: 128, kernel: 3 },
  },
  {
    type: 'TransposedConv2D', label: 'TransposedConv2D', category: 'conv', color: '#6366f1',
    formula: 'H\' = (H-1)·s-2p+k',
    description: '전치 합성곱(디컨볼루션). 특징맵을 업샘플링하며 생성 모델(GAN, U-Net 디코더)에 사용됩니다.',
    defaultParams: { in_ch: 64, out_ch: 32, kernel: 2, stride: 2 },
  },
  {
    type: 'DilatedConv2D', label: 'DilatedConv2D', category: 'conv', color: '#6366f1',
    description: '팽창 합성곱. 커널 사이에 간격을 두어 파라미터 수는 유지하면서 수용 영역(receptive field)을 넓힙니다.',
    defaultParams: { in_ch: 64, out_ch: 64, kernel: 3, dilation: 2 },
  },

  // ── Pooling ─────────────────────────────────────────────────
  {
    type: 'MaxPool2D', label: 'MaxPool2D', category: 'pooling', color: '#f59e0b',
    formula: 'H\' = H/k',
    description: '영역 내 최댓값 추출. 공간 크기를 줄이고 이동 불변성(translation invariance)을 높입니다.',
    defaultParams: { kernel: 2, stride: 2 },
  },
  {
    type: 'AvgPool2D', label: 'AvgPool2D', category: 'pooling', color: '#f59e0b',
    description: '영역 내 평균값. MaxPool보다 부드러운 특징 추출에 사용됩니다.',
    defaultParams: { kernel: 2, stride: 2 },
  },
  {
    type: 'GlobalAvgPool', label: 'GlobalAvgPool', category: 'pooling', color: '#f59e0b',
    formula: '(B,C,H,W) → (B,C)',
    description: '각 채널의 전체 평균. FC layer 없이 분류 헤드를 만들 때 사용됩니다(GAP).',
    defaultParams: {},
  },
  {
    type: 'GlobalMaxPool', label: 'GlobalMaxPool', category: 'pooling', color: '#f59e0b',
    formula: '(B,C,H,W) → (B,C)',
    description: '각 채널의 전체 최댓값.',
    defaultParams: {},
  },
  {
    type: 'AdaptiveAvgPool', label: 'AdaptiveAvgPool', category: 'pooling', color: '#f59e0b',
    description: '출력 크기를 직접 지정할 수 있는 평균 풀링. 다양한 입력 크기를 처리할 때 유용합니다.',
    defaultParams: { output_size: '1,1' },
  },

  // ── Activation ──────────────────────────────────────────────
  {
    type: 'ReLU', label: 'ReLU', category: 'activation', color: '#ec4899',
    formula: 'max(0, x)',
    description: '음수를 0으로 만드는 가장 기본적인 활성화 함수. 학습 속도가 빠르지만 Dying ReLU 문제가 있습니다.',
    defaultParams: {},
  },
  {
    type: 'LeakyReLU', label: 'LeakyReLU', category: 'activation', color: '#ec4899',
    formula: 'x>0: x, else: αx',
    description: '음수 입력에도 작은 기울기(α)를 허용. Dying ReLU 문제를 완화합니다.',
    defaultParams: { alpha: 0.01 },
  },
  {
    type: 'PReLU', label: 'PReLU', category: 'activation', color: '#ec4899',
    description: '음수 기울기 α를 학습으로 결정하는 Parametric ReLU.',
    defaultParams: {},
  },
  {
    type: 'ELU', label: 'ELU', category: 'activation', color: '#ec4899',
    formula: 'x>0: x, else: α(eˣ-1)',
    description: '지수함수 기반. 음수 출력이 가능해 평균이 0에 가까워집니다.',
    defaultParams: { alpha: 1.0 },
  },
  {
    type: 'SELU', label: 'SELU', category: 'activation', color: '#ec4899',
    description: '자기 정규화(self-normalizing). 특수한 초기화와 함께 쓰면 내부 공변량 변화를 줄입니다.',
    defaultParams: {},
  },
  {
    type: 'GELU', label: 'GELU', category: 'activation', color: '#ec4899',
    formula: 'x·Φ(x)',
    description: 'Gaussian Error Linear Unit. BERT, GPT 등 대부분의 Transformer에서 사용되는 표준 활성화 함수.',
    defaultParams: {},
  },
  {
    type: 'SiLU', label: 'SiLU / Swish', category: 'activation', color: '#ec4899',
    formula: 'x·σ(x)',
    description: 'Sigmoid-weighted Linear Unit. LLaMA, EfficientNet 등에서 사용됩니다.',
    defaultParams: {},
  },
  {
    type: 'Mish', label: 'Mish', category: 'activation', color: '#ec4899',
    formula: 'x·tanh(softplus(x))',
    description: '부드럽고 비단조적인 활성화 함수. YOLOv4에서 도입됐습니다.',
    defaultParams: {},
  },
  {
    type: 'Sigmoid', label: 'Sigmoid', category: 'activation', color: '#ec4899',
    formula: '1/(1+e⁻ˣ)',
    description: '출력을 0~1로 압축. 이진 분류의 출력층이나 게이트(LSTM)에 사용됩니다.',
    defaultParams: {},
  },
  {
    type: 'Tanh', label: 'Tanh', category: 'activation', color: '#ec4899',
    formula: '(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)',
    description: '출력을 -1~1로 압축. Sigmoid보다 기울기 소실이 적습니다.',
    defaultParams: {},
  },
  {
    type: 'Softmax', label: 'Softmax', category: 'activation', color: '#ec4899',
    formula: 'eˣⁱ/Σeˣʲ',
    description: '다중 클래스 확률 분포 출력. 분류 모델의 마지막 레이어에 사용됩니다.',
    defaultParams: { dim: -1 },
  },
  {
    type: 'HardSwish', label: 'HardSwish', category: 'activation', color: '#ec4899',
    description: 'Swish의 경량화 버전. MobileNetV3에 도입됐습니다.',
    defaultParams: {},
  },

  // ── Normalization ────────────────────────────────────────────
  {
    type: 'BatchNorm2D', label: 'BatchNorm2D', category: 'norm', color: '#14b8a6',
    description: '배치 단위로 각 채널을 정규화. 학습 안정화에 핵심적입니다. 추론 시에는 이동 평균 통계를 사용합니다.',
    defaultParams: { num_features: 64 },
  },
  {
    type: 'LayerNorm', label: 'LayerNorm', category: 'norm', color: '#14b8a6',
    description: '각 샘플의 feature 차원을 정규화. 배치 크기에 독립적이라 Transformer의 표준 정규화입니다.',
    defaultParams: { normalized_shape: 512 },
  },
  {
    type: 'GroupNorm', label: 'GroupNorm', category: 'norm', color: '#14b8a6',
    description: '채널을 그룹으로 나눠 정규화. 소배치 환경에서 BatchNorm의 대안입니다.',
    defaultParams: { num_groups: 32, num_channels: 64 },
  },
  {
    type: 'InstanceNorm', label: 'InstanceNorm', category: 'norm', color: '#14b8a6',
    description: '각 샘플·채널 독립 정규화. 스타일 전이(style transfer) 등에 사용됩니다.',
    defaultParams: { num_features: 64 },
  },
  {
    type: 'RMSNorm', label: 'RMSNorm', category: 'norm', color: '#14b8a6',
    formula: 'x/RMS(x)·γ',
    description: 'Root Mean Square 정규화. LLaMA, Mistral 등 최신 LLM에서 LayerNorm 대신 사용됩니다.',
    defaultParams: { dim: 512 },
  },

  // ── Attention ────────────────────────────────────────────────
  {
    type: 'MultiHeadAttention', label: 'Multi-Head Attention', category: 'attention', color: '#8b5cf6',
    formula: 'softmax(QKᵀ/√d)V',
    description: '여러 헤드가 병렬로 attention을 계산. 토큰 간 관계(문맥)를 학습합니다. Transformer의 핵심 연산입니다.',
    defaultParams: { embed_dim: 512, num_heads: 8 },
  },
  {
    type: 'SelfAttention', label: 'Self-Attention', category: 'attention', color: '#8b5cf6',
    description: '같은 시퀀스에서 Q, K, V를 모두 추출. 시퀀스 내 요소 간 관계를 학습합니다.',
    defaultParams: { embed_dim: 512, num_heads: 8 },
  },
  {
    type: 'CrossAttention', label: 'Cross-Attention', category: 'attention', color: '#8b5cf6',
    description: 'Q는 디코더, K·V는 인코더에서 추출. 번역, 이미지 캡셔닝 등 두 시퀀스 관계를 학습합니다.',
    defaultParams: { embed_dim: 512, num_heads: 8 },
  },
  {
    type: 'FlashAttention', label: 'Flash Attention', category: 'attention', color: '#8b5cf6',
    description: 'IO-aware attention 알고리즘. 수학적으로 동일하지만 메모리를 O(N) 대신 O(√N)만 사용해 매우 빠릅니다.',
    defaultParams: { embed_dim: 512, num_heads: 8 },
  },
  {
    type: 'GQA', label: 'GQA (Grouped Query)', category: 'attention', color: '#8b5cf6',
    description: 'Grouped Query Attention. K·V 헤드를 그룹으로 공유해 추론 속도와 메모리를 개선. LLaMA3, Mistral에서 사용됩니다.',
    defaultParams: { embed_dim: 512, num_heads: 8, num_kv_heads: 2 },
  },
  {
    type: 'MQA', label: 'MQA (Multi-Query)', category: 'attention', color: '#8b5cf6',
    description: 'Multi-Query Attention. K·V 헤드를 하나로 공유. GQA의 극단적 버전으로 추론이 매우 빠릅니다.',
    defaultParams: { embed_dim: 512, num_heads: 8 },
  },
  {
    type: 'LinearAttention', label: 'Linear Attention', category: 'attention', color: '#8b5cf6',
    formula: 'O(N) 복잡도',
    description: '표준 attention은 O(N²) 복잡도지만, 커널 함수를 이용해 O(N)으로 줄입니다. 긴 시퀀스 처리에 유리합니다.',
    defaultParams: { embed_dim: 512 },
  },

  // ── Recurrent ────────────────────────────────────────────────
  {
    type: 'LSTM', label: 'LSTM', category: 'recurrent', color: '#f97316',
    description: '장단기 기억 네트워크. forget/input/output 게이트로 장거리 의존성을 학습합니다. 시계열, NLP에 사용됩니다.',
    defaultParams: { input_size: 128, hidden_size: 256, num_layers: 1 },
  },
  {
    type: 'GRU', label: 'GRU', category: 'recurrent', color: '#f97316',
    description: 'Gated Recurrent Unit. LSTM보다 파라미터가 적고 빠릅니다.',
    defaultParams: { input_size: 128, hidden_size: 256 },
  },
  {
    type: 'RNN', label: 'RNN', category: 'recurrent', color: '#f97316',
    description: '기본 순환 신경망. 장거리 의존성 학습이 어렵지만(기울기 소실) 구조가 단순합니다.',
    defaultParams: { input_size: 128, hidden_size: 256 },
  },
  {
    type: 'BiLSTM', label: 'Bidirectional LSTM', category: 'recurrent', color: '#f97316',
    description: '양방향 LSTM. 순방향과 역방향을 합쳐 더 풍부한 문맥을 학습합니다.',
    defaultParams: { input_size: 128, hidden_size: 256 },
  },
  {
    type: 'BiGRU', label: 'Bidirectional GRU', category: 'recurrent', color: '#f97316',
    description: '양방향 GRU. BiLSTM보다 가볍습니다.',
    defaultParams: { input_size: 128, hidden_size: 256 },
  },

  // ── Embedding ────────────────────────────────────────────────
  {
    type: 'Embedding', label: 'Embedding', category: 'embedding', color: '#06b6d4',
    description: '정수 인덱스 → 밀집 벡터. 단어/토큰을 실수 벡터 공간에 매핑합니다.',
    defaultParams: { num_embeddings: 30000, embedding_dim: 512 },
  },
  {
    type: 'PatchEmbedding', label: 'Patch Embedding', category: 'embedding', color: '#06b6d4',
    description: '이미지를 패치로 나눠 선형 투영. Vision Transformer(ViT)의 입력 단계입니다.',
    defaultParams: { patch_size: 16, embed_dim: 768 },
  },
  {
    type: 'SinPE', label: 'Positional Enc (Sin)', category: 'embedding', color: '#06b6d4',
    formula: 'sin/cos(pos/10000^(2i/d))',
    description: '고정 삼각함수 위치 인코딩. 원래 Transformer 논문에서 제안됐습니다.',
    defaultParams: { d_model: 512, max_len: 5000 },
  },
  {
    type: 'LearnedPE', label: 'Positional Enc (Learned)', category: 'embedding', color: '#06b6d4',
    description: '학습 가능한 위치 임베딩. BERT, GPT 계열에서 사용됩니다.',
    defaultParams: { max_len: 512, d_model: 768 },
  },
  {
    type: 'RoPE', label: 'RoPE', category: 'embedding', color: '#06b6d4',
    description: 'Rotary Position Embedding. 상대 위치 정보를 rotation으로 인코딩. LLaMA, GPT-NeoX에서 사용됩니다.',
    defaultParams: { dim: 512 },
  },
  {
    type: 'ALiBi', label: 'ALiBi', category: 'embedding', color: '#06b6d4',
    description: 'Attention with Linear Biases. 위치 정보를 attention bias로 추가. 훈련보다 긴 시퀀스로 외삽이 가능합니다.',
    defaultParams: { num_heads: 8 },
  },

  // ── Transformer ──────────────────────────────────────────────
  {
    type: 'TransformerEncoderLayer', label: 'Transformer Encoder', category: 'transformer', color: '#a855f7',
    description: 'MHA → Add&Norm → FFN → Add&Norm. BERT, ViT의 기본 블록.',
    defaultParams: { d_model: 512, nhead: 8, dim_feedforward: 2048 },
  },
  {
    type: 'TransformerDecoderLayer', label: 'Transformer Decoder', category: 'transformer', color: '#a855f7',
    description: 'Masked MHA → Add&Norm → Cross-Attention → Add&Norm → FFN → Add&Norm. GPT, 번역 모델의 기본 블록.',
    defaultParams: { d_model: 512, nhead: 8, dim_feedforward: 2048 },
  },
  {
    type: 'FFN', label: 'Feed-Forward Network', category: 'transformer', color: '#a855f7',
    description: 'Linear → Activation → Linear. Transformer 내 position-wise FC 레이어.',
    defaultParams: { d_model: 512, d_ff: 2048 },
  },
  {
    type: 'MoE', label: 'MoE Layer', category: 'transformer', color: '#a855f7',
    description: 'Mixture of Experts. 입력마다 전문가 네트워크 중 일부만 활성화. Mixtral, Switch Transformer에서 사용됩니다.',
    defaultParams: { num_experts: 8, top_k: 2 },
  },

  // ── Shape Ops ────────────────────────────────────────────────
  {
    type: 'Flatten', label: 'Flatten', category: 'shape', color: '#64748b',
    formula: '(B,C,H,W) → (B,C·H·W)',
    description: '다차원 텐서를 1D로 펼칩니다. Conv 레이어 후 FC 레이어로 넘기기 전에 필요합니다.',
    defaultParams: { start_dim: 1 },
  },
  {
    type: 'Reshape', label: 'Reshape', category: 'shape', color: '#64748b',
    description: '텐서를 임의의 shape으로 변환. 총 원소 수는 유지됩니다.',
    defaultParams: { shape: 'B,-1' },
  },
  {
    type: 'Permute', label: 'Permute / Transpose', category: 'shape', color: '#64748b',
    description: '차원 순서를 바꿉니다. 예: (B,H,W,C) → (B,C,H,W).',
    defaultParams: { dims: '0,2,3,1' },
  },
  {
    type: 'Squeeze', label: 'Squeeze / Unsqueeze', category: 'shape', color: '#64748b',
    description: '크기 1인 차원을 제거(Squeeze) 또는 추가(Unsqueeze)합니다.',
    defaultParams: { dim: 1 },
  },
  {
    type: 'Concat', label: 'Concat', category: 'shape', color: '#64748b',
    description: '여러 텐서를 특정 차원으로 이어붙입니다. Skip connection에서 사용됩니다.',
    defaultParams: { dim: 1 },
  },
  {
    type: 'Add', label: 'Add (Residual)', category: 'shape', color: '#64748b',
    formula: 'y = F(x) + x',
    description: '잔차 연결(skip connection). ResNet의 핵심으로 기울기 소실 문제를 해결합니다.',
    defaultParams: {},
  },
  {
    type: 'Split', label: 'Split', category: 'shape', color: '#64748b',
    description: '텐서를 특정 차원으로 분리합니다.',
    defaultParams: { split_size: 1, dim: 1 },
  },

  // ── Regularization ───────────────────────────────────────────
  {
    type: 'Dropout', label: 'Dropout', category: 'regularize', color: '#94a3b8',
    description: '훈련 중 무작위로 뉴런을 비활성화. 과적합을 방지하는 가장 일반적인 기법입니다.',
    defaultParams: { p: 0.5 },
  },
  {
    type: 'Dropout2D', label: 'Dropout2D', category: 'regularize', color: '#94a3b8',
    description: '2D 특징맵에서 채널 단위로 Dropout. 공간적 상관관계를 고려합니다.',
    defaultParams: { p: 0.5 },
  },
  {
    type: 'DropPath', label: 'DropPath / StochasticDepth', category: 'regularize', color: '#94a3b8',
    description: '전체 경로(레이어)를 확률적으로 건너뜁니다. Vision Transformer, EfficientNet에서 사용됩니다.',
    defaultParams: { drop_prob: 0.1 },
  },
  {
    type: 'SpatialDropout', label: 'Spatial Dropout', category: 'regularize', color: '#94a3b8',
    description: '공간 위치 단위로 Dropout. 텍스트/이미지 특징의 공간 구조를 유지합니다.',
    defaultParams: { p: 0.2 },
  },

  // ── Linear ───────────────────────────────────────────────────
  {
    type: 'Linear', label: 'Linear (FC)', category: 'linear', color: '#3b82f6',
    formula: 'y = xW + b',
    description: '완전 연결 레이어. 모든 입력 뉴런과 출력 뉴런이 연결됩니다.',
    defaultParams: { in_features: 512, out_features: 10 },
  },
  {
    type: 'Bilinear', label: 'Bilinear', category: 'linear', color: '#3b82f6',
    formula: 'y = x₁Wx₂ + b',
    description: '두 입력 벡터를 결합. 자연어처리의 관계 모델링에 사용됩니다.',
    defaultParams: { in1: 256, in2: 256, out: 128 },
  },
]

export const BLOCK_MAP = Object.fromEntries(BLOCKS.map(b => [b.type, b]))
