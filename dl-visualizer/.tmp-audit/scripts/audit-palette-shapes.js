import { getBlockCatalog } from '../src/data/blocks.js';
import { calcOutputShape } from '../src/lib/shapeCalculator.js';
const CANDIDATES = [
    { label: 'image-2d', shape: [-1, 3, 224, 224] },
    { label: 'feature-2d', shape: [-1, 512] },
    { label: 'sequence-3d', shape: [-1, 16, 512] },
    { label: 'signal-1d', shape: [-1, 16, 256] },
    { label: 'video-3d', shape: [-1, 3, 16, 224, 224] },
];
function normalizeBlockTypeName(blockType) {
    return blockType.toLowerCase().replace(/[^a-z0-9]+/g, '');
}
function parseNumericParam(value, fallback) {
    if (typeof value === 'number')
        return value;
    if (typeof value === 'string') {
        const parsed = Number(value.trim());
        return Number.isFinite(parsed) ? parsed : fallback;
    }
    return fallback;
}
function chooseInputShapes(block, candidate) {
    const maxInputs = block.maxInputs ?? 1;
    return Array.from({ length: maxInputs }, () => [...candidate.shape]);
}
function autoTuneParams(block, inputShape) {
    const params = { ...block.defaultParams };
    const type = normalizeBlockTypeName(block.type);
    const channelDim = inputShape[1];
    const featureDim = inputShape[inputShape.length - 1];
    if (type === 'conv1d' || type === 'conv2d' || type === 'conv3d' || type === 'depthwiseconv2d' || type === 'dilatedconv2d' || type === 'transposedconv2d') {
        if (channelDim > 0)
            params.in_ch = channelDim;
    }
    if (type === 'batchnorm2d' || type === 'instancenorm') {
        if (channelDim > 0)
            params.num_features = channelDim;
    }
    if (type === 'groupnorm') {
        if (channelDim > 0)
            params.num_channels = channelDim;
        const groups = parseNumericParam(params.num_groups, 1);
        if (channelDim > 0 && groups > channelDim)
            params.num_groups = channelDim;
    }
    if (type === 'layernorm' || type === 'rmsnorm') {
        if (featureDim > 0)
            params.normalized_shape = featureDim;
    }
    if (type === 'linear') {
        if (featureDim > 0)
            params.in_features = featureDim;
    }
    if (type === 'multiheadattention' || type === 'selfattention' || type === 'crossattention' || type === 'flashattention' || type === 'gqa' || type === 'mqa' || type === 'linearattention') {
        if (featureDim > 0)
            params.embed_dim = featureDim;
        let heads = parseNumericParam(params.num_heads, 8);
        while (heads > 1 && featureDim > 0 && featureDim % heads !== 0)
            heads -= 1;
        params.num_heads = Math.max(1, heads);
    }
    if (type === 'transformerencoderlayer' || type === 'transformerdecoderlayer') {
        if (featureDim > 0)
            params.d_model = featureDim;
        let heads = parseNumericParam(params.nhead, 8);
        while (heads > 1 && featureDim > 0 && featureDim % heads !== 0)
            heads -= 1;
        params.nhead = Math.max(1, heads);
    }
    if (type === 'lstm' || type === 'gru' || type === 'rnn' || type === 'bilstm' || type === 'bigru') {
        if (featureDim > 0)
            params.input_size = featureDim;
    }
    return params;
}
function auditBlock(block) {
    if (block.type === 'Input') {
        return { block, matched: { label: 'self', shape: [-1, 3, 224, 224] } };
    }
    let lastError = 'no candidate matched';
    for (const candidate of CANDIDATES) {
        const inputShapes = chooseInputShapes(block, candidate);
        const params = autoTuneParams(block, candidate.shape);
        const result = calcOutputShape(block.type, params, inputShapes[0] ?? null, inputShapes);
        if (!result.error) {
            return { block, matched: candidate };
        }
        lastError = result.str;
    }
    return { block, error: lastError };
}
const blocks = getBlockCatalog().filter((block) => block.type !== 'Output');
const failures = blocks
    .map(auditBlock)
    .filter((result) => result.error)
    .sort((left, right) => left.block.category.localeCompare(right.block.category) || left.block.type.localeCompare(right.block.type));
const successes = blocks.length - failures.length;
console.log(JSON.stringify({
    total: blocks.length,
    successes,
    failures: failures.map((result) => ({
        type: result.block.type,
        label: result.block.label,
        category: result.block.category,
        error: result.error,
    })),
}, null, 2));
