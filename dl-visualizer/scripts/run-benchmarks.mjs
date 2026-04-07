import fs from 'fs'
import path from 'path'
import { spawn, execFileSync } from 'child_process'
import { chromium } from 'playwright'
import yaml from 'yaml'

const projectDir = path.resolve(new URL('..', import.meta.url).pathname)
const rootDir = path.resolve(projectDir, '..')
const manifestPath = path.join(rootDir, 'benchmarks', 'manifest.yaml')
const reportDir = path.join(rootDir, 'benchmarks', 'reports')
const screenshotDir = path.join(reportDir, 'screenshots')
const manifest = yaml.parse(fs.readFileSync(manifestPath, 'utf8'))

fs.mkdirSync(reportDir, { recursive: true })
fs.mkdirSync(screenshotDir, { recursive: true })
for (const entry of fs.readdirSync(screenshotDir)) {
  if (entry.endsWith('.png')) {
    fs.rmSync(path.join(screenshotDir, entry), { force: true })
  }
}

function freePort(port) {
  try {
    const pids = execFileSync('bash', ['-lc', `lsof -ti :${port}`], { encoding: 'utf8' })
      .trim()
      .split('\n')
      .filter(Boolean)
    for (const pid of pids) {
      process.kill(Number(pid), 'SIGTERM')
    }
  } catch {
    // No existing listener is fine.
  }
}

freePort(5173)

const server = spawn('npm', ['start'], { cwd: projectDir, stdio: 'pipe' })
let serverOutput = ''
server.stdout.on('data', (chunk) => { serverOutput += chunk.toString() })
server.stderr.on('data', (chunk) => { serverOutput += chunk.toString() })

async function waitForServer() {
  const startedAt = Date.now()
  while (Date.now() - startedAt < 30000) {
    try {
      const res = await fetch('http://127.0.0.1:5173/api/benchmarks')
      const contentType = res.headers.get('content-type') || ''
      if (!res.ok || !contentType.includes('application/json')) {
        throw new Error(`unexpected content-type: ${contentType}`)
      }
      const payload = await res.json()
      if (Array.isArray(payload)) return
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error(`server did not start in time\n${serverOutput}`)
}

function cleanup() {
  if (!server.killed) server.kill('SIGINT')
}

async function dragBlockOntoCanvas(page, blockTestId, canvasLocator, xOffset, yOffset) {
  const block = page.getByTestId(blockTestId)
  await block.dragTo(canvasLocator, {
    targetPosition: {
      x: xOffset,
      y: yOffset,
    },
  })
  await page.waitForTimeout(250)
}

async function listCanvasNodes(page) {
  return await page.locator('.react-flow__node').evaluateAll((elements) => (
    elements.map((element) => ({
      id: element.getAttribute('data-id') || '',
      text: (element.textContent || '').trim(),
    }))
  ))
}

async function connectNodes(page, sourceId, targetId) {
  await page.getByTestId(`connect-node-${sourceId}`).click()
  await page.getByTestId(`node-${targetId}`).click()
  await page.waitForTimeout(400)
}

process.on('exit', cleanup)
process.on('SIGINT', () => {
  cleanup()
  process.exit(1)
})

await waitForServer()

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } })

const report = {
  generatedAt: new Date().toISOString(),
  benchmarks: [],
  editFlow: null,
}

for (const benchmark of (manifest.benchmarks || []).filter((item) => item.official !== false)) {
  console.log(`running benchmark: ${benchmark.id}`)
  await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' })
  await page.getByRole('button', { name: /Open Project/i }).click()
  await page.getByTestId(`benchmark-card-${benchmark.id}`).click()
  await page.getByTestId('loaded-model-name').waitFor({ timeout: 120000 })
  await page.waitForTimeout(1500)

  const nodeCount = await page.locator('.react-flow__node').count()
  const errorCount = await page.locator('[data-diagnostic-severity="error"]').count()
  const warningCount = await page.locator('[data-diagnostic-severity="warning"]').count()
  const traceMode = await page.getByTestId('trace-mode').textContent()
  const exactness = await page.getByTestId('trace-exactness').textContent()
  const sampleSource = await page.getByTestId('sample-preview-source').textContent()
  const sampleStrategy = await page.getByTestId('sample-preview-strategy').textContent()
  const sampleResolution = await page.getByTestId('sample-preview-resolution').textContent()
  const samplePath = await page.getByTestId('sample-preview-path').textContent()
  const sampleEvidence = await page.getByTestId('sample-preview-evidence').textContent()
  const unsupportedReason = await page.getByTestId('trace-unsupported-reason').textContent().catch(() => '')
  await page.screenshot({ path: path.join(screenshotDir, `${benchmark.id}.png`) })
  await page.getByTestId('drawer-tab-diagnostics').click()
  await page.waitForTimeout(300)
  const diagnostics = await page.locator('[data-testid="diagnostic-item"]').count()
  const errorDiagnostics = await page
    .locator('[data-testid="diagnostic-item"][data-severity="error"]')
    .evaluateAll((elements) => elements.map((element) => ({
      severity: element.getAttribute('data-severity') || 'error',
      code: element.getAttribute('data-code') || 'unknown',
      nodeLabel: element.getAttribute('data-node-label') || 'unknown',
      title: element.getAttribute('data-title') || 'Untitled diagnostic',
      detail: element.getAttribute('data-detail') || '',
    })))
  await page.screenshot({ path: path.join(screenshotDir, `${benchmark.id}-diagnostics.png`) })

  report.benchmarks.push({
    id: benchmark.id,
    level: benchmark.level,
    label: benchmark.label,
    modelName: benchmark.model_name,
    nodeCount,
    errorCount,
    warningCount,
    diagnostics,
    traceMode,
    exactness,
    unsupportedReason,
    sampleSource,
    sampleStrategy,
    sampleResolution,
    samplePath,
    sampleEvidence,
    errorDiagnostics,
    screenshot: `benchmarks/reports/screenshots/${benchmark.id}.png`,
    diagnosticsScreenshot: `benchmarks/reports/screenshots/${benchmark.id}-diagnostics.png`,
  })
}

await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' })
const manualCanvas = page.getByTestId('flow-canvas')
await dragBlockOntoCanvas(page, 'block-Input', manualCanvas, 140, 210)
await dragBlockOntoCanvas(page, 'block-Conv2D', manualCanvas, 360, 210)
await dragBlockOntoCanvas(page, 'block-ReLU', manualCanvas, 600, 210)
await dragBlockOntoCanvas(page, 'block-Conv2D', manualCanvas, 840, 210)

const initialNodes = await listCanvasNodes(page)
if (initialNodes.length < 4) {
  throw new Error(`manual edit flow expected 4 nodes, found ${initialNodes.length}`)
}

const [inputNode, convNodeA, reluNode, convNodeB] = initialNodes
await connectNodes(page, inputNode.id, convNodeA.id)
await connectNodes(page, convNodeA.id, reluNode.id)
await connectNodes(page, reluNode.id, convNodeB.id)

const initialEdgeCount = await page.locator('.react-flow__edge').count()
await page.getByTestId(`delete-node-${reluNode.id}`).click()
await page.waitForTimeout(500)

const nodesAfterDelete = await listCanvasNodes(page)
const edgeCountAfterDelete = await page.locator('.react-flow__edge').count()
const reluDeleted = !nodesAfterDelete.some((node) => node.id === reluNode.id)

await dragBlockOntoCanvas(page, 'block-LeakyReLU', manualCanvas, 600, 210)
const nodesAfterLeaky = await listCanvasNodes(page)
const leakyNode = nodesAfterLeaky.find((node) => !nodesAfterDelete.some((candidate) => candidate.id === node.id))
if (!leakyNode) {
  throw new Error('failed to identify inserted LeakyReLU node')
}

await connectNodes(page, convNodeA.id, leakyNode.id)
await connectNodes(page, leakyNode.id, convNodeB.id)
const edgeCountAfterReconnect = await page.locator('.react-flow__edge').count()
await page.screenshot({ path: path.join(screenshotDir, 'manual-edit-flow.png') })

await page.getByTestId('generate-code-button').click()
await page.getByText('Generated PyTorch Code', { exact: true }).waitFor({ timeout: 10000 })
const manualGeneratedCode = await page.locator('pre').textContent()
await page.getByRole('button', { name: /Close/i }).click()

await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' })
await page.getByRole('button', { name: /Open Project/i }).click()
await page.getByTestId('benchmark-card-level2-resnet18').click()
await page.getByTestId('loaded-model-name').waitFor({ timeout: 120000 })
await page.waitForTimeout(1200)

const block = page.getByTestId('block-ReLU')
const canvas = page.getByTestId('flow-canvas')
await dragBlockOntoCanvas(page, 'block-ReLU', canvas, 400, 220)
await page.waitForTimeout(1000)
await page.getByTestId('generate-code-button').click()
await page.getByText('Generated PyTorch Code', { exact: true }).waitFor({ timeout: 10000 })
const generatedCode = await page.locator('pre').textContent()
await page.getByRole('button', { name: /Close/i }).click()
await page.getByTestId('save-worktree-button').click()
await page.getByText(/saved to branch/i).waitFor({ timeout: 20000 })

const paletteSearchInput = page.getByTestId('block-search-input')
await paletteSearchInput.fill('relu')
await page.waitForTimeout(300)
const reluMatches = await page.locator('[data-testid^="block-"]').count()
await paletteSearchInput.fill('')
await page.getByTestId('category-toggle-activation').click()
await page.waitForTimeout(200)
const activationVisibleAfterCollapse = await page.getByTestId('block-ReLU').isVisible().catch(() => false)
await page.getByTestId('category-toggle-activation').click()
await page.waitForTimeout(200)

const codexEditResponse = await fetch('http://127.0.0.1:5173/api/codex-edit/apply', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    repoRoot: path.join(rootDir, 'benchmarks', 'vendor', 'pytorch-vision'),
    sourceFile: path.join(rootDir, 'benchmarks', 'vendor', 'pytorch-vision', 'torchvision', 'models', 'resnet.py'),
    instruction: 'Insert a single-line comment `# codex benchmark marker` immediately above the `class ResNet` definition. Do not change any other file.',
  }),
})

if (!codexEditResponse.ok) {
  throw new Error(`codex edit failed: ${await codexEditResponse.text()}`)
}
const codexEditResult = await codexEditResponse.json()

report.editFlow = {
  manualCanvas: {
    insertedNodes: initialNodes.map((node) => node.text.split('\n')[0] || node.text),
    initialEdgeCount,
    reluDeleted,
    edgeCountAfterDelete,
    edgeCountAfterReconnect,
    replacementNode: leakyNode.text.split('\n')[0] || leakyNode.text,
    generatedCodeContainsLeakyReLU: Boolean(manualGeneratedCode?.includes('LeakyReLU')),
    screenshot: 'benchmarks/reports/screenshots/manual-edit-flow.png',
  },
  benchmarkId: 'level2-resnet18',
  draggedBlock: 'ReLU',
  codeGenerated: Boolean(generatedCode?.trim()),
  saveToWorktree: true,
  paletteSearch: {
    query: 'relu',
    visibleMatches: reluMatches,
    activationCollapsedHidesReLU: !activationVisibleAfterCollapse,
  },
  sourceAwareEdit: {
    success: codexEditResult.success,
    branch: codexEditResult.branch,
    commit: codexEditResult.commit,
    changedFiles: codexEditResult.changedFiles,
    diffSummary: String(codexEditResult.diffSummary || '').split('\n').slice(0, 20).join('\n'),
  },
}

await browser.close()
cleanup()

fs.writeFileSync(path.join(reportDir, 'latest.json'), JSON.stringify(report, null, 2))

const md = [
  '# Benchmark Report',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '| Benchmark | Level | Nodes | Errors | Warnings | Diagnostics | Sample | Strategy | Resolution |',
  '| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |',
  ...report.benchmarks.map((item) => (
    `| ${item.label} | ${item.level} | ${item.nodeCount} | ${item.errorCount} | ${item.warningCount} | ${item.diagnostics} | ${item.sampleSource ?? 'n/a'} | ${item.sampleStrategy ?? 'n/a'} | ${item.sampleResolution ?? 'unknown'} |`
  )),
  '',
  '## Trace Modes',
  '',
  '| Benchmark | Trace Mode | Exactness | Unsupported Reason |',
  '| --- | --- | --- | --- |',
  ...report.benchmarks.map((item) => (
    `| ${item.label} | ${item.traceMode ?? 'n/a'} | ${item.exactness ?? 'n/a'} | ${(item.unsupportedReason || '').replace(/\|/g, '\\|') || '—'} |`
  )),
  '',
  '## Sample Provenance',
  '',
  ...report.benchmarks.map((item) => [
    `### ${item.label}`,
    `- source: ${item.sampleSource ?? 'n/a'}`,
    `- strategy: ${item.sampleStrategy ?? 'n/a'}`,
    `- path: ${item.samplePath ?? 'n/a'}`,
    `- resolution: ${item.sampleResolution ?? 'unknown'}`,
    `- evidence: ${item.sampleEvidence ?? 'n/a'}`,
    '',
  ]).flat(),
  '',
  '## Error Diagnostics',
  '',
  ...report.benchmarks.flatMap((item) => (
    item.errorDiagnostics?.length
      ? [
          `### ${item.label}`,
          ...item.errorDiagnostics.map((entry) => `- ${entry.nodeLabel} [${entry.code}] ${entry.title}: ${entry.detail}`),
          '',
        ]
      : []
  )),
  '',
  '## Edit Flow',
  '',
  '### Manual Delete + Reconnect',
  '',
  `- inserted nodes: ${report.editFlow.manualCanvas.insertedNodes.join(' -> ')}`,
  `- initial edge count: ${report.editFlow.manualCanvas.initialEdgeCount}`,
  `- ReLU deleted via x button: ${report.editFlow.manualCanvas.reluDeleted}`,
  `- edge count after delete: ${report.editFlow.manualCanvas.edgeCountAfterDelete}`,
  `- edge count after reconnect: ${report.editFlow.manualCanvas.edgeCountAfterReconnect}`,
  `- replacement node: ${report.editFlow.manualCanvas.replacementNode}`,
  `- generated code contains LeakyReLU: ${report.editFlow.manualCanvas.generatedCodeContainsLeakyReLU}`,
  `- screenshot: ${report.editFlow.manualCanvas.screenshot}`,
  '',
  '### Loaded Benchmark Edit',
  '',
  `- benchmark: ${report.editFlow.benchmarkId}`,
  `- dragged block: ${report.editFlow.draggedBlock}`,
  `- code generated: ${report.editFlow.codeGenerated}`,
  `- save to worktree: ${report.editFlow.saveToWorktree}`,
  `- block palette search 'relu' matches: ${report.editFlow.paletteSearch.visibleMatches}`,
  `- activation collapse hides ReLU: ${report.editFlow.paletteSearch.activationCollapsedHidesReLU}`,
  `- codex source-aware edit commit: ${report.editFlow.sourceAwareEdit.commit}`,
  `- codex source-aware edit branch: ${report.editFlow.sourceAwareEdit.branch}`,
  `- codex changed files: ${report.editFlow.sourceAwareEdit.changedFiles.join(', ')}`,
  '',
  '### Codex Source-aware Edit Diff',
  '',
  '```diff',
  report.editFlow.sourceAwareEdit.diffSummary,
  '```',
]
fs.writeFileSync(path.join(reportDir, 'latest.md'), md.join('\n'))
process.exit(0)
