import fs from 'fs'
import path from 'path'
import { chromium } from 'playwright'
import {
  ensureDir,
  ensureServer,
  openApp,
  dragBlockOntoCanvas,
  listCanvasNodes,
  connectNodes,
  editParam,
  nodeText,
  waitForTextToDisappear,
} from './e2e-helpers.mjs'

const rootDir = path.resolve(new URL('../..', import.meta.url).pathname)
const reportDir = path.join(rootDir, 'benchmarks', 'reports')
const screenshotDir = path.join(reportDir, 'screenshots')
const screenshotPath = path.join(screenshotDir, 'canvas-e2e-check.png')
const reportPath = path.join(reportDir, 'canvas-e2e-latest.json')

ensureDir(screenshotDir)
await ensureServer()

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1680, height: 980 } })

try {
  await openApp(page)
  const canvas = page.getByTestId('flow-canvas')
  const searchInput = page.getByTestId('block-search-input')

  await searchInput.fill('multihead')
  const attentionVisible = await page.getByTestId('block-MultiHeadAttention').isVisible()
  await searchInput.fill('')
  if (!attentionVisible) {
    throw new Error('MultiHeadAttention block was not visible in palette search')
  }

  await dragBlockOntoCanvas(page, 'block-Input', canvas, 120, 180)
  await dragBlockOntoCanvas(page, 'block-MultiHeadAttention', canvas, 380, 180)
  await dragBlockOntoCanvas(page, 'block-Output', canvas, 720, 180)

  const nodes = await listCanvasNodes(page)
  if (nodes.length < 3) {
    throw new Error(`expected at least 3 nodes, found ${nodes.length}`)
  }
  const [inputNode, attentionNode, outputNode] = nodes

  await editParam(page, inputNode.id, 'shape', 'B,16,128')
  await editParam(page, attentionNode.id, 'embed_dim', 128)
  await editParam(page, attentionNode.id, 'num_heads', 8)

  await connectNodes(page, inputNode.id, attentionNode.id)
  await waitForTextToDisappear(page, attentionNode.id, '⚠')
  await connectNodes(page, attentionNode.id, outputNode.id)
  await waitForTextToDisappear(page, outputNode.id, '⚠')

  const attentionTextAfterConnect = await nodeText(page, attentionNode.id)
  const outputTextAfterConnect = await nodeText(page, outputNode.id)
  if (attentionTextAfterConnect.includes('⚠') || outputTextAfterConnect.includes('⚠')) {
    throw new Error('attention chain still has warnings after handle-based reconnect')
  }

  await editParam(page, attentionNode.id, 'num_heads', 7)
  const invalidAttentionText = await nodeText(page, attentionNode.id)
  if (!invalidAttentionText.includes('not divisible by num_heads')) {
    throw new Error(`expected divisibility warning after invalid num_heads edit, got: ${invalidAttentionText}`)
  }
  await page.getByTestId('drawer-tab-diagnostics').click()
  const healthPanelWarning = (await page.getByTestId('graph-health-panel').textContent()) || ''
  if (!healthPanelWarning.includes('Warnings Present')) {
    throw new Error(`expected warning health panel, got: ${healthPanelWarning}`)
  }
  await page.getByTestId('recovery-auto-layout').click()
  const statusAfterAutoLayout = (await page.getByTestId('surface-status-message').textContent()) || ''
  if (!statusAfterAutoLayout.includes('Auto layout applied')) {
    throw new Error(`expected auto layout surface status, got: ${statusAfterAutoLayout}`)
  }

  await editParam(page, attentionNode.id, 'num_heads', 8)
  await waitForTextToDisappear(page, attentionNode.id, 'not divisible by num_heads')
  const healthPanelRecovered = (await page.getByTestId('graph-health-panel').textContent()) || ''
  if (!healthPanelRecovered.includes('Healthy')) {
    throw new Error(`expected healthy status after recovery, got: ${healthPanelRecovered}`)
  }

  await page.getByTestId('generate-code-button').click()
  await page.getByText('Generated PyTorch Code', { exact: true }).waitFor({ timeout: 10000 })
  const generatedCode = await page.getByTestId('generated-code-output').textContent()
  await page.getByRole('button', { name: /Close/i }).click()

  if (!generatedCode?.includes('nn.MultiheadAttention(128, 8, batch_first=True)')) {
    throw new Error('generated code did not reflect edited MultiHeadAttention params')
  }

  await page.getByTestId('recovery-clear-graph').click()
  await page.waitForTimeout(300)
  const remainingNodes = await page.locator('.react-flow__node').count()
  if (remainingNodes !== 0) {
    throw new Error(`expected clear button to remove all nodes, found ${remainingNodes}`)
  }

  await page.screenshot({ path: screenshotPath })

  const report = {
    generatedAt: new Date().toISOString(),
    attentionVisible,
    attentionTextAfterConnect,
    outputTextAfterConnect,
    generatedCodeContainsMha: true,
    clearedCanvas: true,
    screenshot: screenshotPath,
  }

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))
} finally {
  await browser.close()
}
