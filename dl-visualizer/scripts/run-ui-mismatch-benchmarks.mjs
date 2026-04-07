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
  sendPrompt,
  waitForAgentIdle,
} from './e2e-helpers.mjs'

const rootDir = path.resolve(new URL('../..', import.meta.url).pathname)
const reportDir = path.join(rootDir, 'benchmarks', 'reports')
const screenshotDir = path.join(reportDir, 'screenshots')
const jsonPath = path.join(reportDir, 'ui-mismatch-latest.json')
const mdPath = path.join(reportDir, 'ui-mismatch-latest.md')

ensureDir(reportDir)
ensureDir(screenshotDir)
await ensureServer()

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1680, height: 980 } })

const cases = []

try {
  await openApp(page)
  const canvas = page.getByTestId('flow-canvas')

  await dragBlockOntoCanvas(page, 'block-Input', canvas, 120, 180)
  await dragBlockOntoCanvas(page, 'block-Conv2D', canvas, 360, 180)
  await dragBlockOntoCanvas(page, 'block-GELU', canvas, 600, 180)
  await dragBlockOntoCanvas(page, 'block-Conv2D', canvas, 840, 180)

  const convNodes = await listCanvasNodes(page)
  const [inputNode, conv1, geluNode, conv2] = convNodes
  await editParam(page, inputNode.id, 'shape', 'B,1,64,64')
  await editParam(page, conv1.id, 'in_ch', 1)
  await editParam(page, conv1.id, 'out_ch', 12)
  await editParam(page, conv1.id, 'kernel', '5,5')
  await editParam(page, conv2.id, 'in_ch', 12)
  await editParam(page, conv2.id, 'out_ch', 24)
  await editParam(page, conv2.id, 'kernel', '3,3')
  await connectNodes(page, inputNode.id, conv1.id)
  await connectNodes(page, conv1.id, geluNode.id)
  await connectNodes(page, geluNode.id, conv2.id)
  await page.getByTestId('generate-code-button').click()
  await page.getByText('Generated PyTorch Code', { exact: true }).waitFor({ timeout: 10000 })
  const convCode = await page.getByTestId('generated-code-output').textContent()
  await page.getByRole('button', { name: /Close/i }).click()
  const convTexts = (await listCanvasNodes(page)).map((node) => node.text)
  const convWarnings = convTexts.filter((text) => text.includes('⚠'))
  const convCaseScreenshot = path.join(screenshotDir, 'ui-mismatch-conv.png')
  await page.screenshot({ path: convCaseScreenshot })
  cases.push({
    id: 'conv_shape_codegen_consistency',
    passed: convWarnings.length === 0
      && convTexts.some((text) => text.includes('shapeB,1,64,64'))
      && convTexts.some((text) => text.includes('out_ch12'))
      && convTexts.some((text) => text.includes('out_ch24'))
      && Boolean(convCode?.includes('nn.Conv2d(1, 12, kernel_size=(5,5)'))
      && Boolean(convCode?.includes('nn.Conv2d(12, 24, kernel_size=(3,3)')),
    warnings: convWarnings.length,
    screenshot: convCaseScreenshot,
  })

  await page.getByRole('button', { name: 'Clear' }).click()
  await page.waitForTimeout(250)

  await dragBlockOntoCanvas(page, 'block-Input', canvas, 140, 200)
  await dragBlockOntoCanvas(page, 'block-MultiHeadAttention', canvas, 420, 200)
  await dragBlockOntoCanvas(page, 'block-Output', canvas, 760, 200)
  const attentionNodes = await listCanvasNodes(page)
  const [seqInput, mhaNode, seqOutput] = attentionNodes
  await editParam(page, seqInput.id, 'shape', 'B,16,128')
  await editParam(page, mhaNode.id, 'embed_dim', 128)
  await editParam(page, mhaNode.id, 'num_heads', 8)
  await connectNodes(page, seqInput.id, mhaNode.id)
  await connectNodes(page, mhaNode.id, seqOutput.id)
  await page.getByTestId('generate-code-button').click()
  await page.getByText('Generated PyTorch Code', { exact: true }).waitFor({ timeout: 10000 })
  const attentionCode = await page.getByTestId('generated-code-output').textContent()
  await page.getByRole('button', { name: /Close/i }).click()
  const attentionTexts = (await listCanvasNodes(page)).map((node) => node.text)
  const attentionWarnings = attentionTexts.filter((text) => text.includes('⚠'))
  const attentionCaseScreenshot = path.join(screenshotDir, 'ui-mismatch-attention.png')
  await page.screenshot({ path: attentionCaseScreenshot })
  cases.push({
    id: 'attention_shape_codegen_consistency',
    passed: attentionWarnings.length === 0
      && attentionTexts.some((text) => text.includes('embed_dim128'))
      && attentionTexts.some((text) => text.includes('num_heads8'))
      && Boolean(attentionCode?.includes('nn.MultiheadAttention(128, 8, batch_first=True)')),
    warnings: attentionWarnings.length,
    screenshot: attentionCaseScreenshot,
  })

  await page.getByRole('button', { name: 'Clear' }).click()
  await page.waitForTimeout(250)

  await sendPrompt(page, '빈 캔버스에 Input, Conv2D, GELU, Conv2D를 배치하고 연결해줘. 첫 Conv2D는 in_ch=1 out_ch=16 kernel=3 padding=1, 두 번째 Conv2D는 in_ch=16 out_ch=32 kernel=3 padding=1로 설정하고 auto layout 해줘.')
  await waitForAgentIdle(page)
  await sendPrompt(page, '현재 그래프에서 GELU를 LeakyReLU(alpha=0.2)로 바꾸고 Input 노드를 위로 옮겨줘.')
  await waitForAgentIdle(page)
  await page.getByTestId('generate-code-button').click()
  await page.getByText('Generated PyTorch Code', { exact: true }).waitFor({ timeout: 10000 })
  const agentCode = await page.getByTestId('generated-code-output').textContent()
  await page.getByRole('button', { name: /Close/i }).click()
  const agentTexts = (await listCanvasNodes(page)).map((node) => node.text)
  const agentWarnings = agentTexts.filter((text) => text.includes('⚠'))
  const agentCaseScreenshot = path.join(screenshotDir, 'ui-mismatch-agent.png')
  await page.screenshot({ path: agentCaseScreenshot })
  cases.push({
    id: 'agent_ui_shape_codegen_consistency',
    passed: agentWarnings.length === 0
      && agentTexts.some((text) => text.includes('LeakyReLU'))
      && !agentTexts.some((text) => text.includes('GELU↗'))
      && Boolean(agentCode?.includes('nn.LeakyReLU(0.2)')),
    warnings: agentWarnings.length,
    screenshot: agentCaseScreenshot,
  })
} finally {
  await browser.close()
}

const report = {
  generatedAt: new Date().toISOString(),
  passed: cases.every((item) => item.passed),
  cases,
}

fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2))

const markdown = [
  '# UI / Shape / Codegen Mismatch Benchmarks',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '| Case | Passed | Warnings | Screenshot |',
  '| --- | --- | ---: | --- |',
  ...cases.map((item) => `| ${item.id} | ${item.passed ? 'yes' : 'no'} | ${item.warnings ?? 0} | ${path.relative(rootDir, item.screenshot)} |`),
  '',
]
fs.writeFileSync(mdPath, markdown.join('\n'))

console.log(JSON.stringify(report, null, 2))

if (!report.passed) {
  process.exit(1)
}
