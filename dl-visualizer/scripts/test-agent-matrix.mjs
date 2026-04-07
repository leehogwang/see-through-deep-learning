import fs from 'fs'
import path from 'path'
import { chromium } from 'playwright'
import { ensureDir, ensureServer, openApp, sendPrompt, waitForAgentIdle, listCanvasNodes } from './e2e-helpers.mjs'

const rootDir = path.resolve(new URL('../..', import.meta.url).pathname)
const reportDir = path.join(rootDir, 'benchmarks', 'reports')
const screenshotDir = path.join(reportDir, 'screenshots')
const screenshotPath = path.join(screenshotDir, 'agent-matrix-check.png')
const reportPath = path.join(reportDir, 'agent-matrix-latest.json')

ensureDir(screenshotDir)
await ensureServer()

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1680, height: 980 } })

const cases = []

try {
  await openApp(page)

  await sendPrompt(page, 'GELU가 무엇인지 두 문장으로 설명해줘. 답변에는 GELU라는 단어를 꼭 포함하고, 캔버스는 수정하지 마.')
  await waitForAgentIdle(page)
  const questionMessages = await page.getByTestId('agent-message').allTextContents()
  const questionAnswer = questionMessages.at(-1) || ''
  const questionNodeCount = await page.locator('.react-flow__node').count()
  cases.push({
    id: 'question_only',
    passed: questionAnswer.includes('GELU') && questionNodeCount === 0,
    answerContainsGELU: questionAnswer.includes('GELU'),
    nodeCount: questionNodeCount,
  })

  await sendPrompt(page, '빈 캔버스에 Input, Conv2D, ReLU, Conv2D를 배치하고 연결해줘. 첫 Conv2D는 in_ch=1 out_ch=16 kernel=3 padding=1, 두 번째 Conv2D는 in_ch=16 out_ch=32 kernel=3 padding=1로 설정하고 auto layout 해줘.')
  await waitForAgentIdle(page)
  const buildNodes = await listCanvasNodes(page)
  const buildWarnings = buildNodes.filter((node) => node.text.includes('⚠'))
  const buildEdges = await page.locator('.react-flow__edge').count()
  cases.push({
    id: 'build_chain',
    passed: buildNodes.length === 4 && buildEdges === 3 && buildWarnings.length === 0 && buildNodes.every((node) => node.agentEdited),
    nodeCount: buildNodes.length,
    edgeCount: buildEdges,
    warnings: buildWarnings.length,
    agentEditedCount: buildNodes.filter((node) => node.agentEdited).length,
  })

  await sendPrompt(page, '현재 그래프에서 첫 Conv2D의 out_ch를 24로 바꾸고 두 번째 Conv2D의 in_ch도 24로 맞춰줘.')
  await waitForAgentIdle(page)
  const updateTexts = (await listCanvasNodes(page)).map((node) => node.text)
  const updatedNodes = await listCanvasNodes(page)
  cases.push({
    id: 'param_update',
    passed: updateTexts.some((text) => text.includes('out_ch24')) && updateTexts.some((text) => text.includes('in_ch24')) && updatedNodes.filter((node) => node.agentEdited).length >= 2,
    convTexts: updateTexts.filter((text) => text.includes('Conv2D')),
    agentEditedCount: updatedNodes.filter((node) => node.agentEdited).length,
  })

  await sendPrompt(page, '현재 그래프에서 가운데 ReLU를 LeakyReLU(alpha=0.15)로 교체해줘.')
  await waitForAgentIdle(page)
  const replaceNodes = await listCanvasNodes(page)
  const replaceTexts = replaceNodes.map((node) => node.text)
  cases.push({
    id: 'replace_activation',
    passed: replaceTexts.some((text) => text.includes('LeakyReLU')) && !replaceTexts.some((text) => text.startsWith('GELU')) && replaceNodes.some((node) => node.text.includes('LeakyReLU') && node.agentEdited),
    texts: replaceTexts,
  })

  await sendPrompt(page, '현재 그래프에서 가운데 활성화 노드를 삭제하고, 필요하면 앞뒤를 다시 연결해줘.')
  await waitForAgentIdle(page)
  const deleteTexts = (await listCanvasNodes(page)).map((node) => node.text)
  const deleteEdges = await page.locator('.react-flow__edge').count()
  cases.push({
    id: 'delete_and_reconnect',
    passed: deleteTexts.length === 3 && deleteEdges === 2 && deleteTexts.every((text) => !text.includes('⚠')),
    nodeCount: deleteTexts.length,
    edgeCount: deleteEdges,
  })

  await sendPrompt(page, '캔버스를 비워줘.')
  await waitForAgentIdle(page)
  const clearNodeCount = await page.locator('.react-flow__node').count()
  cases.push({
    id: 'clear_canvas',
    passed: clearNodeCount === 0,
    nodeCount: clearNodeCount,
  })

  await page.screenshot({ path: screenshotPath })
} finally {
  await browser.close()
}

const summary = {
  generatedAt: new Date().toISOString(),
  passed: cases.every((item) => item.passed),
  cases,
  screenshot: screenshotPath,
}

fs.writeFileSync(reportPath, JSON.stringify(summary, null, 2))
console.log(JSON.stringify(summary, null, 2))

if (!summary.passed) {
  process.exit(1)
}
