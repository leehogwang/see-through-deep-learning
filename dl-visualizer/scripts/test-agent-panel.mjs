import fs from 'fs'
import path from 'path'
import { chromium } from 'playwright'

const rootDir = path.resolve(new URL('../..', import.meta.url).pathname)
const screenshotPath = path.join(rootDir, 'benchmarks', 'reports', 'screenshots', 'agent-panel-check.png')

async function ensureServer() {
  const res = await fetch('http://127.0.0.1:5173')
  if (!res.ok) {
    throw new Error(`server returned ${res.status}`)
  }
}

async function sendPrompt(page, prompt) {
  await page.getByTestId('agent-chat-input').fill(prompt)
  await page.getByTestId('agent-chat-send').click()
}

async function waitForIdle(page) {
  await page.waitForFunction(() => {
    const button = document.querySelector('[data-testid="agent-chat-send"]')
    return button && button.textContent?.trim() === '↑'
  }, undefined, { timeout: 240000 })
}

async function listNodeTexts(page) {
  return page.locator('.react-flow__node').evaluateAll((nodes) => (
    nodes.map((node) => ({
      text: (node.textContent || '').trim(),
      agentEdited: node.querySelector('[data-agent-edited="true"]') !== null,
    }))
  ))
}

async function nodeBox(page, text) {
  const locator = page.locator('.react-flow__node').filter({ hasText: text }).first()
  return locator.boundingBox()
}

await ensureServer()
fs.mkdirSync(path.dirname(screenshotPath), { recursive: true })

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1680, height: 980 } })

try {
  await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' })

  await sendPrompt(page, 'GELU가 무엇인지 두 문장으로 설명해줘. 답변에는 GELU라는 단어를 꼭 포함하고, 캔버스는 수정하지 마.')
  await waitForIdle(page)
  const agentMessagesAfterQuestion = await page.getByTestId('agent-message').allTextContents()
  const latestAnswer = agentMessagesAfterQuestion.at(-1) || ''
  if (!latestAnswer.includes('GELU')) {
    throw new Error(`expected GELU in latest agent answer, got: ${latestAnswer}`)
  }
  const nodeCountAfterQuestion = await page.locator('.react-flow__node').count()
  if (nodeCountAfterQuestion !== 0) {
    throw new Error(`expected question-only prompt to keep canvas empty, found ${nodeCountAfterQuestion} nodes`)
  }

  await sendPrompt(
    page,
    '빈 캔버스에 정확히 4개 노드만 배치해줘: Input, Conv2D, GELU, Conv2D. 이 순서대로 연결하고, 첫 Conv2D는 in_ch=1 out_ch=16 kernel=3 padding=1, 두 번째 Conv2D는 in_ch=16 out_ch=32 kernel=3 padding=1로 설정해. 마지막에 auto_layout을 해줘.',
  )
  await waitForIdle(page)

  const nodeStatesAfterBuild = await listNodeTexts(page)
  const edgeCountAfterBuild = await page.locator('.react-flow__edge').count()
  if (nodeStatesAfterBuild.length !== 4) {
    throw new Error(`expected 4 nodes after agent build, found ${nodeStatesAfterBuild.length}`)
  }
  if (edgeCountAfterBuild !== 3) {
    throw new Error(`expected 3 edges after agent build, found ${edgeCountAfterBuild}`)
  }
  if (!nodeStatesAfterBuild.some((node) => node.text.includes('Input'))) throw new Error('Input node missing after agent build')
  if (!nodeStatesAfterBuild.some((node) => node.text.includes('GELU'))) throw new Error('GELU node missing after agent build')
  if (nodeStatesAfterBuild.some((node) => node.text.includes('⚠'))) {
    throw new Error(`expected clean graph after agent build, found warnings: ${nodeStatesAfterBuild.map((node) => node.text).join(' || ')}`)
  }
  if (!nodeStatesAfterBuild.every((node) => node.agentEdited)) {
    throw new Error('expected all agent-created nodes to be marked as agent edited')
  }

  await sendPrompt(
    page,
    '현재 그래프에서 GELU를 LeakyReLU(alpha=0.2)로 바꾸고, Input 노드의 y 좌표가 다른 노드들보다 작아지도록 위로 옮겨줘.',
  )
  await waitForIdle(page)

  const nodeStatesAfterEdit = await listNodeTexts(page)
  const inputBox = await nodeBox(page, 'Input')
  const convBox = await nodeBox(page, 'Conv2D')
  if (!nodeStatesAfterEdit.some((node) => node.text.includes('LeakyReLU'))) {
    throw new Error('LeakyReLU node missing after agent edit')
  }
  if (nodeStatesAfterEdit.some((node) => node.text.includes('GELU'))) {
    throw new Error('GELU node still present after replacement')
  }
  if (!inputBox || !convBox) {
    throw new Error('failed to measure node positions after agent edit')
  }
  if (!(inputBox.y < convBox.y)) {
    throw new Error(`expected Input to move above Conv2D, got input.y=${inputBox.y}, conv.y=${convBox.y}`)
  }
  if (nodeStatesAfterEdit.some((node) => node.text.includes('⚠'))) {
    throw new Error(`expected clean graph after agent edit, found warnings: ${nodeStatesAfterEdit.map((node) => node.text).join(' || ')}`)
  }
  if (!nodeStatesAfterEdit.some((node) => node.text.includes('LeakyReLU') && node.agentEdited)) {
    throw new Error('expected replaced LeakyReLU node to be marked as agent edited')
  }

  const systemMessages = await page.getByTestId('agent-system-message').allTextContents()
  await page.screenshot({ path: screenshotPath })

  console.log(JSON.stringify({
    latestAnswer,
    nodeCountAfterQuestion,
    nodeStatesAfterBuild,
    edgeCountAfterBuild,
    nodeStatesAfterEdit,
    inputY: inputBox.y,
    convY: convBox.y,
    systemMessages,
    screenshot: screenshotPath,
  }, null, 2))
} finally {
  await browser.close()
}
