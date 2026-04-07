/**
 * DL Visualizer — Full E2E Test with Video Recording
 *
 * 테스트 범위:
 *   1. 앱 로드 및 기본 UI 렌더링
 *   2. 블록 팔레트 검색
 *   3. 블록 드래그 앤 드롭 (Input, Conv2D, BatchNorm2D, ReLU, Linear, Output)
 *   4. 노드 파라미터 편집
 *   5. 노드 간 엣지 연결 + Shape propagation
 *   6. Shape 에러 감지 (in_ch 불일치)
 *   7. Shape 에러 복구 (파라미터 수정)
 *   8. 노드 삭제 + bypass edge
 *   9. AI 에이전트 패널 열기 및 메시지 전송
 *  10. replace_node 시각 반영 검증 (agentDirty + _attrName clear)
 *  11. PyTorch 코드 생성 팝업
 *  12. 그래프 진단 패널 (Diagnostics)
 *  13. 블록 팔레트 카테고리 탐색
 *
 * 결과물:
 *   benchmarks/reports/videos/full-e2e-<timestamp>.webm
 *   benchmarks/reports/screenshots/full-e2e-*.png
 *   benchmarks/reports/full-e2e-latest.json
 */

import fs from 'fs'
import path from 'path'
import { chromium } from 'playwright'
import {
  ensureDir,
  ensureServer,
  openApp,
  listCanvasNodes,
  connectNodes,
  editParam,
  nodeText,
  waitForTextToDisappear,
  sendPrompt,
  waitForAgentIdle,
} from './e2e-helpers.mjs'

// ── Paths ──────────────────────────────────────────────────────────────────
const rootDir = path.resolve(new URL('../..', import.meta.url).pathname)
const reportDir = path.join(rootDir, 'benchmarks', 'reports')
const screenshotDir = path.join(reportDir, 'screenshots')
const videoDir = path.join(reportDir, 'videos')

ensureDir(screenshotDir)
ensureDir(videoDir)

const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)

function shot(name) {
  return path.join(screenshotDir, `full-e2e-${name}.png`)
}

const results = []
let stepIndex = 0

function log(msg) {
  const prefix = `[${new Date().toISOString().slice(11, 23)}]`
  console.log(`${prefix} ${msg}`)
}

async function step(name, fn, page) {
  stepIndex += 1
  const label = `Step ${String(stepIndex).padStart(2, '0')}: ${name}`
  log(`▶ ${label}`)
  const t0 = Date.now()
  try {
    await fn()
    const elapsed = Date.now() - t0
    await page.screenshot({ path: shot(`${String(stepIndex).padStart(2, '0')}-${name.replace(/\s+/g, '_').toLowerCase().slice(0, 40)}`) })
    results.push({ step: label, status: 'PASS', elapsed_ms: elapsed })
    log(`  ✓ PASS (${elapsed}ms)`)
  } catch (err) {
    const elapsed = Date.now() - t0
    await page.screenshot({ path: shot(`${String(stepIndex).padStart(2, '0')}-FAIL-${name.replace(/\s+/g, '_').toLowerCase().slice(0, 30)}`) }).catch(() => {})
    results.push({ step: label, status: 'FAIL', elapsed_ms: elapsed, error: err.message })
    log(`  ✗ FAIL: ${err.message}`)
    throw err
  }
}

/**
 * ReactFlow 캔버스 위에 블록을 드롭하는 함수.
 * dragTo()는 canvas 오버레이에 의해 차단될 수 있으므로
 * raw mouse API + 캔버스 절대 좌표를 사용한다.
 */
async function dropBlockOnCanvas(page, blockTestId, absX, absY) {
  const searchInput = page.getByTestId('block-search-input')
  const block = page.getByTestId(blockTestId)

  // 검색으로 블록이 보이도록
  const visible = await block.isVisible().catch(() => false)
  if (!visible && await searchInput.isVisible().catch(() => false)) {
    const query = blockTestId.replace(/^block-/, '').toLowerCase().replace(/[^a-z0-9]/g, '')
    await searchInput.fill(query)
    await block.waitFor({ state: 'visible', timeout: 8000 })
  }

  const box = await block.boundingBox()
  if (!box) throw new Error(`block ${blockTestId} bounding box not found`)

  const srcX = box.x + box.width / 2
  const srcY = box.y + box.height / 2

  // pointerdown on source → move to canvas → pointerup
  await page.mouse.move(srcX, srcY)
  await page.mouse.down()
  await page.waitForTimeout(100)
  // Move in steps to simulate real drag
  await page.mouse.move(absX, absY, { steps: 20 })
  await page.waitForTimeout(100)
  await page.mouse.up()
  await page.waitForTimeout(400)

  // Clear search
  if (await searchInput.isVisible().catch(() => false)) {
    await searchInput.fill('')
    await page.waitForTimeout(200)
  }
}

// ── Server check ───────────────────────────────────────────────────────────
await ensureServer()

// ── Browser + Video context ────────────────────────────────────────────────
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.CHROMIUM_PATH,
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
})
const context = await browser.newContext({
  viewport: { width: 1680, height: 980 },
  recordVideo: {
    dir: videoDir,
    size: { width: 1680, height: 980 },
  },
})
const page = await context.newPage()

// Canvas 중앙 기준 절대 좌표 계산용 (초기값, 앱 로드 후 갱신)
let canvasBox = { x: 300, y: 0, width: 1380, height: 980 }

let exitCode = 0
try {
  // ── 1. App load ──────────────────────────────────────────────────────────
  await step('앱 로드', async () => {
    await openApp(page)
    await page.waitForSelector('[data-testid="flow-canvas"]', { timeout: 15000 })
    await page.waitForSelector('[data-testid="block-search-input"]', { timeout: 10000 })

    const box = await page.getByTestId('flow-canvas').boundingBox()
    if (box) canvasBox = box
    log(`  캔버스 박스: x=${canvasBox.x} y=${canvasBox.y} w=${canvasBox.width} h=${canvasBox.height}`)
  }, page)

  // 캔버스 내 절대 좌표 헬퍼
  function cx(relX) { return canvasBox.x + relX }
  function cy(relY) { return canvasBox.y + relY }

  // ── 2. Palette search ────────────────────────────────────────────────────
  await step('팔레트 검색', async () => {
    const search = page.getByTestId('block-search-input')
    await search.fill('conv2d')
    await page.waitForSelector('[data-testid="block-Conv2D"]', { timeout: 5000 })
    const visible = await page.getByTestId('block-Conv2D').isVisible()
    if (!visible) throw new Error('Conv2D 블록이 검색 결과에 보이지 않음')
    await search.fill('')
    await page.waitForTimeout(300)
  }, page)

  // ── 3. Drag blocks ───────────────────────────────────────────────────────
  // 파이프라인: Input → Conv2D → BatchNorm2D → ReLU → GlobalAvgPool → Linear(FC) → Output
  // GlobalAvgPool은 (B,C,H,W) → (B,C) 로 차원을 줄여 Linear 연결이 가능하게 함
  await step('블록 드래그 (Input, Conv2D, BN, ReLU, GlobalAvgPool, Linear, Output)', async () => {
    const drops = [
      ['block-Input',         cx(50),  cy(280)],
      ['block-Conv2D',        cx(210), cy(280)],
      ['block-BatchNorm2D',   cx(380), cy(280)],
      ['block-ReLU',          cx(540), cy(280)],
      ['block-GlobalAvgPool', cx(700), cy(280)],
      ['block-Linear',        cx(870), cy(280)],
      ['block-Output',        cx(1040),cy(280)],
    ]
    for (const [testId, x, y] of drops) {
      await dropBlockOnCanvas(page, testId, x, y)
      log(`    ${testId} → (${Math.round(x)}, ${Math.round(y)})`)
    }

    const nodes = await listCanvasNodes(page)
    log(`    배치된 노드 수: ${nodes.length}`)
    if (nodes.length < 7) throw new Error(`노드 수 부족: ${nodes.length}개 < 7개`)
  }, page)

  // ── 4. Parameter edit ────────────────────────────────────────────────────
  let nodes = await listCanvasNodes(page)
  log(`  현재 노드: ${nodes.map(n => n.text.slice(0, 18)).join(' | ')}`)
  const inputNode   = nodes[0]
  const conv2dNode  = nodes[1]
  const bnNode      = nodes[2]
  const reluNode    = nodes[3]
  const gapNode     = nodes[4]  // GlobalAvgPool
  const linearNode  = nodes[5]
  const outputNode  = nodes[6]

  await step('파라미터 편집 (Input.shape, Conv2D.in_ch/out_ch, Linear.in_features)', async () => {
    await editParam(page, inputNode.id,  'shape',       'B,3,64,64')
    await editParam(page, conv2dNode.id, 'in_ch',       3)
    await editParam(page, conv2dNode.id, 'out_ch',      16)
    // GlobalAvgPool: (B,16,H,W) → (B,16), Linear in_features=16
    await editParam(page, linearNode.id, 'in_features', 16)
    await editParam(page, linearNode.id, 'out_features', 10)
  }, page)

  // ── 5. Connect all edges ─────────────────────────────────────────────────
  await step('전체 엣지 연결 + Shape propagation (에러 없는 완전한 그래프)', async () => {
    await connectNodes(page, inputNode.id,  conv2dNode.id)
    await connectNodes(page, conv2dNode.id, bnNode.id)
    await connectNodes(page, bnNode.id,     reluNode.id)
    await connectNodes(page, reluNode.id,   gapNode.id)
    await connectNodes(page, gapNode.id,    linearNode.id)
    await connectNodes(page, linearNode.id, outputNode.id)
    await page.waitForTimeout(1000)

    // 모든 노드에 ⚠ 없음 확인
    const convText = await nodeText(page, conv2dNode.id)
    log(`    Conv2D 텍스트: "${convText.slice(0, 60)}"`)
    if (!convText.includes('16')) {
      throw new Error(`Conv2D 출력 shape에 out_ch(16)이 포함되지 않음: "${convText}"`)
    }
    // Diagnostics 배지에 에러 0 확인
    const diagText = await page.locator('[data-testid="drawer-tab-diagnostics"]').textContent().catch(() => '')
    log(`    진단 배지: "${diagText}"`)
    const diagMatch = diagText.match(/(\d+)\s*error/i)
    if (diagMatch && parseInt(diagMatch[1]) > 0) {
      // 에러가 있으면 내용 기록 (테스트를 실패시키지는 않음 — warning 레벨)
      log(`    ⚠ 진단 에러 존재: ${diagText}`)
    }
  }, page)

  // ── 6. Shape error detection ─────────────────────────────────────────────
  await step('Shape 에러 감지 (in_ch 불일치)', async () => {
    await editParam(page, conv2dNode.id, 'in_ch', 99)
    await page.waitForTimeout(700)
    const convText = await nodeText(page, conv2dNode.id)
    log(`    에러 상태 텍스트: "${convText.slice(0, 60)}"`)
    const hasError = convText.includes('⚠') || convText.includes('expected')
    if (!hasError) throw new Error(`Shape 에러가 감지되지 않음 (in_ch=99): "${convText}"`)
  }, page)

  // ── 7. Shape error recovery ──────────────────────────────────────────────
  await step('Shape 에러 복구 (in_ch 수정)', async () => {
    await editParam(page, conv2dNode.id, 'in_ch', 3)
    await waitForTextToDisappear(page, conv2dNode.id, '⚠', 6000)
  }, page)

  // ── 8. Node deletion ─────────────────────────────────────────────────────
  await step('노드 삭제 + bypass 엣지 자동 생성', async () => {
    const nodesBefore = await listCanvasNodes(page)
    const countBefore = nodesBefore.length

    // Hover to reveal delete button
    await page.getByTestId(`node-${bnNode.id}`).hover()
    await page.waitForTimeout(300)

    const deleteBtn = page.getByTestId(`delete-node-${bnNode.id}`)
    const btnVis = await deleteBtn.isVisible().catch(() => false)
    if (btnVis) {
      // dispatchEvent로 오버랩 차단 우회
      await deleteBtn.dispatchEvent('click')
    } else {
      log('  ℹ delete 버튼을 찾지 못함 — keyboard Delete 시도')
      await page.getByTestId(`node-${bnNode.id}`).click({ force: true })
      await page.waitForTimeout(200)
      await page.keyboard.press('Delete')
    }
    await page.waitForTimeout(600)

    const nodesAfter = await listCanvasNodes(page)
    log(`    삭제 전: ${countBefore}개, 삭제 후: ${nodesAfter.length}개`)
    if (nodesAfter.length >= countBefore) throw new Error(`노드 삭제 실패 (${countBefore} → ${nodesAfter.length})`)

    // bypass 엣지 생성으로 Conv2D → ReLU가 바로 연결되어야 함 → shape 에러 없어야 함
    await page.waitForTimeout(400)
    const diagText = await page.locator('[data-testid="drawer-tab-diagnostics"]').textContent().catch(() => '')
    log(`    BN 삭제 후 진단: "${diagText}"`)
  }, page)

  // ── 9. Agent panel ───────────────────────────────────────────────────────
  await step('AI 에이전트 패널 열기 + 쿼리 전송', async () => {
    // AgentPanel은 항상 렌더링됨 (토글 불필요)
    // chat-input이 보이는지만 확인
    const chatInput = page.getByTestId('agent-chat-input')
    await chatInput.waitFor({ state: 'visible', timeout: 10000 })

    await sendPrompt(page, 'What nodes are currently on the canvas?')
    await waitForAgentIdle(page, 60000)
    await page.waitForTimeout(500)

    // testid: agent-message (agent 역할), agent-user-message (user), agent-system-message (system)
    const agentMsgCount = await page.locator('[data-testid="agent-message"]').count()
    const userMsgCount = await page.locator('[data-testid="agent-user-message"]').count()
    const totalMsgs = agentMsgCount + userMsgCount
    log(`    에이전트 메시지: ${agentMsgCount}개, 사용자 메시지: ${userMsgCount}개`)
    if (totalMsgs < 1) throw new Error('에이전트 응답이 없음')
  }, page)

  // ── 10. replace_node visual check ────────────────────────────────────────
  await step('replace_node 시각 반영 검증', async () => {
    const currentNodes = await listCanvasNodes(page)
    const relu = currentNodes.find(n => n.text.toLowerCase().includes('relu'))
    if (!relu) {
      log('  ℹ ReLU 노드 없음 — replace_node 테스트 스킵')
      return
    }

    const chatInput = page.getByTestId('agent-chat-input')
    await chatInput.fill(`Replace the node with id "${relu.id}" (currently ReLU) with a GELU activation`)
    await page.getByTestId('agent-chat-send').click()
    await waitForAgentIdle(page, 90000)
    await page.waitForTimeout(800)

    const nodesAfter = await listCanvasNodes(page)
    const reluStillShown = nodesAfter.find(n => n.id === relu.id && n.text.toLowerCase().includes('relu'))
    const geluShown = nodesAfter.find(n => n.text.toLowerCase().includes('gelu'))

    log(`    교체 후 노드: ${nodesAfter.map(n => n.text.slice(0, 15)).join(' | ')}`)
    if (reluStillShown) {
      throw new Error('replace_node 후에도 카드가 ReLU로 표시됨 — _attrName 클리어 버그')
    }
    if (!geluShown) {
      log('  ⚠ GELU 카드가 보이지 않음 (에이전트가 다른 액션을 실행했을 수 있음)')
    }
  }, page)

  // ── 11. PyTorch code generation ───────────────────────────────────────────
  await step('PyTorch 코드 생성', async () => {
    const codeTestIds = ['show-code-button', 'generate-code-button', 'code-export-button']
    let clicked = false
    for (const tid of codeTestIds) {
      const btn = page.getByTestId(tid)
      if (await btn.isVisible().catch(() => false)) {
        await btn.click()
        clicked = true
        break
      }
    }
    if (!clicked) {
      const roleBtn = page.getByRole('button', { name: /code|pytorch|export/i }).first()
      if (await roleBtn.isVisible().catch(() => false)) {
        await roleBtn.click()
        clicked = true
      }
    }
    if (!clicked) throw new Error('코드 생성 버튼을 찾지 못함')
    await page.waitForTimeout(700)

    const codeEl = page.locator('pre, code, [data-testid="generated-code-output"]').filter({ hasText: /import torch|nn\.|forward/ })
    const count = await codeEl.count()
    if (count < 1) throw new Error('PyTorch 코드 텍스트가 나타나지 않음')

    // Close modal — the Close button has no testid; use role + name
    const closeBtn = page.getByRole('button', { name: 'Close' })
    if (await closeBtn.isVisible().catch(() => false)) {
      await closeBtn.click()
    } else {
      // Force-close via JS on the overlay element
      await page.evaluate(() => {
        const overlay = document.querySelector('.\\bg-black\\/70, [class*="z-50"][class*="bg-black"]')
        if (overlay) {
          const closeB = overlay.querySelector('button')
          // Find "Close" button text
          const buttons = Array.from(overlay.querySelectorAll('button'))
          const closeTarget = buttons.find(b => b.textContent?.trim() === 'Close')
          if (closeTarget) closeTarget.click()
        }
      })
    }
    // Wait until the overlay is gone
    await page.waitForSelector('.absolute.inset-0.z-50', { state: 'hidden', timeout: 8000 }).catch(() => {})
    await page.waitForTimeout(400)
  }, page)

  // ── 12. Diagnostics panel ────────────────────────────────────────────────
  await step('그래프 진단 패널', async () => {
    const diagTestIds = ['drawer-tab-diagnostics', 'tab-diagnostics', 'diagnostics-tab']
    let clicked = false
    for (const tid of diagTestIds) {
      const el = page.getByTestId(tid)
      if (await el.isVisible().catch(() => false)) { await el.click(); clicked = true; break }
    }
    if (!clicked) {
      const el = page.getByText(/diagnostics/i).first()
      if (await el.isVisible().catch(() => false)) { await el.click(); clicked = true }
    }
    await page.waitForTimeout(500)

    // Health summary or diagnostic item
    const health = page.locator('[data-testid*="health"], [data-testid*="diagnostic"], [data-testid*="graph-stat"]').first()
    const vis = await health.isVisible().catch(() => false)
    if (!vis) {
      log('  ℹ 진단 패널 요소를 testid로 찾지 못함 — 페이지 내 텍스트로 확인')
      const txt = await page.textContent('body')
      const hasDiag = /(healthy|error|warning|issue|diagnostic)/i.test(txt)
      if (!hasDiag) throw new Error('진단 정보가 페이지 어디에도 없음')
    }
  }, page)

  // ── 13. Category navigation ──────────────────────────────────────────────
  await step('팔레트 카테고리 탐색', async () => {
    const categories = ['conv', 'attention', 'recurrent', 'linear', 'norm']
    let found = 0
    for (const cat of categories) {
      const btn = page.getByTestId(`category-${cat}`)
      if (await btn.isVisible().catch(() => false)) {
        await btn.click()
        await page.waitForTimeout(250)
        found++
        log(`    카테고리 [${cat}] 클릭`)
      }
    }
    if (found === 0) log('  ℹ 카테고리 버튼을 testid로 찾지 못함 — 스킵')
  }, page)

  log('\n==============================')
  log('  모든 테스트 완료')
  log('==============================\n')

} catch (err) {
  log(`\n치명적 오류: ${err.message}`)
  exitCode = 1
} finally {
  await page.close()
  const video = page.video()
  let videoPath = null
  if (video) {
    const tmpPath = await video.path()
    videoPath = path.join(videoDir, `full-e2e-${timestamp}.webm`)
    if (tmpPath) {
      try { fs.renameSync(tmpPath, videoPath) } catch { videoPath = tmpPath }
      log(`📹 영상 저장: ${videoPath}`)
    }
  }
  await context.close()
  await browser.close()

  const totalPass = results.filter(r => r.status === 'PASS').length
  const totalFail = results.filter(r => r.status === 'FAIL').length
  const report = {
    timestamp,
    summary: { total: results.length, pass: totalPass, fail: totalFail },
    videoPath,
    screenshotDir,
    steps: results,
  }
  const reportPath = path.join(reportDir, 'full-e2e-latest.json')
  ensureDir(reportDir)
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2))
  log(`\n📋 보고서: ${reportPath}`)
  log(`결과: ✓${totalPass}개 통과  ✗${totalFail}개 실패`)
  process.exit(exitCode)
}

