import fs from 'fs'
import path from 'path'
import { chromium } from 'playwright'

function normalizeBlockQuery(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '')
}

const rootDir = path.resolve(new URL('../..', import.meta.url).pathname)
const screenshotPath = path.join(rootDir, 'benchmarks', 'reports', 'screenshots', 'manual-edit-flow-check.png')
const videoDir = path.join(rootDir, 'benchmarks', 'reports', 'videos')
const videoPath = path.join(videoDir, 'manual-edit-flow-check.webm')

async function ensureServer() {
  const res = await fetch('http://127.0.0.1:5173')
  if (!res.ok) {
    throw new Error(`server returned ${res.status}`)
  }
}

async function dragBlockOntoCanvas(page, blockTestId, canvasLocator, xOffset, yOffset) {
  const block = page.getByTestId(blockTestId)
  const searchInput = page.getByTestId('block-search-input')
  if (!(await block.isVisible().catch(() => false)) && await searchInput.isVisible().catch(() => false)) {
    await searchInput.fill(normalizeBlockQuery(blockTestId.replace(/^block-/, '')))
    await block.waitFor({ state: 'visible', timeout: 10000 })
  }
  await block.dragTo(canvasLocator, {
    targetPosition: {
      x: xOffset,
      y: yOffset,
    },
  })
  if (await searchInput.isVisible().catch(() => false)) {
    await searchInput.fill('')
  }
  await page.waitForTimeout(300)
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
  await page.getByTestId(`connect-node-${sourceId}`).evaluate((element) => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
  })
  await page.getByTestId(`node-${targetId}`).evaluate((element) => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
  })
  await page.waitForTimeout(400)
}

async function dragConnectHandles(page, sourceId, targetId) {
  const source = page.getByTestId(`source-handle-${sourceId}`)
  const target = page.getByTestId(`target-handle-${targetId}`)
  const sourceBox = await source.boundingBox()
  const targetBox = await target.boundingBox()
  if (!sourceBox || !targetBox) {
    throw new Error(`missing handle bounds for ${sourceId} -> ${targetId}`)
  }
  await page.mouse.move(sourceBox.x + sourceBox.width / 2, sourceBox.y + sourceBox.height / 2)
  await page.mouse.down()
  await page.mouse.move(targetBox.x + targetBox.width / 2, targetBox.y + targetBox.height / 2, { steps: 18 })
  await page.mouse.up()
  await page.waitForTimeout(500)
}

async function editParam(page, nodeId, key, value) {
  await page.getByTestId(`param-value-${nodeId}-${key}`).evaluate((element) => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
  })
  const input = page.getByTestId(`param-input-${nodeId}-${key}`)
  await input.waitFor({ state: 'visible', timeout: 10000 })
  await input.fill(String(value))
  await input.press('Enter')
  await page.waitForTimeout(250)
}

async function nodeText(page, nodeId) {
  return (await page.getByTestId(`node-${nodeId}`).textContent()) || ''
}

async function waitForNodeToDropWarning(page, nodeId, fragment = 'no input') {
  await page.waitForFunction(
    ({ id, token }) => {
      const node = document.querySelector(`[data-testid="node-${id}"]`)
      return node && !(node.textContent || '').includes(token)
    },
    { id: nodeId, token: fragment },
    { timeout: 5000 },
  )
}

await ensureServer()
fs.mkdirSync(path.dirname(screenshotPath), { recursive: true })
fs.mkdirSync(videoDir, { recursive: true })

const browser = await chromium.launch({ headless: true })
const context = await browser.newContext({
  viewport: { width: 1600, height: 950 },
  recordVideo: {
    dir: videoDir,
    size: { width: 1600, height: 950 },
  },
})
const page = await context.newPage()
const video = page.video()

try {
  await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' })
  const canvas = page.getByTestId('flow-canvas')
  const searchInput = page.getByTestId('block-search-input')
  const categoryToggleIds = await page.locator('[data-testid^="category-toggle-"]').evaluateAll((elements) => (
    elements.map((element) => element.getAttribute('data-testid') || '')
  ))
  const uniqueCategoryToggleIds = new Set(categoryToggleIds)
  if (uniqueCategoryToggleIds.size !== categoryToggleIds.length) {
    throw new Error(`expected unique category toggles, found duplicates: ${categoryToggleIds.length - uniqueCategoryToggleIds.size}`)
  }

  await searchInput.fill('upsample')
  const dynamicUpsampleVisible = await page.getByTestId('block-Upsample').isVisible()
  if (!dynamicUpsampleVisible) {
    throw new Error('expected dynamic torch.nn Upsample block to appear in palette search')
  }
  await searchInput.fill('')

  await dragBlockOntoCanvas(page, 'block-Input', canvas, 140, 210)
  await dragBlockOntoCanvas(page, 'block-Conv2D', canvas, 360, 210)
  await dragBlockOntoCanvas(page, 'block-ReLU', canvas, 600, 210)
  await dragBlockOntoCanvas(page, 'block-Conv2D', canvas, 840, 210)

  const initialNodes = await listCanvasNodes(page)
  if (initialNodes.length < 4) {
    throw new Error(`expected 4 nodes, found ${initialNodes.length}`)
  }

  const [inputNode, convNodeA, reluNode, convNodeB] = initialNodes
  await editParam(page, inputNode.id, 'shape', 'B,1,28,28')
  await editParam(page, convNodeA.id, 'in_ch', 1)
  await editParam(page, convNodeA.id, 'out_ch', 32)
  await editParam(page, convNodeA.id, 'kernel', '5,5')
  await editParam(page, convNodeB.id, 'in_ch', 32)
  await editParam(page, convNodeB.id, 'out_ch', 64)
  await editParam(page, convNodeB.id, 'kernel', '5,5')

  await connectNodes(page, inputNode.id, convNodeA.id)
  const edgeCountAfterFirstConnect = await page.locator('.react-flow__edge').count()
  await connectNodes(page, convNodeA.id, reluNode.id)
  await waitForNodeToDropWarning(page, reluNode.id)
  const edgeCountAfterSecondConnect = await page.locator('.react-flow__edge').count()
  const reluNodeAfterHandleConnect = await nodeText(page, reluNode.id)
  if (reluNodeAfterHandleConnect.includes('no input')) {
    throw new Error('GELU/ReLU node still reports no input after reconnecting its upstream edge')
  }
  await connectNodes(page, reluNode.id, convNodeB.id)
  await waitForNodeToDropWarning(page, convNodeB.id)
  const edgeCountAfterThirdConnect = await page.locator('.react-flow__edge').count()
  const convNodeBAfterHandleConnect = await nodeText(page, convNodeB.id)
  if (convNodeBAfterHandleConnect.includes('no input')) {
    throw new Error('Downstream Conv2D still reports no input after reconnecting the chain')
  }

  const initialEdgeCount = edgeCountAfterThirdConnect
  await page.getByTestId(`delete-node-${reluNode.id}`).click()
  await page.waitForTimeout(500)

  const afterDeleteNodes = await listCanvasNodes(page)
  const edgeCountAfterDelete = await page.locator('.react-flow__edge').count()
  const reluDeleted = !afterDeleteNodes.some((node) => node.id === reluNode.id)
  if (!reluDeleted) {
    throw new Error('ReLU node remained after clicking delete x')
  }
  if (edgeCountAfterDelete !== 2) {
    throw new Error(`expected delete to preserve a linear bypass edge, found ${edgeCountAfterDelete} edges`)
  }

  await dragBlockOntoCanvas(page, 'block-LeakyReLU', canvas, 600, 210)
  const afterLeakyNodes = await listCanvasNodes(page)
  const leakyNode = afterLeakyNodes.find((node) => !afterDeleteNodes.some((candidate) => candidate.id === node.id))
  if (!leakyNode) {
    throw new Error('failed to identify inserted LeakyReLU node')
  }

  await connectNodes(page, convNodeA.id, leakyNode.id)
  await editParam(page, leakyNode.id, 'alpha', 0.2)
  await waitForNodeToDropWarning(page, leakyNode.id)
  const leakyNodeAfterInputConnect = await nodeText(page, leakyNode.id)
  if (leakyNodeAfterInputConnect.includes('no input')) {
    throw new Error('LeakyReLU still reports no input after reconnecting its upstream edge')
  }
  await connectNodes(page, leakyNode.id, convNodeB.id)
  await waitForNodeToDropWarning(page, convNodeB.id)
  const edgeCountAfterReconnect = await page.locator('.react-flow__edge').count()
  const finalNodeTexts = await Promise.all(
    (await listCanvasNodes(page)).map((node) => nodeText(page, node.id)),
  )
  const finalNoInputWarnings = finalNodeTexts.filter((text) => text.includes('no input')).length
  if (finalNoInputWarnings !== 0) {
    throw new Error(`expected zero final no-input warnings, found ${finalNoInputWarnings}`)
  }

  await page.getByTestId('generate-code-button').click()
  await page.getByText('Generated PyTorch Code', { exact: true }).waitFor({ timeout: 10000 })
  const generatedCode = await page.getByTestId('generated-code-output').textContent()
  await page.getByRole('button', { name: /Close/i }).click()
  await page.screenshot({ path: screenshotPath })

  console.log(JSON.stringify({
    initialNodeLabels: initialNodes.map((node) => node.text.split('\n')[0]),
    edgeCountAfterFirstConnect,
    edgeCountAfterSecondConnect,
    edgeCountAfterThirdConnect,
    reluHasInputAfterHandleConnect: !reluNodeAfterHandleConnect.includes('no input'),
    convHasInputAfterHandleConnect: !convNodeBAfterHandleConnect.includes('no input'),
    initialEdgeCount,
    deletedNodeId: reluNode.id,
    reluDeleted,
    edgeCountAfterDelete,
    replacementNodeId: leakyNode.id,
    leakyNodeHasInputAfterFirstReconnect: !leakyNodeAfterInputConnect.includes('no input'),
    edgeCountAfterReconnect,
    finalNoInputWarnings,
    generatedCodeContainsEditedKernel: Boolean(generatedCode?.includes('kernel_size=(5,5)')),
    generatedCodeContainsEditedAlpha: Boolean(generatedCode?.includes('LeakyReLU(0.2)')),
    generatedCodeContainsLeakyReLU: Boolean(generatedCode?.includes('LeakyReLU')),
    uniqueCategoryToggleCount: uniqueCategoryToggleIds.size,
    dynamicCatalogHasUpsample: dynamicUpsampleVisible,
    screenshot: screenshotPath,
    video: videoPath,
  }, null, 2))
} finally {
  await page.close()
  if (video) {
    const rawVideoPath = await video.path()
    if (rawVideoPath !== videoPath) {
      if (fs.existsSync(videoPath)) fs.rmSync(videoPath, { force: true })
      fs.renameSync(rawVideoPath, videoPath)
    }
  }
  await context.close()
  await browser.close()
}
