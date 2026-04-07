import fs from 'fs'

function normalizeBlockQuery(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '')
}

export async function ensureServer() {
  const res = await fetch('http://127.0.0.1:5173')
  if (!res.ok) {
    throw new Error(`server returned ${res.status}`)
  }
}

export async function openApp(page) {
  await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' })
}

export async function dragBlockOntoCanvas(page, blockTestId, canvasLocator, xOffset, yOffset) {
  const block = page.getByTestId(blockTestId)
  const searchInput = page.getByTestId('block-search-input')
  if (!(await block.isVisible().catch(() => false)) && await searchInput.isVisible().catch(() => false)) {
    const fallbackQuery = normalizeBlockQuery(blockTestId.replace(/^block-/, ''))
    await searchInput.fill(fallbackQuery)
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
  await page.waitForTimeout(250)
}

export async function listCanvasNodes(page) {
  return await page.locator('.react-flow__node').evaluateAll((elements) => (
    elements.map((element) => ({
      id: element.getAttribute('data-id') || '',
      text: (element.textContent || '').trim(),
      agentEdited: element.querySelector('[data-agent-edited="true"]') !== null,
    }))
  ))
}

export async function connectNodes(page, sourceId, targetId) {
  const sourceButton = page.getByTestId(`connect-node-${sourceId}`)
  await sourceButton.evaluate((element) => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
  })
  const targetNode = page.getByTestId(`node-${targetId}`)
  await targetNode.evaluate((element) => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
  })
  await page.waitForTimeout(350)
}

export async function dragConnectHandles(page, sourceId, targetId) {
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
  await page.waitForTimeout(450)
}

export async function editParam(page, nodeId, key, value) {
  const button = page.getByTestId(`param-value-${nodeId}-${key}`)
  await button.evaluate((element) => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
  })
  const input = page.getByTestId(`param-input-${nodeId}-${key}`)
  await input.waitFor({ state: 'visible', timeout: 10000 })
  await input.fill(String(value))
  await input.press('Enter')
  await page.waitForTimeout(250)
}

export async function nodeText(page, nodeId) {
  return (await page.getByTestId(`node-${nodeId}`).textContent()) || ''
}

export async function waitForTextToDisappear(page, nodeId, fragment, timeout = 5000) {
  await page.waitForFunction(
    ({ id, token }) => {
      const node = document.querySelector(`[data-testid="node-${id}"]`)
      return node && !(node.textContent || '').includes(token)
    },
    { id: nodeId, token: fragment },
    { timeout },
  )
}

export async function sendPrompt(page, prompt) {
  await page.getByTestId('agent-chat-input').fill(prompt)
  await page.getByTestId('agent-chat-send').click()
}

export async function waitForAgentIdle(page, timeout = 240000) {
  await page.waitForFunction(() => {
    const button = document.querySelector('[data-testid="agent-chat-send"]')
    return button && button.textContent?.trim() === '↑'
  }, undefined, { timeout })
}

export function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true })
}
