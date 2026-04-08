import fs from 'fs'
import path from 'path'
import { chromium } from 'playwright'
import { ensureDir, ensureServer, openApp, dragBlockOntoCanvas, listCanvasNodes } from './e2e-helpers.mjs'

const rootDir = path.resolve(new URL('../..', import.meta.url).pathname)
const reportDir = path.join(rootDir, 'benchmarks', 'reports')
const screenshotDir = path.join(reportDir, 'screenshots')
const reportPath = path.join(reportDir, 'palette-ui-insertions-latest.json')

ensureDir(reportDir)
ensureDir(screenshotDir)
await ensureServer()

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1680, height: 980 } })

async function collectAllBlockTestIds(page) {
  const toggleIds = await page.locator('[data-testid^="category-toggle-"]').evaluateAll((elements) => (
    elements
      .map((element) => element.getAttribute('data-testid') || '')
      .filter(Boolean)
  ))

  for (const toggleId of toggleIds) {
    const toggle = page.getByTestId(toggleId)
    const expanded = await toggle.textContent().catch(() => '')
    if (expanded?.includes('▼')) {
      await toggle.click()
      await page.waitForTimeout(100)
    }
  }

  return await page.locator('[data-testid^="block-"][draggable="true"]').evaluateAll((elements) => {
    const ids = elements
      .map((element) => element.getAttribute('data-testid') || '')
      .filter(Boolean)
    return [...new Set(ids)]
  })
}

try {
  await openApp(page)
  const blockTestIds = await collectAllBlockTestIds(page)
  console.log(`[palette-ui] collected ${blockTestIds.length} palette blocks`)

  const results = []

  for (const blockTestId of blockTestIds) {
    console.log(`[palette-ui] testing ${blockTestId}`)
    await openApp(page)
    const canvas = page.getByTestId('flow-canvas')
    const beforeCount = await page.locator('.react-flow__node').count()
    let inserted = false
    let nodeText = ''
    let healthText = ''
    let error = null

    try {
      await dragBlockOntoCanvas(page, blockTestId, canvas, 360, 220)
      await page.waitForTimeout(250)

      const afterNodes = await listCanvasNodes(page)
      const afterCount = afterNodes.length
      inserted = afterCount === beforeCount + 1
      const newestNode = afterNodes[afterNodes.length - 1]
      nodeText = newestNode?.text ?? ''
      healthText = (await page.getByTestId('graph-health-panel').textContent().catch(() => '')) || ''

      if (!inserted) {
        error = `expected node count ${beforeCount + 1}, got ${afterCount}`
      }
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause)
    }

    const screenshotPath = path.join(screenshotDir, `${blockTestId.replace(/[^a-z0-9_-]+/gi, '_')}.png`)
    await page.screenshot({ path: screenshotPath })

    results.push({
      blockTestId,
      inserted,
      nodeText,
      healthText,
      error,
      screenshot: screenshotPath,
    })

    console.log(`[palette-ui] ${blockTestId} -> ${error ? `FAIL: ${error}` : 'OK'}`)
  }

  const failures = results.filter((result) => !result.inserted || result.error)
  const report = {
    generatedAt: new Date().toISOString(),
    total: results.length,
    inserted: results.filter((result) => result.inserted).length,
    failed: failures.length,
    failures,
    results,
  }

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))

  if (failures.length > 0) {
    process.exitCode = 1
  }
} finally {
  await browser.close()
}