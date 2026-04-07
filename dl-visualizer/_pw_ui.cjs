const { chromium } = require('playwright');
const fs = require('fs');
const outDir = '/tmp/pw_shots';
fs.mkdirSync(outDir, { recursive: true });

(async () => {
  const browser = await chromium.launch({
    executablePath: '/home/students/cs/202421012/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome',
    args: ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  await page.goto('http://localhost:5174', { waitUntil: 'networkidle', timeout: 30000 });

  // 01 초기
  await page.screenshot({ path: `${outDir}/01_initial.png` });
  console.log('01_initial.png saved');

  // 02 Simple CNN 로드
  await page.click('button:has-text("Simple CNN")');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${outDir}/02_simplecnn.png` });
  const nodeCount = await page.locator('.react-flow__node').count();
  console.log(`02_simplecnn.png - nodes: ${nodeCount}`);

  // 03 노드 텍스트 목록
  const nodeTexts = await page.locator('.react-flow__node').evaluateAll(els =>
    els.map(e => e.querySelector('[class*="label"],[class*="title"],span,h3,h4')?.textContent?.trim().slice(0,20) || e.textContent?.trim().slice(0,20))
  );
  console.log('Node labels:', nodeTexts);

  // 04 첫 노드 클릭 (파라미터 편집 확인)
  if (nodeCount > 0) {
    await page.locator('.react-flow__node').first().click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${outDir}/03_node_selected.png` });

    // 파라미터 input 있는지
    const paramInputs = await page.locator('.react-flow__node input[type="number"], .react-flow__node input[type="text"]').count();
    console.log(`03_node_selected.png - param inputs: ${paramInputs}`);
  }

  // 05 _dataPreview 있는 노드 수 (이미지/차트 있는 노드)
  const previewCount = await page.locator('.react-flow__node img, .react-flow__node svg').count();
  console.log(`Nodes with preview (img/svg): ${previewCount}`);

  // 06 전체 스크린샷
  await page.screenshot({ path: `${outDir}/04_final.png` });
  console.log('04_final.png saved');

  await browser.close();
  console.log('ALL DONE');
})().catch(e => { console.error('ERROR:', e.message); process.exit(1); });
