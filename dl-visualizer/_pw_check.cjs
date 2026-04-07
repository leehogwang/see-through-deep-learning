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

  console.log('Navigating to app...');
  await page.goto('http://localhost:5174', { waitUntil: 'networkidle', timeout: 30000 });
  await page.screenshot({ path: `${outDir}/01_initial.png` });
  console.log('01_initial.png saved');

  // 노드 있는지 확인
  const nodeCount = await page.locator('.react-flow__node').count();
  console.log('Node count on load:', nodeCount);

  // 모델 로딩이 필요하면 input 파악
  const inputs = await page.locator('input, button').evaluateAll(els =>
    els.map(e => ({ tag: e.tagName, text: e.textContent?.trim().slice(0,30), type: e.type, placeholder: e.placeholder }))
  );
  console.log('Inputs/buttons found:', JSON.stringify(inputs.slice(0, 10), null, 2));

  await browser.close();
  console.log('Done');
})().catch(e => { console.error(e.message); process.exit(1); });
