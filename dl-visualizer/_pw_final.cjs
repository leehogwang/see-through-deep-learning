const { chromium } = require('playwright');
const fs = require('fs');
const outDir = '/tmp/pw_shots2';
fs.mkdirSync(outDir, { recursive: true });

(async () => {
  const browser = await chromium.launch({
    executablePath: '/home/students/cs/202421012/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome',
    args: ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  
  // 콘솔 에러 수집
  const consoleErrors = [];
  page.on('console', msg => { if(msg.type()==='error') consoleErrors.push(msg.text()); });
  
  await page.goto('http://localhost:5174', { waitUntil: 'networkidle', timeout: 30000 });
  await page.screenshot({ path: `${outDir}/01_initial.png` });
  console.log('01_initial.png');

  // Simple CNN 클릭
  await page.click('button:has-text("Simple CNN")');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${outDir}/02_after_simple_cnn.png` });
  const nodeCount = await page.locator('.react-flow__node').count();
  console.log('nodes after Simple CNN:', nodeCount);
  
  // 채팅창 메시지 수집
  const messages = await page.locator('[class*="message"],[class*="chat"] div').evaluateAll(
    els => els.filter(e=>e.textContent?.length>5).map(e=>e.textContent?.trim().slice(0,60)).slice(0,5)
  );
  console.log('Chat messages:', messages);

  // Console errors
  console.log('Console errors:', consoleErrors.slice(0,5));

  await browser.close();
  console.log('DONE');
})().catch(e => { console.error('ERROR:', e.message); process.exit(1); });
