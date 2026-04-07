const { chromium } = require('playwright');
const fs = require('fs');
const outDir = '/tmp/pw_shots';
fs.mkdirSync(outDir, { recursive: true });

const MODEL_PAYLOAD = {
  repoRoot: '/home/students/cs/202421012/food_weight_estimation/VIF2/recipe_adamine',
  sourceFile: '/home/students/cs/202421012/food_weight_estimation/VIF2/recipe_adamine/model_recipe_rag_RNS.py',
  modelName: 'FusionNutritionModelRecipeRAG',
};

(async () => {
  const browser = await chromium.launch({
    executablePath: '/home/students/cs/202421012/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome',
    args: ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  // 1. 앱 초기화면
  await page.goto('http://localhost:5174', { waitUntil: 'networkidle', timeout: 30000 });
  await page.screenshot({ path: `${outDir}/01_initial.png` });
  console.log('01_initial.png - 초기 빈 캔버스');

  // 2. API로 직접 모델 로드 트리거 (Open Project도 되지만 파일 선택 다이얼로그는 자동화 불가)
  //    대신 런타임 트레이스 API를 직접 호출해서 UI 상태 확인
  const traceResp = await page.evaluate(async (payload) => {
    const r = await fetch('http://localhost:5173/api/trace-layer-data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return r.json();
  }, MODEL_PAYLOAD);

  console.log('Trace result error:', traceResp.error);
  console.log('Preview keys:', Object.keys(traceResp.previews).slice(0, 8));
  console.log('Input preview keys:', Object.keys(traceResp.inputPreviews));

  // 3. "Simple CNN" 데모 버튼 클릭해서 노드가 나오는지 확인
  await page.click('button:has-text("Simple CNN")');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${outDir}/02_simple_cnn.png` });
  const nodeCount = await page.locator('.react-flow__node').count();
  console.log(`02_simple_cnn.png - 노드 수: ${nodeCount}`);

  // 4. 노드 하나 클릭 → 카드 파라미터 확인
  if (nodeCount > 0) {
    await page.locator('.react-flow__node').first().click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${outDir}/03_node_selected.png` });
    console.log('03_node_selected.png - 노드 선택 후');
  }

  // 5. 전체 캔버스 스크린샷
  await page.screenshot({ path: `${outDir}/04_full_canvas.png`, fullPage: true });
  console.log('04_full_canvas.png - 전체 페이지');

  await browser.close();
  console.log('Done. Screenshots in', outDir);
})().catch(e => { console.error('ERROR:', e.message); process.exit(1); });
