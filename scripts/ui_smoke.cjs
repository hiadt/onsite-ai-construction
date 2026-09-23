// Local visual smoke test. Run with NODE_PATH set to the bundled node_modules.
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const failures = [];
  page.on('pageerror', error => failures.push(error.message));
  const output = 'D:/Codex_output/PathGuard产品体验重构';
  fs.mkdirSync(output, { recursive: true });
  await page.goto('http://127.0.0.1:8522/', { waitUntil: 'networkidle' });
  await page.getByText('欢迎来到 PATHGUARD', { exact: true }).waitFor({ timeout: 60000 });
  await page.screenshot({ path: path.join(output, '01-首页.png'), fullPage: true });
  await page.getByText('产品介绍', { exact: true }).first().click();
  await page.getByText('从路线到行动', { exact: false }).first().waitFor();
  await page.screenshot({ path: path.join(output, '02-产品介绍.png'), fullPage: true });
  await page.mouse.wheel(0, 760);
  await page.waitForTimeout(350);
  await page.screenshot({ path: path.join(output, '02b-产品介绍内容.png') });
  await page.getByText('风险工作台', { exact: true }).first().click();
  await page.getByText('从哪里开始？', { exact: true }).first().waitFor();
  await page.screenshot({ path: path.join(output, '03-工作台.png'), fullPage: true });
  await page.getByText('历史路线', { exact: true }).first().click();
  await page.waitForTimeout(2200);
  await page.mouse.wheel(0, 1000);
  await page.waitForTimeout(900);
  await page.screenshot({ path: path.join(output, '04-历史路线.png') });
  await page.getByText('路线详情', { exact: true }).first().click();
  await page.waitForTimeout(2200);
  await page.mouse.wheel(0, 680);
  await page.waitForTimeout(900);
  await page.screenshot({ path: path.join(output, '05-路线详情.png') });
  await page.getByText('讲解示例（15）', { exact: true }).first().click();
  await page.waitForTimeout(1400);
  await page.getByText('路线详情', { exact: true }).first().click();
  const routeSelect = page.locator('[data-testid="stSelectbox"]').filter({ has: page.getByText('选择候选路线', { exact: true }) });
  await routeSelect.locator('[role="combobox"]').fill('历史路线 052');
  await page.getByText('历史路线 052', { exact: false }).last().click();
  await page.waitForTimeout(1600);
  await page.mouse.wheel(0, 640);
  await page.screenshot({ path: path.join(output, '06-可信边界示例.png') });
  await page.mouse.wheel(0, 780);
  await page.waitForTimeout(350);
  await page.screenshot({ path: path.join(output, '06b-全程与局部.png') });
  await page.getByText('专业指标、几何设置与原始追溯信息', { exact: true }).click();
  const widthInput = page.getByRole('spinbutton', { name: '车宽 / m' });
  await widthInput.fill('7.0');
  await widthInput.press('Tab');
  await page.getByText('试算假设尺寸下的边界余量（仅供演示）', { exact: true }).click();
  await page.waitForTimeout(1800);
  const scenarioText = await page.locator('body').innerText();
  if (!scenarioText.includes('假设尺寸试算')) failures.push('Scenario result did not identify assumed dimensions');
  await page.getByText('假设尺寸试算，仅展示尺寸敏感性', { exact: false }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, '06c-假设尺寸试算.png') });
  await page.getByText('专业依据', { exact: true }).first().click();
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(output, '09-专业依据.png') });
  const labels = await page.locator('body').innerText();
  console.log(JSON.stringify({ failures, hasHash: labels.includes('sample_'), hasRoute: labels.includes('历史路线'),
    visibleModebars: await page.locator('.modebar:visible').count(), svgFrames: await page.locator('iframe').count() }));
  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
  mobile.on('pageerror', error => failures.push(error.message));
  await mobile.goto('http://127.0.0.1:8522/', { waitUntil: 'networkidle' });
  await mobile.getByText('欢迎来到 PATHGUARD', { exact: true }).waitFor({ timeout: 60000 });
  await mobile.screenshot({ path: path.join(output, '07-手机首页.png') });
  await mobile.getByText('产品介绍', { exact: true }).first().click();
  await mobile.getByText('从路线到行动', { exact: false }).first().waitFor();
  await mobile.screenshot({ path: path.join(output, '08-手机产品介绍.png') });
  console.log(JSON.stringify({ mobileHorizontalOverflow: await mobile.evaluate(() => document.documentElement.scrollWidth > innerWidth), failures }));
  await mobile.close();
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
