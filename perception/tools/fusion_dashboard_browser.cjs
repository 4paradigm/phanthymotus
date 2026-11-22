/* Run against a clean, local fusion Dashboard fixture. */
const { chromium } = require(process.argv[2]);
const fs = require('node:fs');
const path = require('node:path');
const [edge, output, url = 'https://127.0.0.1:15729/'] = process.argv.slice(3);

async function run(page) {
  let stateRoute;
  let stateConnections = 0;
  await page.routeWebSocket(/\/ws\/bus\/.*semantic_surface_fusion.*state/, route => {
    stateRoute = route;
    stateConnections++;
    route.connectToServer();
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  const search = page.locator('#sidebar-search');
  await search.waitFor();
  const canvas = page.locator('#canvas-area');
  const card = name => page.locator('.canvas-card').filter({
    has: page.locator(`.canvas-card-tool[title="${name}"]`),
  });
  async function add(name, x, y) {
    await search.fill(name);
    const tool = page.locator(`.sidebar-tool-list [data-tool-name="${name}"]`);
    await tool.waitFor();
    if (!(await card(name).count())) {
      await tool.click();
      await page.waitForTimeout(500);
      await tool.dragTo(canvas, { targetPosition: { x, y } });
    }
    await card(name).waitFor();
  }
  async function wire(source, target, format) {
    const start = await card(source).locator(`.canvas-port.out[data-format="${format}"]`).boundingBox();
    const end = await card(target).locator(`.canvas-port.in[data-format="${format}"]`).boundingBox();
    if (!start || !end) throw new Error(`Missing ${format} port from ${source} to ${target}`);
    await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
    await page.mouse.down();
    await page.mouse.move(end.x + end.width / 2, end.y + end.height / 2, { steps: 18 });
    await page.mouse.up();
    await page.waitForTimeout(250);
  }
  const replay = 'fusion_test_replay';
  const segment = 'segment_fixture_chair';
  const fusion = 'semantic_surface_fusion';
  await page.getByRole('button', { name: '编辑', exact: true }).click();
  await add(replay, 110, 275);
  await add(segment, 440, 275);
  await add(fusion, 770, 275);
  await wire(replay, segment, 'image/jpeg');
  await wire(segment, fusion, 'image/png');
  await wire(segment, fusion, 'data/json');
  await wire(replay, fusion, 'image/depth-zlib');
  const screenshot = name => page.screenshot({ path: path.join(output, name) });
  await screenshot('01-canvas.png');

  const toggle = page.locator('#canvas-project-toggle');
  const waitToggle = text => toggle.getByText(text, { exact: true }).or(
    page.locator('#canvas-project-toggle').filter({ hasText: text })).first().waitFor();
  const mode = name => page.locator(`.mode-toggle-btn[data-mode="${name}"]`).click();
  const monitor = page.locator('#monitor-dashboard');
  const statePanel = monitor.locator('.monitor-card[data-topic$="/state"]');
  const value = key => statePanel.locator('.kv-cell').filter({
    has: page.getByText(key, { exact: true }),
  }).locator('.kv-value');
  const latest = () => statePanel.locator('.monitor-card-mode-btn[data-mode="kv"]').click();
  const geometry = async () => {
    await value('geometry_scope').filter({ hasText: /^class_visible_surface$/ }).waitFor();
    await value('state').filter({ hasText: /^valid$/ }).waitFor();
    await value('visible_center_m').filter({ hasText: /^-0\.02, -0\.02, 2$/ }).waitFor();
    await value('visible_bounds_m.min').waitFor();
    await page.waitForFunction(() =>
      document.querySelector('.monitor-card[data-topic$="/mask"] img')?.naturalWidth === 64);
  };
  await toggle.click();
  await waitToggle('停止智能控制');
  await mode('monitor');
  await latest();
  await geometry();
  await screenshot('02-live-monitor.png');
  const service = await page.locator(`.sidebar-tool-list [data-tool-name="${replay}"]`).getAttribute('data-mcp-id');
  async function setDepth(action) {
    const response = await page.request.post(new URL(`/api/mcp/${service}/call`, url).toString(), {
      data: { tool: replay, arguments: { action } },
    });
    const result = await response.json();
    if (!response.ok() || result.code !== 200) throw new Error(JSON.stringify(result));
  }
  await setDepth('pause_depth');
  await value('reason').filter({ hasText: /^stream_stale:depth$/ }).waitFor();
  await value('state').filter({ hasText: /^stale$/ }).waitFor();
  await value('visible_center_m').filter({ hasText: /^—$/ }).waitFor();
  await value('visible_bounds_m.min').waitFor({ state: 'detached' });
  await screenshot('03-depth-paused.png');
  await setDepth('resume_depth');
  await geometry();
  await screenshot('04-depth-resumed.png');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await mode('monitor');
  await latest();
  await geometry();
  await screenshot('05-after-refresh.png');
  if (!stateRoute) throw new Error('Monitor has no fusion state WebSocket');
  const beforeDisconnect = stateConnections;
  await stateRoute.close();
  const reconnectDeadline = Date.now() + 15000;
  while (stateConnections <= beforeDisconnect && Date.now() < reconnectDeadline) {
    await page.waitForTimeout(100);
  }
  if (stateConnections <= beforeDisconnect) throw new Error('Monitor did not reconnect');
  await geometry();
  await screenshot('06-socket-reconnected.png');
  await mode('configure');
  await waitToggle('停止智能控制');
  await toggle.click();
  await waitToggle('开启智能控制');
  await screenshot('07-stopped.png');
  await toggle.click();
  await waitToggle('停止智能控制');
  await mode('monitor');
  await latest();
  await geometry();
  await screenshot('08-restarted.png');
  await mode('configure');
  await toggle.click();
  await waitToggle('开启智能控制');
  console.log('FUSION_BROWSER_ACCEPTANCE_PASS');
}

(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: edge });
  const page = await browser.newPage({ ignoreHTTPSErrors: true, viewport: { width: 1600, height: 1000 } });
  try {
    await run(page);
  } catch (error) {
    await page.screenshot({ path: path.join(output, 'failure.png') });
    console.error((await page.locator('body').innerText()).slice(0, 3000));
    throw error;
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
