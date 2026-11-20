import test from 'node:test';
import assert from 'node:assert/strict';
import { PLATFORMS, getReadiness, getGuide, getGuideHelp, makeStreamUrl, makeXiaohongshuStreamUrl } from './platforms.js';
import { callLiveStream, saveLiveStreamConfig, authorizeStoredLiveStream, startApprovedLiveStream } from './api.js';
import { mountLiveStreamCard } from './card-ui.js';

const XHS = 'rtmp://live-push.xhscdn.com/live/987?txSecret=fake_only&txTime=ffffffff';
const HUYA = 'rtmp://tx.direct.huya.com/huyalive/fake_only?seq=11&type=simple';
const stubFetch = async (fn, cb) => {
  const original = globalThis.fetch;
  globalThis.fetch = fn;
  try { await cb(); } finally { globalThis.fetch = original; }
};

test('card UI loads together with its API export contract', () => {
  assert.equal(typeof mountLiveStreamCard, 'function');
  assert.equal(typeof authorizeStoredLiveStream, 'function');
  assert.equal(typeof startApprovedLiveStream, 'function');
});

test('only supported platforms are selectable', () => {
  assert.deepEqual(Object.keys(PLATFORMS), ['xiaohongshu', 'huya']);
});
test('all readiness branches are represented', () => {
  for (const platform of Object.keys(PLATFORMS)) {
    assert.ok(getGuide(platform, 'new').length >= 5);
    assert.ok(getGuide(platform, 'permission').length >= 4);
    assert.ok(getGuide(platform, 'find').length >= 3);
    assert.deepEqual(getGuide(platform, 'ready'), []);
    for (const step of getGuide(platform, 'new')) {
      assert.ok(getGuideHelp(platform, step.title).length > 12);
    }
  }
});
test('guides use official destinations and avoid secrets', () => {
  for (const p of Object.keys(PLATFORMS)) {
    for (const step of getGuide(p, 'new')) {
      assert.ok(step.title && step.detail);
      if (!step.url) {
        assert.ok(step.place, 'App steps do not need misleading web links');
        continue;
      }
      const url = new URL(step.url);
      assert.ok(url.protocol === 'https:');
      assert.ok(p === 'huya' ? url.hostname.endsWith('huya.com') : url.hostname.endsWith('xiaohongshu.com'));
    }
  }
});
test('Xiaohongshu includes OBS application and correct preparation order', () => {
  assert.match(getReadiness('xiaohongshu')[1].title, /三方推流/);
  const newSteps = getGuide('xiaohongshu', 'new').map(item => item.title);
  const required = [
    '打开小红书主播中心',
    '准备首次直播封面和标题',
    '完成实名认证',
    '完成一次真正的手机直播',
    '用同一账号打开 OBS 官方网页',
    '申请 OBS 开播权限',
    '下载小红书直播助手',
    '扫码登录电脑直播助手',
    '切换到「三方推流」',
    '设置本场直播封面和标题',
    '复制本场服务器和推流码',
  ];
  assert.deepEqual(newSteps, required);
  const permissionSteps = getGuide('xiaohongshu', 'permission').map(item => item.title);
  assert.equal(permissionSteps[0], '用同一账号打开 OBS 官方网页');
  assert.equal(permissionSteps.at(-1), '复制本场服务器和推流码');
  const findSteps = getGuide('xiaohongshu', 'find').map(item => item.title);
  assert.deepEqual(findSteps, required.slice(-3));
  const application = getGuide('xiaohongshu', 'new')[5];
  assert.equal(application.url, 'https://www.xiaohongshu.com/zhibo/obs');
  assert.equal(newSteps.length, 11);
  assert.equal(newSteps[newSteps.indexOf('申请 OBS 开播权限') + 1], '下载小红书直播助手');
  assert.ok(!newSteps.includes('确认权限已经开放'));
  assert.ok(!newSteps.some(title => /验证码/.test(title)));
});

test('Huya uses verified direct links at every guided step', () => {
  const expectedSettings = 'https://i.huya.com/index.php?m=ProfileSetting';
  const expectedAuth = 'https://i.huya.com/index.php?m=ProfileAuth';
  const full = getGuide('huya', 'new');
  assert.equal(full.length, 6);
  assert.deepEqual(full.map(step => step.title), [
    '登录虎牙',
    '完成主播实名认证',
    '打开主播设置',
    '选择「开通远程推流」',
    '填写直播分类和标题',
    '获取并复制完整 RTMP 地址',
  ]);
  assert.equal(full[0].url, 'https://www.huya.com/');
  assert.equal(full[1].url, expectedAuth);
  for (const step of full.slice(2)) {
    assert.equal(step.url, expectedSettings);
    assert.equal(step.helpUrl, 'https://help.huya.com/355');
  }
  for (const readiness of ['new', 'permission', 'find']) {
    for (const step of getGuide('huya', readiness)) {
      assert.ok(step.url?.startsWith('https://'));
      assert.ok(step.linkText, 'each step should have a specific direct-link action');
    }
  }
  assert.equal(getGuide('huya', 'permission')[0].url, expectedAuth);
});

test('accepts Xiaohongshu complete URL and stream key from official server', () => {
  assert.equal(makeStreamUrl('xiaohongshu', XHS), XHS);
  assert.equal(makeStreamUrl('xiaohongshu', '987?txSecret=fake_only&txTime=ffffffff'), XHS);
});
test('accepts the official reg-tx server, rejects near-match hosts', () => {
  const key = '987?txSecret=fake_only&txTime=ffffffff';
  const regServer = 'rtmp://live-push-reg-tx.xhscdn.com/live';
  assert.equal(makeXiaohongshuStreamUrl(regServer, key), regServer + '/' + key);
  assert.equal(makeStreamUrl('xiaohongshu', regServer + '/' + key), regServer + '/' + key);
  for (const badHost of [
    'live-push-reg-tx.xhscdn.com.evil.example',
    'fake-live-push-reg-tx.xhscdn.com',
    'live-push-reg-tx.evil.example',
  ]) {
    assert.throws(() => makeXiaohongshuStreamUrl('rtmp://' + badHost + '/live', key));
  }
});

test('Xiaohongshu separate server and stream key are joined safely', () => {
  const server = 'rtmp://live-push.xhscdn.com/live';
  const key = '987?txSecret=fake_only&txTime=ffffffff';
  assert.equal(makeXiaohongshuStreamUrl(server, key), XHS);
  assert.equal(makeXiaohongshuStreamUrl(' ' + server + '/ ', ' ' + key + ' '), XHS);
  assert.throws(() => makeXiaohongshuStreamUrl('', key), /服务器地址/);
  assert.throws(() => makeXiaohongshuStreamUrl(server, ''), /推流码/);
  assert.throws(() => makeXiaohongshuStreamUrl(server, XHS), /只需填写推流码/);
  assert.throws(() => makeXiaohongshuStreamUrl('rtmp://evil.example/live', key), /小红书推流信息/);
  assert.throws(() => makeXiaohongshuStreamUrl(server, '987?txTime=ffffffff'), /小红书推流信息/);
  assert.throws(() => makeXiaohongshuStreamUrl('https://live-push.xhscdn.com/live', key));
});

test('accepts Huya full URL only, never guesses Huya server', () => {
  assert.equal(makeStreamUrl('huya', HUYA), HUYA);
  assert.throws(() => makeStreamUrl('huya', 'fake_only'), /服务器地址/);
});
test('rejects wrong hosts, incomplete parameters, cross-platform info, and invalid protocols', () => {
  assert.throws(() => makeStreamUrl('xiaohongshu', 'rtmp://evil.example/live/123?txSecret=fake&txTime=fake'));
  assert.throws(() => makeStreamUrl('xiaohongshu', '123?txSecret=fake'));
  assert.throws(() => makeStreamUrl('xiaohongshu', HUYA));
  assert.throws(() => makeStreamUrl('huya', XHS));
  assert.throws(() => makeStreamUrl('huya', 'https://tx.direct.huya.com/huyalive/123'));
  assert.throws(() => makeStreamUrl('huya', 'rtmp://tx.direct.huya.com/huyalive/'));
});
test('config must be confirmed by Driver, not just HTTP 200', async () => {
  await stubFetch(async () => ({
    ok: true, json: async () => ({ code: 200, applied: false, ready_to_go_live: true }),
  }), async () => {
    await assert.rejects(saveLiveStreamConfig('mcp-test', 'card-a', XHS), /无法确认/);
  });
  await stubFetch(async () => ({
    ok: true, json: async () => ({ code: 200, applied: true, ready_to_go_live: false }),
  }), async () => {
    await assert.rejects(saveLiveStreamConfig('mcp-test', 'card-a', XHS), /新的直播信息/);
  });
});
test('save sends secret only in authenticated configuration request body', async () => {
  const seen = [];
  await stubFetch(async (url, options) => {
    seen.push({ url, options });
    return { ok: true, json: async () => ({ code: 200, applied: true, ready_to_go_live: true }) };
  }, async () => {
    assert.equal(await saveLiveStreamConfig('mcp', 'card-a', HUYA), true);
    assert.equal(seen[0].url, '/api/canvas/tool-config/mcp/live_stream/card-a');
    assert.equal(JSON.parse(seen[0].options.body).rtmp_url, HUYA);
    assert.equal(seen[0].options.method, 'PUT');
  });
});
test('explicit start authorization uses a POST without resending the secret', async () => {
  const calls = [];
  await stubFetch(async (url, options) => {
    calls.push({ url, options });
    return {
      ok: true,
      json: async () => ({ code: 200, ready_to_go_live: true }),
    };
  }, async () => {
    assert.equal(await authorizeStoredLiveStream('qa-mcp', 'qa-card'), true);
  });
  assert.equal(calls[0].url, '/api/canvas/tool-config/qa-mcp/live_stream/qa-card/authorize-start');
  assert.equal(calls[0].options.method, 'POST');
  assert.ok(!calls[0].options.body, 'No RTMP URL must be sent from browser on rearm');
});

test('explicit reauthorization refuses an unconfirmed Driver response', async () => {
  await stubFetch(async () => ({
    ok: true,
    json: async () => ({ code: 200, ready_to_go_live: false }),
  }), async () => {
    await assert.rejects(authorizeStoredLiveStream('qa-mcp', 'qa-card'), /无法重新授权/);
  });
});

test('confirmed public broadcasting uses authenticated Canvas endpoint', async () => {
  const calls = [];
  await stubFetch(async (url, options) => {
    calls.push({ url, options });
    return { ok: true, json: async () => ({ code: 200, state: 'starting' }) };
  }, async () => {
    assert.equal((await startApprovedLiveStream('qa-mcp', 'qa-card')).state, 'starting');
  });
  assert.equal(calls[0].url, '/api/canvas/tool-config/qa-mcp/live_stream/qa-card/go-live');
  assert.equal(calls[0].options.method, 'POST');
  assert.equal(calls[0].options.body, undefined);
});

test('MCP business-level errors are not treated as success', async () => {
  await stubFetch(async () => ({
    ok: true, json: async () => ({
      code: 200, data: [{ text: JSON.stringify({ ok: false, error_code: 'APPROVAL_REQUIRED' }) }],
    }),
  }), async () => {
    await assert.rejects(callLiveStream('mcp', 'go_live'), /已失效/);
  });
});
test('successful MCP status is decoded', async () => {
  await stubFetch(async () => ({
    ok: true, json: async () => ({ code: 200, data: [{ text: '{"state":"streaming","video_frames":33}' }] }),
  }), async () => {
    const status = await callLiveStream('mcp', 'info');
    assert.equal(status.state, 'streaming');
    assert.equal(status.video_frames, 33);
  });
});
