// Public platform metadata and local-only input validation. Never log or store stream secrets.
export const PLATFORMS = Object.freeze({
  xiaohongshu: {
    name: '小红书',
    helpUrl: 'https://www.xiaohongshu.com/zhibo/obs',
    steps: ['先完成手机普通直播，再在电脑 OBS 官方网页申请权限。', '直播助手选择「三方推流」，设置本场封面和标题。', '复制服务器与推流码，回到 Bumi 检查并开播。'],
    defaultServer: 'rtmp://live-push.xhscdn.com/live',
  },
  huya: {
    name: '虎牙',
    helpUrl: 'https://help.huya.com/355',
    steps: ['登录虎牙并完成主播实名认证。', '直达主播设置，选择「开通远程推流」。', '填写直播分类和标题，获取完整 RTMP 地址，再交给 Bumi。'],
    defaultServer: '',
  },
});

// User guidance is data, not Canvas branches. These are instructions, not account-status claims.
export const READINESS = Object.freeze([
  { id: 'new', title: '我还没有账号' },
  { id: 'permission', title: '我有账号，但还不能直播' },
  { id: 'find', title: '我能直播，但找不到直播信息' },
  { id: 'ready', title: '我已经复制好直播信息' },
]);

export const READINESS_BY_PLATFORM = Object.freeze({
  xiaohongshu: [
    { id: 'new', title: '我还没完成首次手机直播' },
    { id: 'permission', title: '手机能直播，但没有三方推流' },
    { id: 'find', title: '有三方推流，但还没拿到推流码' },
    { id: 'ready', title: '我已经有本场服务器和推流码' },
  ],
  huya: READINESS,
});

export function getReadiness(platform) {
  return READINESS_BY_PLATFORM[platform] || READINESS;
}

export const GUIDES = Object.freeze({
  xiaohongshu: {
    new: [
      { stage: '普通直播', title: '打开小红书主播中心', detail: '在手机小红书 App 登录本人账号，进入「创作者中心 → 主播中心 → 去开播 → 开始直播」。', place: '手机 App' },
      { stage: '普通直播', title: '准备首次直播封面和标题', detail: '上传直播封面，填写标题。', place: '手机 App' },
      { stage: '普通直播', title: '完成实名认证', detail: '按提示完成实名认证。', place: '手机 App' },
      { stage: '普通直播', title: '完成一次真正的手机直播', detail: '点击「开始直播」，按提示同意协议并进入直播。确认开播成功后，再正常「结束直播」。', place: '手机 App' },
    ],
    permission: [
      { stage: 'OBS 权限', title: '用同一账号打开 OBS 官方网页', detail: '在电脑浏览器打开小红书 OBS 页面，用刚才成功手机开播的同一个账号扫码或手机验证登录。', url: 'https://www.xiaohongshu.com/zhibo/obs', place: '电脑网页' },
      { stage: 'OBS 权限', title: '申请 OBS 开播权限', detail: '如页面显示「暂无 OBS 开播权限」，点击「申请」。成功开通后会进入直播助手下载页面，可直接继续下载。', url: 'https://www.xiaohongshu.com/zhibo/obs', place: '电脑网页' },
      { stage: '电脑助手', title: '下载小红书直播助手', detail: '在跳转的官方页面下载安装小红书直播助手。', url: 'https://www.xiaohongshu.com/zhibo/robs', place: '电脑' },
      { stage: '电脑助手', title: '扫码登录电脑直播助手', detail: '打开小红书电脑直播助手，用刚才申请权限的同一账号扫码登录。', place: '电脑直播助手' },
    ],
    find: [
      { stage: '本场推流', title: '切换到「三方推流」', detail: '在电脑直播助手左上角选择「三方推流」。', place: '电脑直播助手' },
      { stage: '本场推流', title: '设置本场直播封面和标题', detail: '点击「去设置」，填写标题和封面并保存。', place: '电脑直播助手' },
      { stage: '本场推流', title: '复制本场服务器和推流码', detail: '复制「服务器地址」和「直播推流码」，回到 Bumi 分别粘贴。', place: '电脑直播助手' },
    ],
  },
  huya: {
    new: [
      { stage: '账号准备', title: '登录虎牙', detail: '打开虎牙官网，点击右上角登录。', url: 'https://www.huya.com/', linkText: '打开虎牙登录页面', place: '电脑网页' },
      { stage: '账号准备', title: '完成主播实名认证', detail: '进入认证页面，按提示完成实名认证。', url: 'https://i.huya.com/index.php?m=ProfileAuth', linkText: '直达主播认证页面', helpUrl: 'https://help.huya.com/165', place: '电脑网页' },
    ],
    permission: [
      { stage: '账号准备', title: '完成主播实名认证', detail: '主播尚未认证时，在虎牙个人认证页面完成认证；认证后再去主播设置获取远程推流地址。', url: 'https://i.huya.com/index.php?m=ProfileAuth', linkText: '直达主播认证页面', helpUrl: 'https://help.huya.com/165', place: '电脑网页' },
    ],
    find: [
      { stage: '远程推流', title: '打开主播设置', detail: '直接进入虎牙主播设置。若页面要求登录，请登录刚才完成认证的同一账号。', url: 'https://i.huya.com/index.php?m=ProfileSetting', linkText: '直达虎牙主播设置', helpUrl: 'https://help.huya.com/355', place: '电脑网页' },
      { stage: '远程推流', title: '选择「开通远程推流」', detail: '在主播设置中找到「普通/VR 远程推流」，点击「开通远程推流」。', url: 'https://i.huya.com/index.php?m=ProfileSetting', linkText: '打开远程推流设置', helpUrl: 'https://help.huya.com/355', place: '电脑网页' },
      { stage: '本场直播', title: '填写直播分类和标题', detail: '选择本场直播品类，再选择「普通直播」，按虎牙页面提示填写直播标题。', url: 'https://i.huya.com/index.php?m=ProfileSetting', linkText: '进入直播设置页面', helpUrl: 'https://help.huya.com/355', place: '电脑网页' },
      { stage: '本场直播', title: '获取并复制完整 RTMP 地址', detail: '点击「获取推流地址并开播」，复制完整 RTMP 地址，粘贴到 Bumi。', url: 'https://i.huya.com/index.php?m=ProfileSetting', linkText: '前往获取推流地址', helpUrl: 'https://help.huya.com/355', place: '电脑网页' },
    ],
  },
});

export const TROUBLESHOOT = Object.freeze({
  xiaohongshu: {
    '打开小红书主播中心': '没有直播入口时，先在手机 App 的创作者中心或主播中心查看账号提示。普通直播权限与 OBS 推流权限是两回事。',
    '准备首次直播封面和标题': '按当前账号页面要求填写标题和封面。不同账号可能要求不止一张封面，不要照搬其他账号的字段数量。',
    '完成实名认证': '实名认证只能在小红书官方页面完成；如审核失败，请按照具体原因处理，不要跳过。',
    '完成一次真正的手机直播': '需要确认首次直播真正开始，再正常结束。未获得普通直播资格时，请先处理平台提示。',
    '用同一账号打开 OBS 官方网页': '网址是 xiaohongshu.com/zhibo/obs；请确认登录的就是刚才首播成功、准备接入 Bumi 的账号。',
    '申请 OBS 开播权限': '点击申请成功后通常直接进入下载页。若仍提示无权限或按钮不可用，请先查看平台给出的资格、审核或错误提示，不要跳过申请。',
    '下载小红书直播助手': '这是小红书官方电脑直播助手，不是 OBS 软件。Bumi 自己可以发送 RTMP 直播流。',
    '扫码登录电脑直播助手': '确保使用申请 OBS 权限的同一小红书账号；必要时退出旧账号重新扫码。',
    '切换到「三方推流」': '如果菜单仍只有「常规模式」「嘉宾模式」，说明这个账号还不能按三方推流流程操作，请回到 OBS 官方权限页面核查。',
    '设置本场直播封面和标题': '选择「三方推流」后点「去设置」，完成本场封面和标题并保存。「开始直播」是后面的步骤，不要现在直接点击。',
    '复制本场服务器和推流码': 'Bumi 页面与小红书一样，分别填写「服务器地址」和「直播推流码」。系统会自动合成 RTMP 地址，不需要人工拼接。不要泄露推流密钥。',
  },
  huya: {
    '登录虎牙': '网页登录后仍要求登录时，确认同一个浏览器中已登录本人账号。可以从右上角头像进入「个人中心」。',
    '完成主播实名认证': '可直接打开认证页面。如果要求先登录，登录后重新打开链接；若认证失败，以虎牙官方提示为准，不能借用他人身份。',
    '打开主播设置': '如果直达链接没有显示设置，先进入 https://i.huya.com/，在左侧找到「我是主播 → 主播设置」。也可点击下方官方操作说明。',
    '选择「开通远程推流」': '「远程推流」是主播设置中的功能，不一定有独立网址。找不到时先确认主播认证完成，再对照官方说明核查当前账号。',
    '填写直播分类和标题': '依次选择直播品类、「普通直播」及本场标题。这几项都在主播设置的远程推流区域完成。',
    '获取并复制完整 RTMP 地址': '这是完整的一条 RTMP 地址，Bumi 直接粘贴即可。点击「获取推流地址并开播」可能同步启动平台侧直播，务必先准备好 Bumi。',
  },
});
export function getGuideHelp(platform, title) {
  return TROUBLESHOOT[platform]?.[title]
    || '找不到入口时，请确认是本人账号，检查平台资格和审核提示。暂时无法继续时不要跳过认证；可以稍后回来。';
}

export function getGuide(platform, readiness) {
  if (!Object.hasOwn(PLATFORMS, platform)) return [];
  if (readiness === 'ready') return [];
  const guides = GUIDES[platform];
  const steps = guides[readiness] || [];
  if (platform === 'xiaohongshu' && readiness === 'new') {
    return [...steps, ...guides.permission, ...guides.find];
  }
  return readiness === 'new' || readiness === 'permission'
    ? [...steps, ...guides.find] : steps;
}

const MAX_URL = 4096;
const STREAM_PROTOCOLS = new Set(['rtmp:', 'rtmps:']);
const PORTS = new Set(['', '1935', '443']);
// Both hosts were observed in the official Xiaohongshu live assistant.
const XIAOHONGSHU_PUSH_HOSTS = new Set(['live-push.xhscdn.com', 'live-push-reg-tx.xhscdn.com']);

export function makeXiaohongshuStreamUrl(server, streamKey) {
  const base = String(server || '').trim();
  const key = String(streamKey || '').trim();
  if (!base) throw new Error('请先填写服务器地址');
  if (!key) throw new Error('请先填写推流码');
  if (/^rtmps?:\/\//i.test(key)) {
    throw new Error('推流码栏只需填写推流码，不要粘贴完整推流地址');
  }
  return makeStreamUrl('xiaohongshu', key, base);
}

export function makeStreamUrl(platform, input, server = '') {
  if (!Object.hasOwn(PLATFORMS, platform)) throw new Error('暂不支持这个直播平台');
  const value = String(input || '').trim();
  if (!value) throw new Error('请先粘贴本场直播信息');
  let url = value;
  if (!/^rtmps?:\/\//i.test(value)) {
    const base = String(server || PLATFORMS[platform].defaultServer || '').trim();
    if (!base) throw new Error('还需要填写直播平台提供的服务器地址');
    if (!/^rtmps?:\/\//i.test(base) || /[?#]/.test(base)) {
      throw new Error('服务器地址不正确，请从直播平台重新复制');
    }
    url = base.replace(/\/+$/, '') + '/' + value.replace(/^\/+/, '');
  }
  if (url.length > MAX_URL || /[\s\x00-\x1f\x7f]/.test(url)) {
    throw new Error('直播信息格式不正确，请重新复制');
  }
  let parsed;
  try { parsed = new URL(url); } catch { throw new Error('直播信息格式不正确，请重新复制'); }
  if (!STREAM_PROTOCOLS.has(parsed.protocol) || !PORTS.has(parsed.port)
      || parsed.username || parsed.password || parsed.hash) {
    throw new Error('推流地址格式不正确');
  }
  const host = parsed.hostname.toLowerCase();
  if (platform === 'xiaohongshu') {
    const segments = parsed.pathname.split('/');
    if (!XIAOHONGSHU_PUSH_HOSTS.has(host) || segments.length !== 3
        || segments[1] !== 'live' || !segments[2]
        || !parsed.searchParams.get('txSecret') || !parsed.searchParams.get('txTime')) {
      throw new Error('这不是有效的小红书推流信息，请检查是否复制完整');
    }
  } else if (!((host === 'tx.direct.huya.com' || host.endsWith('.direct.huya.com'))
      && parsed.pathname.startsWith('/huyalive/')
      && parsed.pathname.length > '/huyalive/'.length)) {
    throw new Error('这不是有效的虎牙推流地址，请从虎牙主播中心重新复制');
  }
  return url;
}
