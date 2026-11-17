/**
 * QQ Open Platform navigation reference.
 * These are local schematic reproductions of the layout shown in the provided
 * platform screenshots, not screenshots or third-party hosted images.
 * Keep links in one place, and never interpolate unvalidated input into URLs.
 */
export const platformPages = Object.freeze({
  home: 'https://q.qq.com/#/apps',
  bots: 'https://q.qq.com/qqbot/dashboard/',
});

export function developerSettingsURL(appId) {
  const value = String(appId || '').trim();
  return /^[0-9]{5,32}$/.test(value)
    ? 'https://q.qq.com/qqbot/dashboard/manage/' + value + '?sec=dev-settings'
    : platformPages.bots;
}

function tag(x, y, value, width=128, color='#384456', size=16) {
  return `<text x="${x}" y="${y}" font-family="Arial, Microsoft YaHei, sans-serif" font-size="${size}" fill="${color}">${value}</text>`;
}
const box = (x,y,w,h,fill='#fff',r=12,stroke='none') =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}"/>`;
const circle = (x,y,r,fill) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;
const focus = (x,y,w,h) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="13" fill="none" stroke="#f97316" stroke-width="5"/>`;
function shell(content, {sidebar=true}={}) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 600" role="img">
    ${box(0,0,1040,600,'#f5f7fa',0)}
    ${box(0,0,1040,72,'#fff',0)}
    ${circle(37,36,17,'#237bfa')}
    ${tag(67,43,'QQ 开放平台',170,'#182431',26)}
    ${tag(900,44,'帮助文档',120,'#788494',16)}
    ${sidebar ? `${box(0,72,193,528,'#f2f6fc',0)}
      ${tag(28,130,'首页',140,'#354052',20)}
      ${tag(28,191,'机器人',140,'#354052',20)}
      ${tag(28,253,'功能与服务',140,'#354052',20)}` : ''}
    ${content}
  </svg>`;
}
function home() {
  return shell(
    `${box(225,110,780,458)}${tag(270,166,'你好，欢迎来到 QQ 开放平台',600,'#1a2634',27)}
     ${box(254,211,720,72,'#f4f8ff')}${tag(278,255,'机器人　　创建并管理你的 QQ 机器人　　›',640,'#283548',21)}
     ${box(254,304,720,70,'#f5f6f9')}${tag(278,348,'QQ 功能与服务　　　　　　　　　›',600,'#435266',21)}
     ${focus(12,148,170,65)}`);
}
function bots() {
  return shell(
    `${tag(232,141,'我的机器人',300,'#172333',30)}
     ${box(784,101,222,58,'#168bff',14)}${tag(814,138,'＋ 创建机器人',190,'#fff',21)}
     ${box(232,192,460,217,'#fff',16,'#b9dafe')}${circle(283,250,26,'#e4eefc')}
     ${tag(329,257,'我的 QQ 机器人',270,'#172333',22)}
     ${tag(329,306,'离线（服务不可用）',270,'#667588',17)}
     ${tag(260,371,'点击机器人卡片进入管理',400,'#4b6075',16)}
     ${focus(774,92,240,76)}`, {sidebar:false});
}
function botDetails() {
  return shell(
    `${tag(44,143,'←　我的机器人　/　机器人管理',720,'#172333',27)}
     ${box(30,183,290,345,'#fff',22)}
     ${tag(64,235,'账号信息',200,'#2b4057',20)}
     ${tag(64,302,'服务范围',200,'#2b4057',20)}
     ${tag(64,369,'运营数据',200,'#2b4057',20)}
     ${tag(64,436,'高级设置',200,'#2b4057',20)}
     ${tag(64,501,'开发设置',200,'#2456a3',20)}
     ${box(347,182,648,347,'#fff',20)}
     ${tag(388,244,'账号信息',300,'#243247',22)}
     ${tag(388,323,'昵称　　　　我的 QQ 机器人',500,'#435267',19)}
     ${tag(388,391,'管理员　　　当前登录的 QQ 账号',500,'#435267',19)}
     ${focus(45,461,264,64)}`, {sidebar:false});
}
function credentials() {
  return shell(
    `${tag(42,146,'←　机器人管理',700,'#172333',26)}
     ${box(32,184,292,348,'#fff',20)}
     ${tag(65,232,'账号信息',220,'#42556c',19)}
     ${tag(65,300,'服务范围',220,'#42556c',19)}
     ${tag(65,367,'高级设置',220,'#42556c',19)}
     ${box(48,397,257,58,'#e8f6ff',12)}
     ${tag(66,434,'开发设置',220,'#208af8',20)}
     ${box(347,185,658,331,'#fff',20)}
     ${tag(386,232,'AppID 接入凭证',500,'#59677b',17)}
     ${tag(388,296,'AppID',150,'#1e293b',20)}
     ${tag(750,296,'点击复制',150,'#277ded',17)}
     ${tag(388,390,'AppSecret',180,'#1e293b',20)}
     ${tag(757,390,'查看 / 复制',190,'#277ded',17)}
     ${tag(389,434,'不要发送给其他人',400,'#97a1ae',16)}
     ${focus(374,259,574,68)}${focus(374,351,574,72)}`, {sidebar:false});
}
const toDataURL = svg => 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg);
export const guideSlides = Object.freeze([
  {
    title: '进入 QQ 开放平台',
    note: '点击左侧「机器人」。如果尚未登录，先用手机 QQ 登录。',
    link: platformPages.home,
    linkLabel: '打开 QQ 开放平台',
    image: toDataURL(home()),
  },
  {
    title: '找到或创建机器人',
    note: '已有机器人：直接点击机器人卡片。没有机器人：点击右上角「创建机器人」。',
    link: platformPages.bots,
    linkLabel: '打开我的机器人',
    image: toDataURL(bots()),
  },
  {
    title: '打开「开发设置」',
    note: '进入机器人详情后，点击左侧的「开发设置」。看到「离线」可先继续配置。',
    link: platformPages.bots,
    linkLabel: '前往机器人管理后台',
    image: toDataURL(botDetails()),
  },
  {
    title: '找到 AppID 和 AppSecret',
    note: '复制 AppID；再点击 AppSecret 右侧的查看或复制按钮。请不要把密钥发给任何人。',
    link: platformPages.bots,
    linkLabel: '打开后台，选择机器人',
    image: toDataURL(credentials()),
  },
]);
