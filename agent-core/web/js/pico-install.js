/* The install code grants an APK download only. Invitation secrets stay in the fragment. */
(() => {
  const $ = id => document.getElementById(id);
  const codePattern = /^(?:[A-HJ-NP-Z2-9]{12}|[A-Za-z0-9_-]{16})$/;
  const path = location.pathname.replace(/\/$/, '');
  const token = path.startsWith('/pico/') ? path.slice('/pico/'.length) : '';
  if (!token) {
    $('code-form').hidden = false;
    $('code-form').onsubmit = event => {
      event.preventDefault();
      const raw = $('code').value.trim();
      const normalized = raw.replace(/[\s-]/g, '').toUpperCase();
      const code = /^[A-HJ-NP-Z2-9]{12}$/.test(normalized) ? normalized : raw;
      if (!codePattern.test(code)) {
        $('error').textContent = '请核对安装码；可直接输入或粘贴，横线和空格可以省略。';
        return;
      }
      location.assign('/pico/' + code);
    };
    return;
  }
  $('entry-link').hidden = false;
  if (!codePattern.test(token)) {
    $('error').textContent = '安装链接无效，请重新输入电脑端显示的安装码。';
    return;
  }
  $('installation').hidden = false;
  function updateInvitation() {
    const payload = location.hash.slice(1);
    $('connect').hidden = true;
    $('connect').removeAttribute('href');
    if (/^[A-Za-z0-9_-]{1,8192}$/.test(payload)) {
      $('connect').href = 'motus-teleop://connect#' + payload;
      $('connect').hidden = false;
      $('invite-hint').textContent = '已有 App 时可直接导入本次邀请；首次安装后也可直接打开 App 查找机器人。';
    }
  }
  updateInvitation();
  addEventListener('hashchange', updateInvitation);
  fetch(path + '/package', {cache:'no-store', credentials:'omit', redirect:'error'})
    .then(async response => {
      if (!response.ok) throw new Error(response.status === 410 ?
        '安装码已过期或被更新。请在电脑端重新生成，再输入新安装码。' : '暂时无法获取安装包，请稍后重试。');
      return response.json();
    })
    .then(info => {
      $('package').textContent = `版本 ${info.version} · ${(info.size_bytes / 1024 / 1024).toFixed(1)} MB`;
      $('digest').textContent = info.sha256;
      $('apk').href = path + '/apk';
      $('apk').hidden = false;
    })
    .catch(error => { $('package').textContent = ''; $('error').textContent = error.message; });
})();
