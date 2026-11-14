/** QQ enhancement for the official Channels panel, scoped to QQ selection only.
 * Telegram, Slack, Feishu, LAN and the built-in Users controls remain untouched.
 */
import { openQQWizard } from './channel-setup.js?v=8';
import { initQQManagement } from './channel-management.js?v=6';
import { initQQSettingsUI, interceptLegacyQQConfigure } from './qq-settings-ui.js?v=7';

function boot() {
  const addButton = document.getElementById('channel-add-btn');
  const list = document.getElementById('channel-list');
  if (!addButton || !list) return;

  // Dedicated QQ details, initially hidden. Never insert QQ settings into
  // the common Users section used by the other messaging platforms.
  const details = document.createElement('div');
  details.id = 'channel-details-panel';
  details.className = 'channel-details-panel';
  details.hidden = true;
  list.after(details);
  initQQManagement();
  initQQSettingsUI();
  interceptLegacyQQConfigure();

  const refreshAfterSetup = () => {
    if (document.getElementById('qq-guided-setup')) return;
    document.getElementById('channel-close')?.click();
    document.getElementById('btn-channels')?.click();
  };

  // Let the official Add Channel click handler run unchanged. If a QQ-only
  // view is open, close that view first so two forms never overlap.
  addButton.addEventListener('click', () => {
    details.hidden = true;
    document.getElementById('qq-guided-setup')
      ?.querySelector('[data-qq-close]')?.click();
  }, true);

  // Only selecting QQ in the existing native selector triggers our wizard.
  document.addEventListener('change', event => {
    const select = event.target;
    if (select?.id !== 'channel-form-platform' || select.value !== 'qq') return;
    document.getElementById('channel-add-form')?.remove();
    openQQWizard({
      onConnected: refreshAfterSetup,
      onOtherPlatform: () => addButton.click(),
    });
  });

  // QQ uses the existing QQ configure action as its sole management entry.
  // Do not add another button or a QQ-specific empty-state shortcut.
  // The official Add Channel button remains the only create-channel entry.
  const removeOldQQShortcuts = () => {
    list.querySelector('#qq-official-quick-add')?.remove();
    list.querySelectorAll('[data-qq-open-management]').forEach(node => node.remove());
  };
  new MutationObserver(removeOldQQShortcuts)
    .observe(list, { childList: true, subtree: true });
  removeOldQQShortcuts();

  // QQ identities have their own verified Owner / private / group ACL page.
  // The original Users section is for the other official channel providers.
  // Keep its underlying records untouched; only avoid showing QQ identities
  // twice (especially orphaned QQ users after a Channel was removed).
  const legacyUsersPanel = document.getElementById('channel-users-panel');
  const legacyUsersList = document.getElementById('channel-users-list');
  const syncLegacyUsers = () => {
    if (!legacyUsersPanel || !legacyUsersList) return;
    const hasNonQQChannel = [...list.querySelectorAll('.channel-item')]
      .some(card => card.querySelector('.channel-item-platform')
        ?.textContent?.trim() !== 'qq');
    legacyUsersPanel.hidden = !hasNonQQChannel;
    if (!hasNonQQChannel) return;
    for (const row of legacyUsersList.querySelectorAll(
      '.user-row[data-user-platform="qq"]')) {
      row.remove();
    }
    if (!legacyUsersList.children.length) {
      const empty = document.createElement('div');
      empty.className = 'channel-empty';
      empty.textContent = 'No users yet — they auto-register on first message, or add one below.';
      legacyUsersList.appendChild(empty);
    }
  };
  new MutationObserver(syncLegacyUsers)
    .observe(list, {childList: true, subtree: true});
  new MutationObserver(syncLegacyUsers)
    .observe(legacyUsersList, {childList: true});
  syncLegacyUsers();

  // Closing Channels closes any QQ-only detail view; next opening is
  // indistinguishable from the official default state.
  document.getElementById('channel-close')?.addEventListener('click', () => {
    details.hidden = true;
    document.getElementById('qq-guided-setup')
      ?.querySelector('[data-qq-close]')?.click();
  });
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, {once: true});
} else {
  boot();
}
