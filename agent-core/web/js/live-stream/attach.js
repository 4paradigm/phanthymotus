// Canvas adapter for the live_stream card. No generic Card renderer is modified.
// It mounts only on a recognized live_stream card, keeping its header and Canvas ports.
import { cardsForTool } from '../canvas.js';
import { mountLiveStreamCard } from './card-ui.js';

export function attachLiveStreamCards(getMcps, authorize) {
  const viewport = document.getElementById('canvas-viewport');
  if (!viewport) return { sync: () => {}, destroy: () => {} };
  const mounted = new Map();

  function scan() {
    const currentNodes = new Map(
      [...viewport.children]
        .filter(el => el.classList?.contains('canvas-card') && el.dataset?.cardId)
        .map(el => [el.dataset.cardId, el]),
    );
    const available = new Map();
    for (const mcp of getMcps() || []) {
      if (!mcp.id || !(mcp.tools || []).some(tool =>
        (typeof tool === 'string' ? tool : tool.name) === 'live_stream')) continue;
      for (const card of cardsForTool(mcp.id, 'live_stream')) {
        available.set(card.id, mcp.id);
      }
    }

    for (const [cardId, record] of mounted) {
      if (currentNodes.get(cardId) !== record.element || !available.has(cardId)) {
        record.destroy();
        mounted.delete(cardId);
      }
    }
    for (const [cardId, mcpId] of available) {
      if (mounted.has(cardId)) continue;
      const element = currentNodes.get(cardId);
      const wrap = element?.querySelector('.canvas-card-body-wrap');
      if (!wrap) continue;
      // The original generic action selector and Execute button are not
      // appropriate for first-time users. Preserve header/info/close controls.
      wrap.querySelector('.canvas-card-body')?.remove();
      wrap.querySelector('.canvas-card-footer')?.remove();
      const header = wrap.querySelector('.canvas-card-tool');
      if (header) { header.textContent = 'Bumi 直播助手'; header.title = '直播助手'; }
      const host = document.createElement('div');
      host.className = 'ls-card-host';
      wrap.appendChild(host);
      element.classList.add('live-stream-card');
      const destroy = mountLiveStreamCard(host, { mcpId, cardId, authorize });
      mounted.set(cardId, { element, destroy });
    }
  }

  const observer = new MutationObserver(scan);
  observer.observe(viewport, { childList: true });
  scan();
  return { sync: scan, destroy: () => {
    observer.disconnect();
    for (const record of mounted.values()) record.destroy();
    mounted.clear();
  } };
}
