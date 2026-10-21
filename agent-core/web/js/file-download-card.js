import { openCaptureDownloads } from './vision-capture-download.js';

// File references use the existing canvas connection records, without a DDS topic.
export const FILE_FORMAT = 'file/reference';

export function filePortHtml(card, direction) {
  const supported = direction === 'out' ? card.toolName === 'vision_capture'
    : card.mcpId === 'agentcore' && card.toolName === 'file_download';
  return supported ? `<div class="canvas-port ${direction} fmt-json" data-dir="${direction}"
    data-format="${FILE_FORMAT}" data-topic="" data-idx="files" data-tip="可下载文件"></div>` : '';
}

export function connectedFileSources(cardId, cards, connections) {
  const sources = [];
  for (const conn of connections) {
    if (conn.toCardId !== cardId || conn.format !== FILE_FORMAT ||
        conn.fromPortIdx !== 'files' || conn.toPortIdx !== 'files') continue;
    const source = cards.find(card => card.id === conn.fromCardId && card.toolName === 'vision_capture');
    if (source && !sources.some(card => card.id === source.id)) sources.push(source);
  }
  return sources;
}

export function installFileDownloadCard(el, getSources) {
  const footer = el.querySelector('.canvas-card-footer');
  footer.innerHTML = `<select class="canvas-field-input canvas-download-type" aria-label="下载文件类型">
    <option value="photo">照片</option><option value="video">视频</option></select>
    <button type="button" class="canvas-view-btn">查看文件并下载</button>`;
  footer.querySelector('button').onclick = event => {
    event.stopPropagation();
    openCaptureDownloads(getSources, footer.querySelector('select').value);
  };
}
