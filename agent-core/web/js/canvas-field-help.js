const HTML_ESCAPES = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => HTML_ESCAPES[character]);
}

/**
 * Builds accessible help wiring for one canvas form field.
 *
 * @param {string | null | undefined} description Full field help text.
 * @param {string} inputId Deterministic ID of the associated input.
 * @returns {{ inputAttributes: string, helpMarkup: string }}
 */
export function renderCanvasFieldHelp(description, inputId) {
  if (description == null || String(description).trim() === '') {
    return { inputAttributes: '', helpMarkup: '' };
  }

  const helpId = `${String(inputId)}-help`;
  const escapedHelpId = escapeHtml(helpId);

  return {
    inputAttributes: ` aria-describedby="${escapedHelpId}"`,
    helpMarkup: `<div class="canvas-field-help" id="${escapedHelpId}">${escapeHtml(description)}</div>`
  };
}
