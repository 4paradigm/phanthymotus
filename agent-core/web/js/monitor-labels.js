/** Use the publishing device identity, never topic order, to label PiPER photos. */
export function monitorTopicLabel(topicPath, mcp) {
  const photo = (mcp?.tools || []).find(tool => tool?.name === 'piper_photo'
    && (tool.topic_out || []).some(out => out.topic === topicPath));
  if (photo && mcp.name) return `${mcp.name}｜相机画面`;
  return topicPath.split('/').filter(Boolean).pop() || topicPath;
}
