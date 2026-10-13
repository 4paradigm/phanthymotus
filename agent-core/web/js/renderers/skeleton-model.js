/** A topic detail panel has a topic path, not an MCP server called "detail". */
export async function resolveSkeletonMcp(mcpId, topicPath, fetcher = fetch, options = {}) {
  if (mcpId && mcpId !== 'detail' && mcpId !== 'dashboard') return mcpId;
  if (!topicPath) return null;
  const response = await fetcher('/api/mcp', options);
  if (!response.ok) throw new Error('Cannot discover the skeleton model service');
  const body = await response.json();
  const services = Array.isArray(body.data) ? body.data : [];
  const matches = services.filter(service => {
    const tools = (service.tools || []).filter(tool => tool && typeof tool === 'object');
    const outputs = [...(service.topic_out || []), ...tools.flatMap(tool => tool.topic_out || [])];
    return service.id && service.online !== false && tools.some(tool => tool.name === 'model')
      && outputs.some(output => output.topic === topicPath);
  });
  if (matches.length !== 1) throw new Error('No unique model service for this skeleton topic');
  return matches[0].id;
}
