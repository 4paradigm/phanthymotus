/**
 * inspector-host.js — No-op shim.
 * Inspector is now proxied through agent-core at /api/inspector/* and /ws/bus/*.
 * This file is kept to avoid breaking any imports.
 */

export async function initInspectorHost() {}

export function getInspectorHost() { return null; }
