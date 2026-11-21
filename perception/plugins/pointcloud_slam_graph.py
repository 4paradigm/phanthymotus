"""Connectivity through actual pose constraints in RTAB-Map graphs."""


def connected_nodes(nodes, links, anchors):
    # Neighbor, global/local closure, user closure and merged neighbor.
    # Virtual links, geometric priors and landmark IDs are insufficient here.
    adjacency = {node: [] for node in nodes if node > 0}
    for source, target, kind in links:
        if kind in (0, 1, 2, 3, 4, 6) and source != target and source in adjacency and target in adjacency:
            adjacency[source].append(target)
            adjacency[target].append(source)
    reached = adjacency.keys() & set(anchors)
    pending = list(reached)
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached
