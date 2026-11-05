"""Offline summary; geometry gates live in the packaged plugin."""
from plugins.pointcloud_slam_geometry import classify_registration


def summarize(items):
    """Summarize separately supplied labels; missing classes have unknown rates."""
    counts = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
    for item in items:
        if (not isinstance(item, dict) or type(item.get('match')) is not bool or
                type(item.get('passes')) is not bool):
            raise ValueError('each labelled outcome needs boolean match and passes')
        key = ('tp' if item['passes'] else 'fn') if item['match'] else (
              'fp' if item['passes'] else 'tn')
        counts[key] += 1
    positives, negatives = counts['tp'] + counts['fn'], counts['fp'] + counts['tn']
    return {**counts, 'recall': counts['tp'] / positives if positives else None,
            'false_positive_rate': counts['fp'] / negatives if negatives else None}
