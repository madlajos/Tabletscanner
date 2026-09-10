"""Display-only histogram grouping while preserving per-image PCA input."""
import json

import numpy as np


def build_histogram_summary(data, mode='per_image', group_labels='[]'):
    if mode not in ('per_image', 'pooled', 'grouped'):
        data['error'] = 'E2307'
        return
    histograms = data.get('results', {}).get('histograms', [])
    stats = data.get('results', {}).get('histogram_stats', [])
    single_index = data.get('_single_image_index', -1)
    indices = [single_index] if single_index >= 0 and len(histograms) == 1 else list(range(len(histograms)))
    labels = []
    if mode == 'grouped':
        try:
            labels = json.loads(group_labels) if isinstance(group_labels, str) else group_labels
        except (ValueError, TypeError):
            labels = None
        expected = data.get('_original_count', len(histograms))
        if (not isinstance(labels, list) or len(labels) != expected
                or not all(isinstance(label, str) and label.strip() for label in labels)):
            data['error'] = 'E2308'
            return
        labels = [label.strip() for label in labels]
    samples = [
        {'label': str(image_index + 1), 'image_index': image_index,
         'histogram': histograms[local_index], 'stats': stats[local_index]}
        for local_index, image_index in enumerate(indices)
    ]
    grouped = {}
    for local_index, image_index in enumerate(indices):
        label = labels[image_index] if mode == 'grouped' else 'Összes minta' if mode == 'pooled' else str(image_index + 1)
        key = image_index if mode == 'per_image' else label
        grouped.setdefault(key, {'label': label, 'indices': [], 'local_indices': []})
        grouped[key]['indices'].append(image_index)
        grouped[key]['local_indices'].append(local_index)
    groups = []
    for group in grouped.values():
        rows = [np.asarray(histograms[index], dtype=np.float64) for index in group.pop('local_indices')]
        if not rows or any(row.shape != rows[0].shape for row in rows):
            data['error'] = 'E2309'
            return
        groups.append({**group, 'sample_count': len(rows), 'histogram': np.sum(rows, axis=0).tolist()})
    data['results']['histogram_summary'] = {'mode': mode, 'samples': samples, 'groups': groups}
