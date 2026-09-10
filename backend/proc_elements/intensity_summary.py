"""Pixel-pooled intensity summaries; per-image curve-fitting data stays intact."""
import json
from pathlib import Path

import numpy as np


def pixel_statistics(pixels, percentiles):
    if pixels.size == 0:
        return None
    values = np.percentile(pixels, percentiles)
    stat = {
        'min': float(np.min(pixels)), 'max': float(np.max(pixels)),
        'mean': float(np.mean(pixels)), 'median': float(np.median(pixels)),
        'std': float(np.std(pixels)), 'pixel_count': int(pixels.size),
    }
    for p, value in zip(percentiles, values):
        stat[f'p{int(p)}' if float(p).is_integer() else f'p{p}'] = float(value)
    stat['dynamic_range'] = stat['p95'] - stat['p5'] if 'p95' in stat and 'p5' in stat else None
    return stat


def build_intensity_summary(data, masks, percentiles, mode, group_labels):
    if mode not in ('per_image', 'pooled', 'grouped'):
        data['error'] = 'E2508'
        return
    images = data['images']
    single_index = data.get('_single_image_index', -1)
    indices = [single_index] if single_index >= 0 and len(images) == 1 else list(range(len(images)))
    paths = data.get('_original_paths', data.get('paths', []))
    labels = []
    if mode == 'grouped':
        try:
            labels = json.loads(group_labels) if isinstance(group_labels, str) else group_labels
        except (ValueError, TypeError):
            labels = None
        expected = data.get('_original_count', len(images))
        if (not isinstance(labels, list) or len(labels) != expected
                or not all(isinstance(label, str) and label.strip() for label in labels)):
            data['error'] = 'E2509'
            return
        labels = [label.strip() for label in labels]
    groups = {}
    for local_index, image_index in enumerate(indices):
        label = (labels[image_index] if mode == 'grouped' else 'Összes minta' if mode == 'pooled'
                 else Path(paths[image_index]).name if image_index < len(paths) else f'Minta {image_index + 1}')
        key = image_index if mode == 'per_image' else label
        groups.setdefault(key, {'label': label, 'indices': []})['indices'].append(local_index)
    rows = []
    for group in groups.values():
        members = group['indices']
        if mode == 'per_image':
            stat = data['results']['intensity_stats'][members[0]]
            channels = stat['channels'] if stat and 'channels' in stat else [stat]
            rows.append({'label': group['label'], 'image_indices': [indices[members[0]]],
                         'sample_count': 1, 'channels': channels})
            continue
        counts = {1 if images[i].ndim == 2 else images[i].shape[2] for i in members}
        if len(counts) != 1:
            data['error'] = 'E2510'
            return
        count = counts.pop()
        channels = []
        for channel in range(count):
            parts = [(images[i] if images[i].ndim == 2 else images[i][:, :, channel])[masks[i] > 0]
                     for i in members]
            pixels = parts[0] if len(parts) == 1 else np.concatenate(parts)
            channels.append(pixel_statistics(pixels, percentiles))
        rows.append({'label': group['label'], 'image_indices': [indices[i] for i in members],
                     'sample_count': len(members), 'channels': channels})
    samples = []
    per_image_stats = data['results']['intensity_stats']
    for local_index, image_index in enumerate(indices):
        stat = per_image_stats[local_index]
        channels = stat['channels'] if stat and 'channels' in stat else [stat]
        samples.append({'label': str(image_index + 1), 'image_index': image_index, 'channels': channels})
    data['results']['intensity_summary'] = {'mode': mode, 'groups': rows, 'samples': samples}
