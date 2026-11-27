"""
④ 入力の加工の追加: 部位別ストロークの部位の境界に線を引く (新規。元リポジトリには無い処理)。

face_input_transform.stroke_parts が塗り分けた部位 (PART_GROUPS。目は eye_scale_w x eye_scale_h 倍に広げた後) が
隣り合う境界だけに線を引く。影の段 (shade_levels) の境界には引かない。顔の外形 (白背景や、パーシングで背景になった
前景 "other" との境界) は silhouette=True のときだけ引く。

線の描き方は元リポジトリの XDoG (guided_diffusion/face_input_transform.py 261-263 行目) と同じく、
ストローク化した後の L* を L* x (1 - strength x 線) で暗くする (a*、b* は変えない)。この後に色の写像をかける。
線の太さは境界の両側に width px ずつ (合計 2 x width px、整列後の 256px で)。blur > 0 なら線をガウスでぼかす (sigma px)。
"""

import numpy as np
import torch as th
from scipy.ndimage import binary_dilation, gaussian_filter

from animeface.face_input_transform import PART_GROUPS, _scale_region

OTHER = 0  # PART_GROUPS の "other" の番号


def part_group_map(fg, labels, eye_scale_w=1.0, eye_scale_h=1.0):
    """
    fg: [H x W] bool、labels: [H x W] (0〜18) -> 塗り分けた部位の番号 [H x W] (PART_GROUPS の番号。背景は -1)。
    stroke_parts と同じ領域・同じ塗る順 (後のものが上) で決める。
    """
    regions = [np.isin(labels, ids) & fg for _, ids in PART_GROUPS]
    eye = [i for i, (name, _) in enumerate(PART_GROUPS) if name == "eye"][0]
    eyes = [_scale_region((labels == c) & fg, eye_scale_w, eye_scale_h) & fg for c in (4, 5)]
    regions[eye] = eyes[0] | eyes[1]
    g = np.full(labels.shape, -1, dtype=np.int64)
    for i, r in enumerate(regions):
        g[r] = i
    return g


def part_boundary_lines(groups, width=1, silhouette=False, blur=0.0):
    """部位の番号 [H x W] -> 線の濃さ [H x W] (0〜1。1 が線)。"""
    parts = groups > OTHER  # "other" と背景は部位として扱わない
    line = np.zeros(groups.shape, dtype=bool)
    for i in np.unique(groups[parts]):
        mi = groups == i
        line |= mi & binary_dilation(parts & ~mi, iterations=width)
    if silhouette:
        line |= parts & binary_dilation(~parts, iterations=width)
    a = line.astype(np.float64)
    if blur > 0:
        a = np.clip(gaussian_filter(a, blur), 0, 1)
    return a * parts


def draw_part_edges(lab, fg, labels, strength, width=1, silhouette=False, blur=0.0, eye_scale_w=1.0,
                    eye_scale_h=1.0):
    """
    lab: [N x 3 x H x W] (ストローク化した後の CIELAB)、fg: [N x 1 x H x W]、labels: [N x H x W]
    -> 部位の境界の L* を暗くした lab (新しいテンソル)。
    """
    out = lab.clone()
    for k in range(lab.shape[0]):
        g = part_group_map(fg[k, 0].numpy() > 0.5, labels[k].numpy(), eye_scale_w, eye_scale_h)
        a = th.from_numpy(part_boundary_lines(g, width, silhouette, blur)).to(out.dtype)
        out[k, 0] = out[k, 0] * (1 - strength * a)
    return out
