"""
変換の出力 (アニメ顔) の目の領域と、瞳の色の置き換え。目の位置は scripts/detect_anime_eyes.py
(hysts/anime-face-detector の 28 点のうち両目の各 6 点) の json から決める。scripts/eye_recolor.py
(色の置き換え) と diagnostics/stroke_eval.py (瞳の色のばらつき) が使う。

目の領域: 6 点の凸包を grow_px 画素だけ膨らませる (負なら削る)。点の信頼度の平均が min_score 未満の目は使わない。
画素の選び方 (select):
  dark_chroma  L* < l_max かつ彩度 C* > c_min (暗く、彩度のある画素)
  interior     L* > l_line (目の輪郭の暗い線を除いた、目の内側の全体。白目や瞳の円盤も含む)
色の置き換え: 選んだ画素の a*, b* を目標の (a*, b*) へ strength の割合だけ寄せる。L* はそのまま。
  選んだ画素のマスクを feather_sigma のガウスでぼかした重みでかける (境界をなめらかにする)。
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import lab2rgb, rgb2lab

SELECT_CHOICES = ("dark_chroma", "interior")


def eye_region(entry, shape, min_score=0.3, grow_px=-1):
    """detect_anime_eyes.py の 1 枚分の結果 entry から目の領域 (bool [H x W]) を作る。目が無ければ全て False。"""
    m = np.zeros(shape, np.uint8)
    if entry:
        for e in entry["eyes"]:
            if e["score"] >= min_score:
                pts = np.round(np.asarray(e["points"], dtype=np.float64)).astype(np.int32)
                cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
    if grow_px:
        k = np.ones((2 * abs(grow_px) + 1,) * 2, np.uint8)
        m = cv2.dilate(m, k) if grow_px > 0 else cv2.erode(m, k)
    return m.astype(bool)


def select_pixels(lab, region, select, l_max=70.0, c_min=8.0, l_line=40.0):
    """lab: [H x W x 3] の CIELAB。region の中から select の規則で画素を選ぶ (bool [H x W])。"""
    L = lab[..., 0]
    if select == "dark_chroma":
        return region & (L < l_max) & (np.hypot(lab[..., 1], lab[..., 2]) > c_min)
    if select == "interior":
        return region & (L > l_line)
    raise ValueError(f"unknown select {select!r}")


def median_ab(lab, sel, min_px=5):
    """選んだ画素の a*, b* の中央値。画素が min_px 未満なら None。"""
    return np.median(lab[sel][:, 1:], 0) if sel.sum() >= min_px else None


def recolor(rgb01, sel, target_ab, strength=1.0, feather_sigma=1.0):
    """rgb01: [H x W x 3] in [0, 1]。sel の画素の a*, b* を target_ab へ寄せた画像と、かけた重み [H x W] を返す。"""
    lab = rgb2lab(rgb01)
    w = sel.astype(np.float64)
    if feather_sigma > 0:
        w = gaussian_filter(w, feather_sigma)
    w = np.clip(w * strength, 0.0, 1.0)
    out = lab.copy()
    out[..., 1:] = lab[..., 1:] + w[..., None] * (np.asarray(target_ab, dtype=np.float64) - lab[..., 1:])
    return np.clip(lab2rgb(out), 0.0, 1.0), w
