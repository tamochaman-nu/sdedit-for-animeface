"""
実写の顔の FFHQ 方式の整列 (切り出し) と、変換後の画像を元のフレームへ貼り戻す逆変換。

アニメ顔の学習データ (~/anime-face-alignment の既定の "head" 設定) は、頭頂〜顎先の大きさと顎先の位置を
FFHQ の整列済み画像に合わせて切り出してある (util/anime_face_alignment.py の ALIGN_PRESETS の注記)。
そのため実写の顔は、FFHQ の整列 (ffhq-dataset の download_ffhq.py の recreate_aligned_images) と同じ式で
切り出せば、学習データの構図にそのまま対応する。

quad: 出力画像の外縁の 4 隅に対応する元画像上の四角形 [左上, 左下, 右下, 右上] (ピクセル中心基準の座標)。
変換は ~/anime-face-alignment/util/quad_warp.py の warp_quad と同じ幾何 (出力画素 (j, i) の中心は
左上 + (j + 0.5) u + (i + 0.5) v) を torch の grid_sample で行う。違い:
  - quad が元画像の外にはみ出す部分は、反射パディングではなく白で埋める (入力は背景除去済みで白背景のため)
  - 先に整数分の 1 に縮小する代わりに、supersample 倍の解像度で双線形にサンプリングしてから平均で縮小する
"""

import numpy as np
import torch as th
import torch.nn.functional as F

# 68 点 (iBUG 300-W) の番号
LEFT_EYE = slice(36, 42)
RIGHT_EYE = slice(42, 48)
MOUTH_LEFT, MOUTH_RIGHT = 48, 54


def ffhq_quad(landmarks):
    """68 点のランドマーク [68 x 2] から FFHQ の quad [4 x 2] を求める (recreate_aligned_images と同じ式)。"""
    lm = np.asarray(landmarks, dtype=np.float64)[:, :2]
    eye_left = lm[LEFT_EYE].mean(0)
    eye_right = lm[RIGHT_EYE].mean(0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm[MOUTH_LEFT] + lm[MOUTH_RIGHT]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y])


# FFHQ の整列済み画像での頭頂と顎先の高さ (出力の一辺に対する割合)。~/anime-face-alignment の
# util/anime_face_alignment.py の ALIGN_PRESETS の注記 ("head" 設定の較正: 頭頂の 85 パーセンタイル 0.037、
# 顎先の中央値 0.906)。アニメ顔の学習データは頭頂〜顎先をこれに合わせて切り出してある
FFHQ_HEAD_TOP = 0.037
FFHQ_CHIN = 0.906
CHIN = 8  # 68 点の顎先


def fg_mask_np(img_u8, thresh=0.94):
    """背景除去済み (白背景) の画像 [H x W x 3] (uint8) の前景 (いずれかのチャネルが thresh より暗い画素)"""
    return (img_u8.min(-1) / 255.0) < thresh


def _axes(quad):
    q = np.asarray(quad, dtype=np.float64)
    ux = (q[3] - q[0]) / np.hypot(*(q[3] - q[0]))
    uy = (q[1] - q[0]) / np.hypot(*(q[1] - q[0]))
    return q.mean(0), ux, uy


def head_extent(landmarks, fg, quad, min_count=0.02, step=2):
    """
    quad の向きの座標 (中心 = quad の中心、横 = 上辺の向き、縦 = 左辺の向き) で、頭頂の高さ・顎先の高さ・
    目の高さより上の前景の横の範囲 (1, 99 パーセンタイル) を測る。単位は元画像の px。
    頭頂: 顔の中心から横に ±quad の一辺の範囲で、前景の画素が (その一辺の min_count 倍の数以上) ある最も上の行
    (髪の毛先の飛び出しを無視するため)。fg は [H x W] の bool。step px おきに数える。
    """
    lm = np.asarray(landmarks, dtype=np.float64)[:, :2]
    c, ux, uy = _axes(quad)
    L = quad_size(quad)
    ys, xs = np.nonzero(fg[::step, ::step])
    P = np.stack([xs, ys], 1).astype(np.float64) * step - c
    px, py = P @ ux, P @ uy
    eye_y = ((lm[36:48].mean(0) - c) @ uy)
    chin_y = (lm[CHIN] - c) @ uy
    band = (np.abs(px) < L) & (py < chin_y)
    rows = np.floor(py[band] / step).astype(int)
    cnt = np.bincount(rows - rows.min())
    need = max(1, int(min_count * L / step))
    top_y = (np.nonzero(cnt >= need)[0][0] + rows.min()) * step
    upper = band & (py < eye_y)
    lo, hi = np.percentile(px[upper], [1, 99]) if upper.sum() > 10 else (-L / 2, L / 2)
    return {"top_y": float(top_y), "chin_y": float(chin_y), "eye_y": float(eye_y), "x_lo": float(lo),
            "x_hi": float(hi), "side": L}


def head_quad(landmarks, fg, rotation="ffhq", center_x="head"):
    """
    頭頂〜顎先の長さと位置を FFHQ (= アニメ顔の学習データの "head" 設定) に合わせた quad。
    rotation: "ffhq" (FFHQ の quad と同じ向き) か "none" (回転なし、画像の上下のまま)。
    center_x: "head" (目より上の前景の横の範囲の中央。横顔で後頭部も入る) か "face" (FFHQ の quad の中心)。
    """
    q = ffhq_quad(landmarks)
    if rotation == "none":
        c0 = q.mean(0)
        h = quad_size(q) / 2
        q = np.stack([c0 + [-h, -h], c0 + [-h, h], c0 + [h, h], c0 + [h, -h]])
    else:
        assert rotation == "ffhq", rotation
    e = head_extent(landmarks, fg, q)
    c, ux, uy = _axes(q)
    side = (e["chin_y"] - e["top_y"]) / (FFHQ_CHIN - FFHQ_HEAD_TOP)
    cy = e["top_y"] - FFHQ_HEAD_TOP * side + side / 2
    cx = (e["x_lo"] + e["x_hi"]) / 2 if center_x == "head" else 0.0
    cen = c + cx * ux + cy * uy
    hx, hy = ux * side / 2, uy * side / 2
    return np.stack([cen - hx - hy, cen - hx + hy, cen + hx + hy, cen + hx - hy]), e


def head_position(landmarks, fg, quad):
    """quad で切り出したときの頭頂と顎先の高さ (一辺に対する割合、上端 = 0)"""
    e = head_extent(landmarks, fg, quad)
    return {"head_top": e["top_y"] / e["side"] + 0.5, "chin": e["chin_y"] / e["side"] + 0.5}


def quad_size(quad):
    return float(np.hypot(*(np.asarray(quad)[3] - np.asarray(quad)[0])))


def out_of_frame_ratio(quad, w, h, n=64):
    """quad のうち元画像 [-0.5, w-0.5] x [-0.5, h-0.5] の外にはみ出す面積の割合 (n x n 点で数える)。"""
    q = np.asarray(quad, dtype=np.float64)
    a = (np.arange(n) + 0.5) / n
    aa, bb = np.meshgrid(a, a, indexing="xy")
    p = q[0] + aa[..., None] * (q[3] - q[0]) + bb[..., None] * (q[1] - q[0])
    inside = (p[..., 0] >= -0.5) & (p[..., 0] <= w - 0.5) & (p[..., 1] >= -0.5) & (p[..., 1] <= h - 0.5)
    return float(1.0 - inside.mean())


def _to_grid(px, py, w, h):
    """ピクセル中心基準の座標 -> grid_sample (align_corners=False) の [-1, 1] 座標"""
    return th.stack([(px + 0.5) / w * 2 - 1, (py + 0.5) / h * 2 - 1], -1)


def warp_quad(img, quad, size=256, supersample=4, fill=1.0):
    """
    img [3 x H x W] (float, [-1, 1]) の quad の範囲を size 四方へ変換する。はみ出す部分は fill (既定は白) で埋める。
    """
    q = th.as_tensor(np.asarray(quad, dtype=np.float64))
    s = size * supersample
    u = (q[3] - q[0]) / s
    v = (q[1] - q[0]) / s
    jj, ii = th.meshgrid(th.arange(s, dtype=th.float64) + 0.5, th.arange(s, dtype=th.float64) + 0.5, indexing="xy")
    p = q[0] + jj[..., None] * u + ii[..., None] * v
    h, w = img.shape[-2:]
    grid = _to_grid(p[..., 0], p[..., 1], w, h).to(img.device, th.float32)[None]
    out = F.grid_sample((img - fill)[None].float(), grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False) + fill
    if supersample > 1:
        out = F.avg_pool2d(out, supersample)
    return out[0]


def paste_back(frame, aligned, quad, feather=0.08, mask=None):
    """
    aligned [3 x s x s] を quad の逆変換で frame [3 x H x W] (どちらも [-1, 1]) へ貼り戻す。
    重みは quad の内側で 1、外縁から一辺の feather 倍の幅で 0 へ滑らかに下がる (smoothstep)。
    mask [1 x s x s] ([0, 1]) を渡すと重みに掛ける (整列後の座標の、貼り戻す領域)。
    戻り値: (貼り戻したフレーム, 重み [1 x H x W])
    """
    q = np.asarray(quad, dtype=np.float64)
    h, w = frame.shape[-2:]
    s = aligned.shape[-1]
    x0, y0 = np.floor(q.min(0)).astype(int)
    x1, y1 = np.ceil(q.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1 + 1, w), min(y1 + 1, h)
    out = frame.clone()
    weight = th.zeros(1, h, w, device=frame.device)
    if x0 >= x1 or y0 >= y1:
        return out, weight
    # 元画像の点 p = q0 + a (q3 - q0) + b (q1 - q0) となる (a, b) ∈ [0, 1]^2 を解く
    e1, e2 = q[3] - q[0], q[1] - q[0]
    inv = np.linalg.inv(np.stack([e1, e2], 1))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    d = np.stack([xx - q[0, 0], yy - q[0, 1]], -1)
    ab = d @ inv.T
    a, b = ab[..., 0], ab[..., 1]
    # 整列後の画像のピクセル中心基準の座標 = a * s - 0.5
    grid = _to_grid(th.from_numpy(a * s - 0.5), th.from_numpy(b * s - 0.5), s, s).to(frame.device, th.float32)[None]
    src = aligned[None].float()
    if mask is not None:
        src = th.cat([src, mask[None].float()], 1)
    samp = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    edge = np.minimum(np.minimum(a, 1 - a), np.minimum(b, 1 - b))
    t = np.clip(edge / feather, 0, 1) if feather > 0 else (edge >= 0).astype(np.float64)
    wgt = th.from_numpy(t * t * (3 - 2 * t)).to(frame.device, th.float32)[None]
    if mask is not None:
        wgt = wgt * samp[3:4]
        samp = samp[:3]
    region = out[:, y0:y1, x0:x1]
    out[:, y0:y1, x0:x1] = wgt * samp + (1 - wgt) * region
    weight[:, y0:y1, x0:x1] = wgt
    return out, weight
