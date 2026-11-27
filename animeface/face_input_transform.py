"""
④ 入力の加工: 切り出した実写の顔を、顔パーシングの部位ごとの平均色で塗ったストローク画にして (parts)、
前景の色をアニメ顔のデータの分布に写す (color_transfer.ColorTransfer)。背景 (前景マスクの外) は元のまま。

  parts  顔パーシングのラベル (scripts/parse_faces.py) の部位ごとに、その部位の平均色で塗る。
         鼻・耳・眼鏡は肌に、眉は髪の色に、目は髪の色 (暗い色) で塗る。shade_levels >= 2 なら、肌・首・髪・服を
         ぼかした L* の分位点で shade_levels 段に分け、段ごとの平均色で塗る (アニメのセル塗りの影)。
         eye_scale_w / eye_scale_h で目の領域を (目ごとの重心を中心に) 横・縦に広げる。

ilvr-for-anime-faceparts (87152b3、未追跡の guided_diffusion/face_input_transform.py、
sha256 de522b8093f63e9f211927229055e002027a4d276fc766fae398fe3984d0ca09) からの書き写し:
  FG_THRESH 28、fg_mask 31-36、to_lab 39-42、from_lab 45-47、PART_GROUPS / SHADED_PARTS 98-109、
  _scale_region 157-167、stroke_parts 185-230、transform_input 233-267
写さなかったもの: 抽象化 (双方向フィルタ、L* の段階化、XDoG)、汎用のストローク (SLIC)、目の塗り方 "iris" (瞳の色の固定)。
stroke_parts と transform_input は、それらの分岐を除いた (残した経路の計算は同じ)。

出力のメモリの並び: from_lab と ColorTransfer は numpy の HWC を permute して返すので、出力は channels_last の並びになる。
⑤ のモデルにはこの並びのまま渡す (並べ替えると cuDNN のカーネルが変わり、元の実行と 1 階調ほどずれる)。
"""

import numpy as np
import torch as th
import torch.nn.functional as F
from skimage.color import lab2rgb, rgb2lab

FG_THRESH = 0.94  # [0, 1] でこれより暗いチャネルがある画素を前景とする (背景除去済みの白背景)


def fg_mask(x, dilate=2):
    """x: [N x 3 x H x W] in [-1, 1] -> [N x 1 x H x W] (0/1 の float)"""
    m = (((x + 1) / 2).min(1, keepdim=True).values < FG_THRESH).float()
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    return m


def to_lab(x):
    """[N x 3 x H x W] in [-1, 1] -> torch [N x 3 x H x W] (CIELAB, float64)"""
    lab = rgb2lab(((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).double().cpu().numpy())
    return th.from_numpy(lab).permute(0, 3, 1, 2)


def from_lab(lab):
    rgb = np.stack([lab2rgb(o) for o in lab.permute(0, 2, 3, 1).numpy()])
    return th.from_numpy(rgb).float().permute(0, 3, 1, 2) * 2 - 1


# parts の部位 (CelebAMask-HQ の 19 クラス。scripts/parse_faces.py の docstring)。塗る順 (後のものが上)
PART_GROUPS = (
    ("other", (0,)),          # 前景だがパーシングで背景になった画素
    ("cloth", (15, 16)),
    ("neck", (14,)),
    ("skin", (1, 6, 7, 8, 9, 10)),
    ("mouth", (11, 12, 13)),
    ("eye", (4, 5)),
    ("brow", (2, 3)),
    ("hair", (17, 18)),
)
SHADED_PARTS = ("cloth", "neck", "skin", "hair")


def _scale_region(mask, sw, sh):
    """bool の領域 [H x W] を重心を中心に横 sw 倍・縦 sh 倍に広げる (逆写像の最近傍)"""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0 or (sw == 1 and sh == 1):
        return mask
    cy, cx = ys.mean(), xs.mean()
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sy = np.clip(np.rint(cy + (yy - cy) / sh).astype(int), 0, h - 1)
    sx = np.clip(np.rint(cx + (xx - cx) / sw).astype(int), 0, w - 1)
    return mask[sy, sx]


def stroke_parts(lab, fg, labels, shade_levels=1, eye_scale_w=1.0, eye_scale_h=1.0, shade_sigma=3.0):
    """
    lab: [N x 3 x H x W] (CIELAB)、fg: [N x 1 x H x W]、labels: [N x H x W] (0〜18 の整数)
    -> 前景を部位ごとの平均色で塗った lab (背景はそのまま)
    """
    from scipy.ndimage import gaussian_filter

    out = lab.clone()
    for k in range(lab.shape[0]):
        L = lab[k].permute(1, 2, 0).numpy()
        m = fg[k, 0].numpy() > 0.5
        lb = labels[k].numpy()
        regions = {name: np.isin(lb, ids) & m for name, ids in PART_GROUPS}
        eyes = [_scale_region((lb == c) & m, eye_scale_w, eye_scale_h) & m for c in (4, 5)]
        regions["eye"] = eyes[0] | eyes[1]
        means = {n: L[r].mean(0) for n, r in regions.items() if r.any()}
        dark = means.get("hair", np.array([20.0, 0.0, 0.0]))
        means["brow"] = dark
        means["eye"] = dark
        # 段の分け方を決める L* は、部位の中だけで正規化してぼかす (隣の部位の明るさが混ざらないように)
        o = L.copy()
        for name, _ in PART_GROUPS:
            r = regions[name]
            if not r.any():
                continue
            if shade_levels >= 2 and name in SHADED_PARTS and r.sum() >= 4 * shade_levels:
                rf = r.astype(np.float64)
                num = gaussian_filter(L[..., 0] * rf, shade_sigma)
                den = gaussian_filter(rf, shade_sigma)
                ls = num / np.maximum(den, 1e-6)
                qs = np.quantile(ls[r], np.linspace(0, 1, shade_levels + 1)[1:-1])
                band = np.digitize(ls, qs)
                for b in range(shade_levels):
                    rb = r & (band == b)
                    if rb.any():
                        o[rb] = L[rb].mean(0)
            else:
                o[r] = means[name]
        out[k] = th.from_numpy(o).permute(2, 0, 1)
    return out


def transform_input(x, parse, color_transfer=None, stroke_shade_levels=1, stroke_eye_scale_w=1.0,
                    stroke_eye_scale_h=1.0):
    """
    x: [N x 3 x H x W] in [-1, 1]、parse: パーシングのラベル [N x H x W] -> 加工した入力 (CPU、channels_last の並び)。
    部位別のストローク化 (stroke_parts) -> 前景だけを置き換え -> 色の写像 (color_transfer。ColorTransfer か None)。
    元の transform_input で stroke="parts"、抽象化なしにしたときと同じ計算・同じ順序。
    """
    assert parse is not None, "部位別のストローク化にはパーシングのラベルが要る"
    m = fg_mask(x).cpu()
    y = x.cpu()
    lab = to_lab(y)
    lab = stroke_parts(lab, m, parse.cpu(), stroke_shade_levels, stroke_eye_scale_w, stroke_eye_scale_h)
    y = m * from_lab(lab) + (1 - m) * y
    if color_transfer is not None:
        y = color_transfer(y, m)  # 前景だけに適用 (m * 写像 + (1 - m) * y)
    return y
