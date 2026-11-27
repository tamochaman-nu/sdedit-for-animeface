"""
④ 色の写像 (ColorTransfer): 実写の顔の色の分布を、アニメ顔のデータの色の分布に写す。統計は lists/face_color_stats.json
(scripts/fit_face_color_stats.py)。写像はデータセット単位で固定 (画像ごとに統計を合わせない)。CIELAB のチャネルごとに:
  lab_meanstd   平均と標準偏差を合わせる一次変換 (Reinhard 型)
  lab_quantile  分位点を合わせる単調変換 (ヒストグラムの一致)
  lab_meanstd_l L* だけ平均と標準偏差を合わせ、a* b* は平均だけずらす (採用構成)
  lab_mean      全チャネルで平均だけずらす
  strength (0〜1) で元の色との間を CIELAB で線形に補間する。

ilvr-for-anime-faceparts (87152b3、guided_diffusion/ilvr_ref_transform.py、コミット済み、
sha256 49f3378810679c35f23118bc96b1f505a70bdaf830bf274483ca6e72aec0564d) からの書き写し:
  COLOR_TRANSFER_CHOICES / NUM_QUANTILES 30-31、to_lab_pixels 34-36、lab_stats 39-49、ColorTransfer 52-81
写さなかったもの: 目のマスクの読み込み (load_masks) と、それが使う ilvr_eye.image_path の import。
"""

import json

import numpy as np
import torch as th
from skimage.color import lab2rgb, rgb2lab

COLOR_TRANSFER_CHOICES = ("none", "lab_meanstd", "lab_quantile", "lab_meanstd_l", "lab_mean")
NUM_QUANTILES = 1001


def to_lab_pixels(x):
    """[N x 3 x H x W] in [-1, 1] -> numpy [N x H x W x 3] (CIELAB)"""
    return rgb2lab(((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).double().cpu().numpy())


def lab_stats(x, masks=None):
    """
    CIELAB のチャネルごとの平均・標準偏差・分位点 (画素単位でまとめる)。masks ([N x 1 x H x W]) を渡すと
    マスクが 0.5 以上の画素 (マスク済みの画像では白い背景を除いた目の部分) だけで求める。
    """
    lab = to_lab_pixels(x).reshape(-1, 3)
    if masks is not None:
        lab = lab[(masks[:, 0] >= 0.5).reshape(-1).numpy()]
    q = np.linspace(0, 1, NUM_QUANTILES)
    return {"mean": lab.mean(0).tolist(), "std": lab.std(0).tolist(),
            "quantiles": [np.quantile(lab[:, c], q).tolist() for c in range(3)]}


class ColorTransfer:
    def __init__(self, stats_path, method, strength=1.0, src="real", dst="anime"):
        assert method in COLOR_TRANSFER_CHOICES and method != "none", method
        with open(stats_path) as f:
            stats = json.load(f)
        self.src, self.dst = stats[src], stats[dst]
        self.method, self.strength = method, strength

    def __call__(self, x, masks=None):
        """
        [N x 3 x H x W] in [-1, 1] -> 同じ形 (CPU)。色域外は lab2rgb が [0, 1] に切り詰める。
        masks ([N x 1 x H x W]) を渡すと、写した色をマスクの部分にだけ使う (m * 写像 + (1 - m) * 元の画像。
        マスク済みの画像の白い背景は白のまま)。
        """
        lab = to_lab_pixels(x)
        out = np.empty_like(lab)
        for c in range(3):
            if self.method == "lab_quantile":
                out[..., c] = np.interp(lab[..., c], self.src["quantiles"][c], self.dst["quantiles"][c])
                continue
            # 一次変換: 平均を合わせ、標準偏差は方式によって合わせる (gain) か実写のまま (1)
            scale_std = self.method == "lab_meanstd" or (self.method == "lab_meanstd_l" and c == 0)
            gain = self.dst["std"][c] / self.src["std"][c] if scale_std else 1.0
            out[..., c] = (lab[..., c] - self.src["mean"][c]) * gain + self.dst["mean"][c]
        out = (1 - self.strength) * lab + self.strength * out
        rgb = np.stack([lab2rgb(o) for o in out])
        y = th.from_numpy(rgb).float().permute(0, 3, 1, 2) * 2 - 1
        if masks is not None:
            y = masks * y + (1 - masks) * x.cpu()
        return y
