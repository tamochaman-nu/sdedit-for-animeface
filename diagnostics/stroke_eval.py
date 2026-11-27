"""
変換の実行結果 (scripts/sdedit_convert.py の --out_dir。整列後の 256px) をまとめて評価し、比較の画像を作る。
元リポジトリ (ilvr-for-anime-faceparts) の diagnostics/stroke_eval.py の書き写し。指標の計算は同じ。

指標 (実行ごと、全入力の平均):
  anime        出力のアニメらしさ (diagnostics/anime_score.py のロジット。--score_path "" で出さない)
  anime_cond   モデルに渡した入力 (refs_cond、無ければ refs) のアニメらしさ
  lf16_lum     輝度を画像ごとに標準化し、16 画素の平均プーリング後の RMSE を元の入力 (refs) と比べたもの (構図のずれ)
  saturation, laplacian_var, luminance   style_stats (元の guided_diffusion/ilvr_metrics.py の style_stats と同じ)
  瞳の色のばらつき (--parse_dir があるとき): 目の領域 (パーシングの目 4, 5 を重心を中心に横 1.2・縦 1.8 倍に
  広げて 2px 膨らませたもの) のうち L* < --eye_l_max の画素の a*, b* の中央値を画像ごとに求め、
  その画素が --eye_min_px 以上ある画像について、ビュー間の標準偏差 (eye_a_std, eye_b_std) と
  平均の彩度 (eye_chroma)、対象の画像の数 (eye_n) を出す
  出力の目での瞳の色のばらつき (実行のディレクトリに anime_eyes.json (scripts/detect_anime_eyes.py) があるか、
  scripts/eye_recolor.py の出力で、その eyes_json が読めるとき): 出力から求めた目の領域
  (animeface/anime_eyes.py。既定の設定) で、画素の選び方 interior と dark_chroma のそれぞれについて、
  a*, b* の中央値のビュー間の標準偏差 (oeye_<選び方>_a_std, _b_std)、平均の彩度 (_chroma)、画像の数 (_n)
出力: <out>_metrics.json (実行ごとの指標と、--per_image なら画像ごとの瞳の a*, b*)、
      <out>_<名前>.jpg (--grid "名前=実行1,実行2,..." ごと。行 = 実行 (入力と出力の 2 行)、列 = --names)

  docker compose run --rm stroke-eval --runs SD=/data/output/convert --parse_dir /data/output/parse \
      --out /data/output/eval/sd
"""

import argparse
import json
import os
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animeface.anime_eyes import SELECT_CHOICES, eye_region, median_ab, select_pixels  # noqa: E402
from animeface.face_input_transform import _scale_region, to_lab  # noqa: E402
from animeface.image_io import load_image  # noqa: E402


def style_stats(x):
    """彩度 (HSV の S)、輝度のラプラシアンの分散、平均輝度をサンプルごとに返す (元の ilvr_metrics.style_stats)。"""
    x01 = (x.clamp(-1, 1) + 1) / 2
    lum = 0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]
    mx, mn = x01.max(1).values, x01.min(1).values
    sat = ((mx - mn) / mx.clamp(min=1e-6)).mean((1, 2))
    k = th.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device)[None, None]
    lap = F.conv2d(lum[:, None], k).flatten(1).var(1)
    return {"saturation": sat, "laplacian_var": lap, "luminance": lum.mean((1, 2))}


def lum_std(x):
    x = (x + 1) / 2
    lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return (lum - lum.mean((2, 3), keepdim=True)) / (lum.std((2, 3), keepdim=True) + 1e-6)


def eye_masks(parse_dir, stems, scale_w=1.2, scale_h=1.8, dilate=2):
    from scipy.ndimage import binary_dilation

    out = []
    for s in stems:
        lb = np.asarray(Image.open(os.path.join(parse_dir, s + ".png")))
        m = _scale_region(lb == 4, scale_w, scale_h) | _scale_region(lb == 5, scale_w, scale_h)
        out.append(binary_dilation(m, iterations=dilate) if dilate > 0 else m)
    return np.stack(out)


def eye_ab(y, em, l_max, min_px):
    """画像ごとの目の領域の暗い画素の a*, b* の中央値。画素が min_px 未満の画像は None。"""
    lab = to_lab(y).permute(0, 2, 3, 1).numpy()
    out = []
    for li, mi in zip(lab, em):
        sel = mi & (li[..., 0] < l_max)
        out.append(np.median(li[sel][:, 1:], 0) if sel.sum() >= min_px else None)
    return out


def find_eyes_json(run_dir, cfg):
    """実行の目の位置: <run_dir>/anime_eyes.json、無ければ eye_recolor.py の出力が使った eyes_json。"""
    p = os.path.join(run_dir, "anime_eyes.json")
    if os.path.exists(p):
        return p
    p = cfg.get("eye_recolor", {}).get("args", {}).get("eyes_json")
    return p if p and os.path.exists(p) else None


def output_eye_stats(y, stems, eyes_path):
    """出力から求めた目の領域での瞳の色のばらつき (画素の選び方ごと) と、画像ごとの a*, b* の中央値。"""
    with open(eyes_path) as f:
        eyes = json.load(f)["images"]
    lab = to_lab(y).permute(0, 2, 3, 1).numpy()
    r, per = {}, {}
    for sel_name in SELECT_CHOICES:
        meds = {}
        for s, li in zip(stems, lab):
            m = median_ab(li, select_pixels(li, eye_region(eyes.get(s), li.shape[:2]), sel_name))
            meds[s] = None if m is None else [float(m[0]), float(m[1])]
        ab = np.array([v for v in meds.values() if v is not None]).reshape(-1, 2)
        k = f"oeye_{sel_name}"
        r.update({f"{k}_n": len(ab), f"{k}_a_std": float(ab[:, 0].std()) if len(ab) else None,
                  f"{k}_b_std": float(ab[:, 1].std()) if len(ab) else None,
                  f"{k}_chroma": float(np.hypot(ab[:, 0], ab[:, 1]).mean()) if len(ab) else None})
        per[sel_name] = meds
    return r, per


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="名前=実行のディレクトリ をカンマ区切りで")
    p.add_argument("--out", required=True, help="出力の接頭辞 (例 /data/output/eval/t0)")
    p.add_argument("--parse_dir", default=None)
    p.add_argument("--names", default="Cam1_frame_00001,Cam1_frame_00008,Cam2_frame_00006,Cam3_frame_00005,"
                                      "Cam4_frame_00006,Cam5_frame_00006")
    p.add_argument("--grid", action="append", default=[], help="名前=実行の名前1,実行の名前2,... (複数回指定できる)")
    p.add_argument("--score_path", default="diagnostics/assets/anime_score.pt",
                   help="アニメらしさの分類器 (diagnostics/anime_score.py)。空なら anime と anime_cond を出さない")
    p.add_argument("--eye_l_max", type=float, default=70.0)
    p.add_argument("--eye_min_px", type=int, default=5)
    p.add_argument("--per_image", action="store_true", help="画像ごとの瞳の a*, b* も json に書く")
    args = p.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    scorer = None
    if args.score_path:
        from diagnostics.anime_score import AnimeScore
        scorer = AnimeScore(args.score_path, device)
    runs = dict(r.split("=", 1) for r in args.runs.split(","))
    res, paths, per_image = {}, {}, {}
    for name, d in runs.items():
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        outs = cfg["outputs"]
        stems = [os.path.splitext(o["input"])[0] for o in outs]
        y = th.stack([load_image(os.path.join(d, o["sample"]), 256) for o in outs])
        x = th.stack([load_image(os.path.join(d, o["ref_png"]), 256) for o in outs])
        c = th.stack([load_image(os.path.join(d, o.get("cond_png", o["ref_png"])), 256) for o in outs])
        yd = y.to(device)
        st = style_stats(yd)
        r = {}
        if scorer is not None:
            r.update(anime=scorer.logit(yd).mean().item(), anime_cond=scorer.logit(c.to(device)).mean().item())
        r.update(lf16_lum=((F.avg_pool2d(lum_std(y), 16) - F.avg_pool2d(lum_std(x), 16)) ** 2)
                 .flatten(1).mean(1).sqrt().mean().item(),
                 **{k: float(v.mean()) for k, v in st.items()})
        if args.parse_dir:
            per = eye_ab(y, eye_masks(args.parse_dir, stems), args.eye_l_max, args.eye_min_px)
            ab = np.array([v for v in per if v is not None]).reshape(-1, 2)
            r.update(eye_n=len(ab), eye_a_std=float(ab[:, 0].std()) if len(ab) else None,
                     eye_b_std=float(ab[:, 1].std()) if len(ab) else None,
                     eye_chroma=float(np.hypot(ab[:, 0], ab[:, 1]).mean()) if len(ab) else None)
            if args.per_image:
                per_image[name] = {s: (None if v is None else [float(v[0]), float(v[1])]) for s, v in zip(stems, per)}
        eyes_path = find_eyes_json(d, cfg)
        if eyes_path:
            oe, oper = output_eye_stats(y, stems, eyes_path)
            r.update(oe)
            if args.per_image:
                per_image.setdefault(name + "_oeye", oper)
        res[name] = r
        paths[name] = {s: (os.path.join(d, o.get("cond_png", o["ref_png"])), os.path.join(d, o["sample"]))
                       for s, o in zip(stems, outs)}
        print(f"{name:22s}", " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()),
              flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {"runs": runs, "metrics": res}
    if per_image:
        out["eye_ab_per_image"] = per_image
    with open(args.out + "_metrics.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    names = args.names.split(",")
    for gspec in args.grid:
        gname, members = gspec.split("=", 1)
        members = members.split(",")
        g = Image.new("RGB", (258 * len(names), 258 * 2 * len(members)), "white")
        for i, m in enumerate(members):
            for j, n in enumerate(names):
                ci, yi = paths[m][n]
                g.paste(Image.open(ci).convert("RGB"), (258 * j, 258 * 2 * i))
                g.paste(Image.open(yi).convert("RGB"), (258 * j, 258 * (2 * i + 1)))
        g.resize((g.width * 3 // 4, g.height * 3 // 4)).save(f"{args.out}_{gname}.jpg", quality=90)


if __name__ == "__main__":
    main()
