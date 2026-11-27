"""
変換の出力 (scripts/sdedit_convert.py の --out_dir。整列後の 256px) の瞳の色を、全ビューで同じ目標の色に寄せる
(貼り戻しの前の後処理)。目の位置は出力から求めたもの (scripts/detect_anime_eyes.py の json) を使う。
領域・画素の選び方・置き換えの式は animeface/anime_eyes.py。

目標の色 (--target_ab):
  auto   入力の実行の全ビューについて、選んだ画素の a*, b* の中央値を画像ごとに求め、そのビュー間の中央値
  a,b    指定した (a*, b*)
出力 (--out_dir): 実行のディレクトリと同じ並びにするので、そのまま scripts/paste_back_faces.py の --run_dir に使える
  samples/<名前>.png   色を置き換えた出力 (目が見つからない・信頼度の低い画像は元のまま)
  refs/, refs_cond/    入力の実行からの複製 (diagnostics/stroke_eval.py 用)
  weights/<名前>.png   かけた重み (0-255。確認用)
  config.json          入力の実行の config.json に "eye_recolor" (このスクリプトの設定、目標の色、画像ごとの
                       置き換えた画素の数と置き換え前の a*, b* の中央値) を足したもの
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animeface.anime_eyes import SELECT_CHOICES, eye_region, median_ab, recolor, select_pixels  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="変換の実行のディレクトリ (config.json, samples/)")
    p.add_argument("--eyes_json", required=True, help="scripts/detect_anime_eyes.py の出力")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--select", default="interior", choices=SELECT_CHOICES)
    p.add_argument("--target_ab", default="auto", help="auto か a,b")
    p.add_argument("--strength", type=float, default=1.0, help="目標の色へ寄せる割合 (0-1)")
    p.add_argument("--min_score", type=float, default=0.3, help="目の点の信頼度の平均がこれ未満の目は使わない")
    p.add_argument("--grow_px", type=int, default=-1, help="目の凸包を膨らませる画素数 (負なら削る)")
    p.add_argument("--l_max", type=float, default=70.0, help="dark_chroma: L* の上限")
    p.add_argument("--c_min", type=float, default=8.0, help="dark_chroma: 彩度 C* の下限")
    p.add_argument("--l_line", type=float, default=40.0, help="interior: これ以下の L* (輪郭の線) を除く")
    p.add_argument("--feather_sigma", type=float, default=1.0, help="重みのぼかし (画素)")
    p.add_argument("--min_px", type=int, default=5, help="選んだ画素がこれ未満の画像は置き換えない")
    a = p.parse_args()
    if os.path.abspath(a.out_dir) == os.path.abspath(a.run_dir):
        raise SystemExit("--out_dir は --run_dir と別にする")

    with open(os.path.join(a.run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(a.eyes_json) as f:
        eyes = json.load(f)["images"]

    items = []
    for o in cfg["outputs"]:
        stem = os.path.splitext(o["input"])[0]
        rgb = np.asarray(Image.open(os.path.join(a.run_dir, o["sample"])).convert("RGB"), dtype=np.float64) / 255
        lab = rgb2lab(rgb)
        sel = select_pixels(lab, eye_region(eyes.get(stem), rgb.shape[:2], a.min_score, a.grow_px), a.select,
                            a.l_max, a.c_min, a.l_line)
        items.append((o, stem, rgb, sel, median_ab(lab, sel, a.min_px)))

    meds = np.array([m for *_, m in items if m is not None])
    if a.target_ab == "auto":
        if not len(meds):
            raise SystemExit("目の画素のある画像が無いので auto の目標の色を決められない")
        target = np.median(meds, 0)
    else:
        target = np.array([float(v) for v in a.target_ab.split(",")])
        assert target.shape == (2,), a.target_ab

    for sub in ("samples", "weights"):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)
    for sub in ("refs", "refs_cond"):
        src = os.path.join(a.run_dir, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(a.out_dir, sub), dirs_exist_ok=True)
    per_image = {}
    for o, stem, rgb, sel, med in items:
        if med is None:
            out, w = rgb, np.zeros(rgb.shape[:2])
        else:
            out, w = recolor(rgb, sel, target, a.strength, a.feather_sigma)
        Image.fromarray(np.round(out * 255).astype(np.uint8)).save(os.path.join(a.out_dir, o["sample"]))
        Image.fromarray(np.round(w * 255).astype(np.uint8)).save(os.path.join(a.out_dir, "weights", stem + ".png"))
        per_image[stem] = {"selected_px": int(sel.sum()), "recolored": med is not None,
                           "ab_before": None if med is None else [float(med[0]), float(med[1])]}
    cfg["eye_recolor"] = {"args": vars(a), "source_run": a.run_dir, "target_ab": [float(t) for t in target],
                          "num_recolored": sum(v["recolored"] for v in per_image.values()), "per_image": per_image}
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"目標の色 (a*, b*) = ({target[0]:.2f}, {target[1]:.2f})、置き換えた画像 {cfg['eye_recolor']['num_recolored']} / "
          f"{len(items)} -> {a.out_dir}")


if __name__ == "__main__":
    main()
