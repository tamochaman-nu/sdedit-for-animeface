"""
変換の出力 (scripts/sdedit_convert.py の --out_dir。整列後の 256px) の目の領域だけを、変換の入力 (--input_dir。
SDEdit の開始点にした切り出し) に戻す (貼り戻しの前の後処理)。3DGS への反復の焼き付けで、ビューごとに位置と形が
ばらばらなアニメの目を目標に入れないために使う (目の領域では目標が今の描画と同じになり、焼き付けても変わらない)。

目の領域 (和集合):
  入力の目  --parse_dir/<名前>.png のラベルが --parse_classes (既定 4 l_eye, 5 r_eye, 6 eye_g) の画素を
            --parse_dilate 画素だけ膨らませたもの
  出力の目  scripts/detect_anime_eyes.py の json (--eyes_json。出力にかけたもの) の 6 点の凸包を --grow_px 画素だけ
            膨らませたもの (animeface/anime_eyes.py の eye_region)。信頼度が --min_score 未満の目は使わない
  追加の部位 --extra_classes (例 10 nose, 11 mouth, 12 u_lip, 13 l_lip) のラベルの画素を --extra_dilate 画素だけ
            膨らませたもの (出力側の位置は検出しないので、入力の位置を大きめに膨らませる。既定は無し)
  出力の目が 1 つも見つからない画像は、入力の目を --fallback_dilate 画素だけ膨らませたものを使う
  (アニメの目は入力の目より下に大きく描かれやすいので、見つからなくても入力の目の周りだけでは足りないことがある)
重み = 領域を --feather_sigma のガウスでぼかしたもの。出力 = 変換の出力 × (1 - 重み) + 入力' × 重み
入力' (--color_match mean、既定): 入力の CIELAB に、領域の外側の輪 (幅 --ring_px、入力のラベルの肌 1 の画素) での
  「出力の平均 − 入力の平均」を足したもの (目の周りに写真の肌の色がまだらに残らないようにする)。none ならそのまま

出力 (--out_dir): 実行のディレクトリと同じ並び (そのまま scripts/paste_back_faces.py の --run_dir に使える)
  samples/<名前>.png   目の領域を入力に戻した出力
  weights/<名前>.png   かけた重み (0-255)
  refs/, refs_cond/    入力の実行からの複製 (あれば)
  config.json          入力の実行の config.json に "eye_keep" (このスクリプトの設定と、画像ごとの出力の目の数・
                       重みの面積) を足したもの

  docker compose run --rm --entrypoint python shell scripts/eye_keep_input.py --run_dir /data/output/convert \\
      --input_dir /data/output/aligned --eyes_json /data/output/convert/anime_eyes.json --out_dir /data/output/eyekeep
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import lab2rgb, rgb2lab

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animeface.anime_eyes import eye_region  # noqa: E402


def dilate(m, px):
    if px <= 0:
        return m
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * px + 1,) * 2, np.uint8)).astype(bool)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="変換の実行のディレクトリ (config.json, samples/)")
    p.add_argument("--input_dir", required=True, help="変換の入力の切り出し (<config の input の名前>)")
    p.add_argument("--eyes_json", required=True, help="scripts/detect_anime_eyes.py を変換の出力にかけた結果")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--parse_dir", default="/data/output/parse", help="入力の目のラベル (scripts/parse_faces.py)")
    p.add_argument("--parse_classes", default="4,5,6")
    p.add_argument("--parse_dilate", type=int, default=4)
    p.add_argument("--extra_classes", default="", help="目のほかに戻す部位のラベル (カンマ区切り)")
    p.add_argument("--extra_dilate", type=int, default=8)
    p.add_argument("--min_score", type=float, default=0.3)
    p.add_argument("--grow_px", type=int, default=4)
    p.add_argument("--fallback_dilate", type=int, default=12)
    p.add_argument("--feather_sigma", type=float, default=2.0)
    p.add_argument("--color_match", default="mean", choices=("mean", "none"))
    p.add_argument("--ring_px", type=int, default=6)
    p.add_argument("--skin_class", type=int, default=1)
    p.add_argument("--min_ring_px", type=int, default=20, help="輪の画素がこれ未満なら色を合わせない")
    a = p.parse_args()
    if os.path.abspath(a.out_dir) == os.path.abspath(a.run_dir):
        raise SystemExit("--out_dir は --run_dir と別にする")

    with open(os.path.join(a.run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(a.eyes_json) as f:
        eyes = json.load(f)["images"]
    classes = [int(c) for c in a.parse_classes.split(",")]
    extra = [int(c) for c in a.extra_classes.split(",") if c]
    for sub in ("samples", "weights"):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)
    for sub in ("refs", "refs_cond"):
        src = os.path.join(a.run_dir, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(a.out_dir, sub), dirs_exist_ok=True)

    per_image = {}
    for o in cfg["outputs"]:
        stem = os.path.splitext(o["input"])[0]
        out = np.asarray(Image.open(os.path.join(a.run_dir, o["sample"])).convert("RGB"), dtype=np.float64)
        inp = np.asarray(Image.open(os.path.join(a.input_dir, o["input"])).convert("RGB"), dtype=np.float64)
        if inp.shape != out.shape:
            raise SystemExit(f"{stem}: 入力 {inp.shape} と出力 {out.shape} の大きさが違う")
        lab = np.asarray(Image.open(os.path.join(a.parse_dir, stem + ".png")))
        if lab.shape != out.shape[:2]:
            lab = cv2.resize(lab, out.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        photo = np.isin(lab, classes)
        entry = eyes.get(stem)
        n_eyes = sum(e["score"] >= a.min_score for e in entry["eyes"]) if entry else 0
        if n_eyes:
            region = dilate(photo, a.parse_dilate) | eye_region(entry, out.shape[:2], a.min_score, a.grow_px)
        else:
            region = dilate(photo, a.fallback_dilate)
        if extra:
            region |= dilate(np.isin(lab, extra), a.extra_dilate)
        shift = None
        if a.color_match == "mean":
            ring = dilate(region, a.ring_px) & ~region & (lab == a.skin_class)
            if ring.sum() >= a.min_ring_px:
                li, lo = rgb2lab(inp / 255), rgb2lab(out / 255)
                shift = lo[ring].mean(0) - li[ring].mean(0)
                inp = np.clip(lab2rgb(li + shift), 0, 1) * 255
        w = region.astype(np.float64)
        if a.feather_sigma > 0:
            w = np.clip(gaussian_filter(w, a.feather_sigma), 0, 1)
        res = out * (1 - w[..., None]) + inp * w[..., None]
        Image.fromarray(np.round(res).astype(np.uint8)).save(os.path.join(a.out_dir, o["sample"]))
        Image.fromarray(np.round(w * 255).astype(np.uint8)).save(os.path.join(a.out_dir, "weights", stem + ".png"))
        per_image[stem] = {"output_eyes": int(n_eyes), "weight_area": float(w.mean()),
                           "lab_shift": None if shift is None else [float(v) for v in shift]}
    cfg["eye_keep"] = {"args": vars(a), "source_run": a.run_dir, "per_image": per_image,
                       "num_without_output_eyes": sum(v["output_eyes"] == 0 for v in per_image.values())}
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"{len(per_image)} 枚 (出力の目が見つからない {cfg['eye_keep']['num_without_output_eyes']} 枚) -> {a.out_dir}")


if __name__ == "__main__":
    main()
