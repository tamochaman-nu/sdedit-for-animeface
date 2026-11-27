"""
実写の顔を size 四方に切り出す (animeface/face_align.py)。方式 (--mode):
  ffhq  FFHQ の整列と同じ式 (目と口の位置から)。上下・左右を向くと目の間隔や目〜口が短く写るため、頭が大きく写り
        頭頂が切れることがある
  head  頭頂〜顎先の長さと位置を FFHQ (頭頂 0.037、顎先 0.906) に合わせる。アニメ顔の学習データの "head" 設定と
        同じ基準。頭頂は前景 (白背景の外) から測る。向きは --rotation、横の中心は --center_x
ランドマークは
scripts/detect_face_landmarks.py の JSON を使う。切り出した画像はそのまま scripts/sdedit_convert.py の
--base_samples に渡せる。貼り戻しは scripts/paste_back_faces.py。

出力 (--out_dir):
  <元の名前の拡張子を除いたもの>.png   切り出した画像 (size 四方)
  quads.json   {"args", "images": {<切り出した画像の名前>: {"source": 元の名前, "quad": [[x, y] x 4],
                "qsize": quad の一辺 (元画像の px), "out_of_frame": はみ出す面積の割合, "size": [w, h],
                "head_top", "chin": 切り出した画像での頭頂と顎先の高さ (一辺に対する割合),
                "ffhq_head_top", "ffhq_chin": ffhq 方式で切り出したときの同じ値}}}
<out_dir>_preview.jpg (--preview) 元画像に quad とランドマークを描いたものを並べた確認用の画像
  (--out_dir の中に置くと sdedit_convert.py の入力として拾われるため、外に置く)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch as th
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animeface.face_align import (  # noqa: E402
    FFHQ_CHIN,
    FFHQ_HEAD_TOP,
    ffhq_quad,
    fg_mask_np,
    head_position,
    head_quad,
    out_of_frame_ratio,
    quad_size,
    warp_quad,
)


def load_full(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1
    return th.from_numpy(a).permute(2, 0, 1)


def to_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--landmarks", required=True, help="scripts/detect_face_landmarks.py の JSON")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--supersample", type=int, default=4)
    p.add_argument("--mode", default="ffhq", choices=("ffhq", "head"))
    p.add_argument("--rotation", default="ffhq", choices=("ffhq", "none"), help="head 方式の向き")
    p.add_argument("--center_x", default="head", choices=("head", "face"), help="head 方式の横の中心")
    p.add_argument("--preview", type=int, default=12, help="preview.jpg に並べる枚数 (0 で作らない)")
    args = p.parse_args()

    with open(args.landmarks) as f:
        lms = json.load(f)["images"]
    os.makedirs(args.out_dir, exist_ok=True)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    res, prev = {}, []
    for name in sorted(lms):
        info = lms[name]
        if info is None:
            print(name, "顔なし: 飛ばす")
            continue
        img = load_full(os.path.join(args.input_dir, name)).to(device)
        fg = fg_mask_np(to_u8(img))
        q_ffhq = ffhq_quad(info["landmarks"])
        quad = q_ffhq if args.mode == "ffhq" else head_quad(info["landmarks"], fg, args.rotation, args.center_x)[0]
        pos, pos_f = head_position(info["landmarks"], fg, quad), head_position(info["landmarks"], fg, q_ffhq)
        h, w = img.shape[-2:]
        out = warp_quad(img, quad, args.size, args.supersample)
        stem = os.path.splitext(name)[0]
        Image.fromarray(to_u8(out)).save(os.path.join(args.out_dir, stem + ".png"))
        res[stem + ".png"] = {"source": name, "quad": quad.tolist(), "qsize": quad_size(quad),
                              "out_of_frame": out_of_frame_ratio(quad, w, h), "size": [w, h],
                              **pos, "ffhq_head_top": pos_f["head_top"], "ffhq_chin": pos_f["chin"]}
        if len(prev) < args.preview:
            im = Image.fromarray(to_u8(img.cpu()))
            dr = ImageDraw.Draw(im)
            dr.line([tuple(v) for v in quad] + [tuple(quad[0])], fill=(255, 0, 0), width=4)
            dr.line([tuple(quad[0]), tuple(quad[3])], fill=(0, 160, 255), width=8)  # 上辺
            for x, y in info["landmarks"]:
                dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 200, 0))
            prev.append((im, Image.fromarray(to_u8(out.cpu()))))
    with open(os.path.join(args.out_dir, "quads.json"), "w") as f:
        json.dump({"args": vars(args), "images": res}, f, indent=1)
    qs = np.array([r["qsize"] for r in res.values()])
    oof = np.array([r["out_of_frame"] for r in res.values()])
    print(f"{len(res)} 枚 -> {args.out_dir}  quad の一辺 {qs.min():.0f}〜{qs.max():.0f}px (中央値 {np.median(qs):.0f})、"
          f"はみ出し 最大 {oof.max():.3f} (0 より大きいもの {int((oof > 0).sum())} 枚)")
    for k, ref in (("head_top", FFHQ_HEAD_TOP), ("chin", FFHQ_CHIN)):
        v = np.array([r[k] for r in res.values()])
        vf = np.array([r["ffhq_" + k] for r in res.values()])
        print(f"  {k}: この方式 中央値 {np.median(v):.3f} [{v.min():.3f}, {v.max():.3f}]、"
              f"ffhq 方式 中央値 {np.median(vf):.3f} [{vf.min():.3f}, {vf.max():.3f}] (FFHQ・学習データ {ref})")
    if prev:
        th_h = 384
        tiles = []
        for im, al in prev:
            r = th_h / im.height
            a = im.resize((int(im.width * r), th_h))
            t = Image.new("RGB", (a.width + th_h, th_h), "white")
            t.paste(a, (0, 0))
            t.paste(al.resize((th_h, th_h)), (a.width, 0))
            tiles.append(t)
        cols = 4
        tw = max(t.width for t in tiles)
        g = Image.new("RGB", (tw * cols, th_h * ((len(tiles) + cols - 1) // cols)), "white")
        for k, t in enumerate(tiles):
            g.paste(t, ((k % cols) * tw, (k // cols) * th_h))
        g.save(args.out_dir.rstrip("/") + "_preview.jpg", quality=90)


if __name__ == "__main__":
    main()
