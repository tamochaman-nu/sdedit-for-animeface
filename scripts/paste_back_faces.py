"""
scripts/sdedit_convert.py の出力 (整列した顔の変換結果) を、scripts/align_faces.py の quad の逆変換で
元のフレームへ貼り戻す (animeface/face_align.paste_back)。

貼り戻す重みは quad の内側で 1、外縁から一辺の --feather 倍の幅で 0 へ滑らかに下がる。
--mask_parse_dir (scripts/parse_faces.py のラベル) を渡すと、重みに部位のマスクを掛ける: --mask_classes の部位
(既定は頭部 = 肌・眉・目・眼鏡・耳・鼻・口・唇・首・髪・帽子。服は含めない) を --mask_dilate px 膨らませ、
--mask_feather px のガウシアンでぼかしたもの (整列後の座標で作り、quad の逆変換で元のフレームへ写す)。
出力 (--out_dir、既定は <run_dir>/pasted):
  <元の名前の拡張子を除いたもの>.png   貼り戻したフレーム (元の解像度)
  grid.jpg   (--grid) 元のフレームと貼り戻したフレームを並べた確認用の画像
"""

import argparse
import json
import os
import sys

import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animeface.face_align import paste_back  # noqa: E402


def load(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1
    return th.from_numpy(a).permute(2, 0, 1)


def to_img(x):
    return Image.fromarray(((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())


def part_mask(path, classes, dilate, feather):
    """ラベル画像から部位のマスク [1 x s x s] ([0, 1]) を作る"""
    from scipy.ndimage import binary_dilation, gaussian_filter

    m = np.isin(np.asarray(Image.open(path)), classes)
    if dilate > 0:
        m = binary_dilation(m, iterations=dilate)
    m = m.astype(np.float32)
    if feather > 0:
        m = gaussian_filter(m, feather, mode="nearest")
    return th.from_numpy(m)[None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="scripts/sdedit_convert.py の --out_dir")
    p.add_argument("--frames_dir", required=True, help="元のフレームのディレクトリ")
    p.add_argument("--quads", required=True, help="scripts/align_faces.py の quads.json")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--feather", type=float, default=0.08)
    p.add_argument("--mask_parse_dir", default=None, help="部位のマスクに使うパーシングのラベル (無ければ quad 全体)")
    p.add_argument("--mask_classes", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,17,18")
    p.add_argument("--mask_dilate", type=int, default=4, help="部位のマスクを膨らませる px (整列後の 256px で)")
    p.add_argument("--mask_feather", type=float, default=4.0, help="部位のマスクのぼかしの sigma px (整列後)")
    p.add_argument("--grid", default="", help="grid.jpg に並べる元の名前 (カンマ区切り。空なら先頭から 6 枚)")
    p.add_argument("--grid_crop", type=float, default=1.6, help="grid.jpg で切り出す範囲 (quad の一辺の倍率)")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "pasted")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "config.json")) as f:
        outputs = json.load(f)["outputs"]
    with open(args.quads) as f:
        quads = json.load(f)["images"]
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    classes = [int(c) for c in args.mask_classes.split(",")]
    done = {}
    for o in outputs:
        q = quads[o["input"]]
        frame = load(os.path.join(args.frames_dir, q["source"])).to(device)
        aligned = load(os.path.join(args.run_dir, o["sample"])).to(device)
        mask = None
        if args.mask_parse_dir:
            mask = part_mask(os.path.join(args.mask_parse_dir, os.path.splitext(o["input"])[0] + ".png"),
                             classes, args.mask_dilate, args.mask_feather).to(device)
        res, _ = paste_back(frame, aligned, q["quad"], args.feather, mask=mask)
        stem = os.path.splitext(q["source"])[0]
        if stem in done:
            stem = os.path.splitext(os.path.basename(o["sample"]))[0]
        to_img(res).save(os.path.join(out_dir, stem + ".png"))
        done[stem] = (q, frame.cpu(), res.cpu())
    print(f"{len(done)} 枚 -> {out_dir}")

    names = [n for n in args.grid.split(",") if n] or list(done)[:6]
    tiles = []
    for n in names:
        if n not in done:
            continue
        q, fr, res = done[n]
        quad = np.asarray(q["quad"])
        c, half = quad.mean(0), q["qsize"] * args.grid_crop / 2
        box = tuple(int(v) for v in (c[0] - half, c[1] - half, c[0] + half, c[1] + half))
        col = []
        for x in (fr, res):
            im = to_img(x)
            pad = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), "white")  # フレームの外は白
            pad.paste(im, (-box[0], -box[1]))
            col.append(pad.resize((320, 320), Image.LANCZOS))
        tiles.append(col)
    if tiles:
        g = Image.new("RGB", (322 * len(tiles), 644), "white")
        for j, col in enumerate(tiles):
            for i, im in enumerate(col):
                g.paste(im, (322 * j, 322 * i))
        g.save(os.path.join(out_dir, "grid.jpg"), quality=92)


if __name__ == "__main__":
    main()
