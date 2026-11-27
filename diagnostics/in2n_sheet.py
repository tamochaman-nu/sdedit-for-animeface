"""
全ビューのフレーム (3DGS の描画など。<名前>.png) から、写真の quad の周りを切り出して一覧の画像を作る (診断用)。
切り出す範囲は quad の中心から一辺の --crop 倍。--mark の名前は赤枠で囲む (目標にしたビューなど)。

  docker compose run --rm --entrypoint python shell diagnostics/in2n_sheet.py \
      --frames_dir /data/output/in2n/r0/bake/renders --out /data/output/in2n/r0/sheet.jpg --mark Cam1_frame_00009
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--quads", default="/data/output/aligned/quads.json")
    p.add_argument("--mark", default="", help="赤枠で囲む名前 (カンマ区切り)")
    p.add_argument("--crop", type=float, default=1.24)
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--cols", type=int, default=16)
    a = p.parse_args()
    with open(a.quads) as f:
        q = {os.path.splitext(v["source"])[0]: v for v in json.load(f)["images"].values()}
    mark = set(n for n in a.mark.split(",") if n)
    T, tiles = a.tile, []
    for n in sorted(q):
        im = Image.open(os.path.join(a.frames_dir, n + ".png")).convert("RGB")
        c, h = np.asarray(q[n]["quad"]).mean(0), q[n]["qsize"] * a.crop / 2
        t = im.crop((int(c[0] - h), int(c[1] - h), int(c[0] + h), int(c[1] + h))).resize((T, T), Image.LANCZOS)
        d = ImageDraw.Draw(t)
        d.text((2, 1), n.replace("_frame_000", "_"), fill=(255, 0, 0) if n in mark else (0, 0, 0))
        if n in mark:
            d.rectangle([0, 0, T - 1, T - 1], outline=(255, 0, 0), width=2)
        tiles.append(t)
    rows = (len(tiles) + a.cols - 1) // a.cols
    g = Image.new("RGB", (a.cols * T, rows * T), "white")
    for i, t in enumerate(tiles):
        g.paste(t, ((i % a.cols) * T, (i // a.cols) * T))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    g.save(a.out, quality=92)


if __name__ == "__main__":
    main()
