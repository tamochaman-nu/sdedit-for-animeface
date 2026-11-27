"""
顔全体の色の写像 (animeface/color_transfer.ColorTransfer) に使う CIELAB の統計を求めて JSON に保存する。

  real   実写の顔 (背景除去済みの FFHQ): --exclude_list (評価に使う参照画像) を除いて、乱数で num 枚
  anime  アニメ顔のデータ: 乱数で num 枚
統計は前景 (白でない画素を 2px 膨らませたもの。face_input_transform.fg_mask) だけで求める。
画素数を抑えるため、--stats_size (既定 128) に縮小してから求める。
"""

import argparse
import json
import os
import random
import sys

import torch as th
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animeface.face_input_transform import fg_mask  # noqa: E402
from animeface.image_io import list_images, load_image  # noqa: E402
from animeface.color_transfer import lab_stats  # noqa: E402


def stats(d, names, size):
    x = th.stack([load_image(os.path.join(d, n), 256) for n in names])
    m = fg_mask(x)
    if size != 256:
        x = F.interpolate(x, size=(size, size), mode="area")
        m = F.interpolate(m.float(), size=(size, size), mode="area")
    s = lab_stats(x, m)
    s["num_images"] = len(names)
    s["fg_ratio"] = float((m >= 0.5).float().mean())
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", required=True)
    p.add_argument("--anime_dir", required=True)
    p.add_argument("--exclude_list", default="")
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--stats_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="lists/face_color_stats.json")
    args = p.parse_args()
    exclude = set()
    if args.exclude_list:
        with open(args.exclude_list) as f:
            exclude = {line.strip() for line in f if line.strip()}
    rng = random.Random(args.seed)
    real = rng.sample([n for n in list_images(args.real_dir) if n not in exclude], args.num)
    anime = rng.sample(list_images(args.anime_dir), args.num)
    out = {"args": vars(args), "real": stats(args.real_dir, real, args.stats_size),
           "anime": stats(args.anime_dir, anime, args.stats_size)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    for k in ("real", "anime"):
        print(k, "mean", [round(v, 2) for v in out[k]["mean"]], "std", [round(v, 2) for v in out[k]["std"]],
              "fg", round(out[k]["fg_ratio"], 3))


if __name__ == "__main__":
    main()
