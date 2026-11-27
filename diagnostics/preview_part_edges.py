"""
部位の境界の線 (animeface/part_edges.py) を足した ④ の入力を、設定ごとに .pt に書き出す (比較用。既存のファイルは変えない)。
書き出した .pt を scripts/sdedit_convert.py --cond_pt に渡して ⑤ を回す:

  docker compose run --rm --entrypoint python shell diagnostics/preview_part_edges.py \
      --settings 0.5:1,0.8:1 --out_root /data/output/edges
  docker compose run --rm convert --cond_pt /data/output/edges/s0.8_w1/cond.pt --out_dir /data/output/edges/s0.8_w1
  docker compose run --rm paste --run_dir /data/output/edges/s0.8_w1 --out_dir /data/output/edges/s0.8_w1/pasted

処理の順は animeface/face_input_transform.transform_input と同じで、ストローク化の後に線を足す:
  fg_mask -> to_lab -> stroke_parts -> draw_part_edges -> 前景だけ置き換え -> 色の写像
--check なら、線なしでこの順に計算した結果が transform_input の出力と (値もメモリの並びも) 同じかを確かめる。

出力 (<out_root>/s<強さ>_w<太さ>[_b<ぼかし>][_sil]/):
  cond.pt            {"names", "cond", "args"} (sdedit_convert.py --cond_pt の入力。メモリの並びは保つ)
  refs_cond/<名前>.png  線を足した ④ の出力 (8 bit。確認用)
"""

import argparse
import os
import sys

import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animeface.color_transfer import COLOR_TRANSFER_CHOICES, ColorTransfer  # noqa: E402
from animeface.face_input_transform import fg_mask, from_lab, stroke_parts, to_lab, transform_input  # noqa: E402
from animeface.image_io import list_images, load_image, save_png  # noqa: E402
from animeface.part_edges import draw_part_edges  # noqa: E402


def make_cond(x, parse, ct, a, strength, width, silhouette, blur):
    m = fg_mask(x).cpu()
    y = x.cpu()
    lab = to_lab(y)
    lab = stroke_parts(lab, m, parse, a.stroke_shade_levels, a.stroke_eye_scale_w, a.stroke_eye_scale_h)
    if strength > 0:
        lab = draw_part_edges(lab, m, parse, strength, width, silhouette, blur, a.stroke_eye_scale_w,
                              a.stroke_eye_scale_h)
    y = m * from_lab(lab) + (1 - m) * y
    return ct(y, m) if ct is not None else y


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_samples", default="/data/output/aligned")
    p.add_argument("--stroke_parse_dir", default="/data/output/parse")
    p.add_argument("--out_root", default="/data/output/edges")
    p.add_argument("--settings", default="0.5:1,0.8:1,0.8:2,1.0:2",
                   help="強さ:太さ[:ぼかし] をカンマ区切りで (太さは境界の片側の px)")
    p.add_argument("--silhouette", action="store_true", help="顔の外形 (背景との境界) にも線を引く")
    # ④ の設定 (scripts/sdedit_convert.py の既定値 = 採用構成)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--stroke_shade_levels", type=int, default=2)
    p.add_argument("--stroke_eye_scale_w", type=float, default=1.2)
    p.add_argument("--stroke_eye_scale_h", type=float, default=1.8)
    p.add_argument("--color_transfer", default="lab_meanstd_l", choices=COLOR_TRANSFER_CHOICES)
    p.add_argument("--color_stats", default="lists/face_color_stats.json")
    p.add_argument("--color_strength", type=float, default=1.0)
    p.add_argument("--check", action="store_true", help="線なしの結果が transform_input と同じかを確かめる")
    a = p.parse_args()

    names = list_images(a.base_samples)
    stems = [os.path.splitext(n)[0] for n in names]
    x = th.stack([load_image(os.path.join(a.base_samples, n), a.image_size) for n in names])
    parse = th.stack([th.from_numpy(np.asarray(Image.open(os.path.join(a.stroke_parse_dir, s + ".png")),
                                               dtype=np.int64)) for s in stems])
    ct = (ColorTransfer(a.color_stats, a.color_transfer, a.color_strength) if a.color_transfer != "none" else None)

    if a.check:
        ref = transform_input(x, parse, color_transfer=ct, stroke_shade_levels=a.stroke_shade_levels,
                              stroke_eye_scale_w=a.stroke_eye_scale_w, stroke_eye_scale_h=a.stroke_eye_scale_h)
        c0 = make_cond(x, parse, ct, a, 0.0, 1, False, 0.0)
        same = th.equal(ref, c0) and ref.stride() == c0.stride()
        print(f"線なし vs transform_input: 値が同じ {th.equal(ref, c0)}、並びが同じ {ref.stride() == c0.stride()}")
        if not same:
            sys.exit(1)

    for s in a.settings.split(","):
        v = [float(t) for t in s.split(":")]
        strength, width, blur = v[0], int(v[1]), (v[2] if len(v) > 2 else 0.0)
        tag = f"s{strength:g}_w{width}" + (f"_b{blur:g}" if blur > 0 else "") + ("_sil" if a.silhouette else "")
        out = os.path.join(a.out_root, tag)
        os.makedirs(os.path.join(out, "refs_cond"), exist_ok=True)
        cond = make_cond(x, parse, ct, a, strength, width, a.silhouette, blur)
        th.save({"names": names, "cond": cond, "args": dict(vars(a), strength=strength, width=width, blur=blur)},
                os.path.join(out, "cond.pt"))
        for stem, c in zip(stems, cond):
            save_png(c, os.path.join(out, "refs_cond", f"{stem}.png"))
        print(f"{tag}: {len(names)} 枚 -> {out}  (channels_last {cond.is_contiguous(memory_format=th.channels_last)})")


if __name__ == "__main__":
    main()
