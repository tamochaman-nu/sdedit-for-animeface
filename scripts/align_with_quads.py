"""
既存の切り出し (scripts/align_faces.py の quads.json) を使い回して、別のフレーム (同じカメラの描画など) を切り出す。
アニメ調の描画ではランドマークが外れやすいので、同じカメラの元の写真で求めた quad をそのまま使う
(描画の解像度と画素の対応が写真と同じであることが前提)。

フレームは --frames_dir/<quads.json の source の拡張子を除いた名前>.<png|jpg> を探す。--names を指定すると
その名前 (拡張子なし、カンマ区切り) だけを処理する。

出力 (--out_dir):
  aligned/<名前>.png   切り出した画像 (size 四方。align_faces.py と同じ warp_quad)
  cond.pt              aligned/ を load_image で読んだもの ({"names": [<名前>.png], "cond": [N x 3 x s x s]})。
                       scripts/sdedit_convert.py の --cond_pt に渡すと、入力の加工 (ストローク・色の写像) を
                       飛ばしてそのまま SDEdit の開始点に使う
  quads.json           入力の quads.json のうち処理した分で、source をこのフレームのファイル名に書き換えたもの
                       (scripts/paste_back_faces.py の --quads と --frames_dir <この --frames_dir> にそのまま使える)
  config.json          diagnostics/stroke_eval.py の --runs に渡せる並び ({"outputs": [{"input", "sample": aligned/<名前>.png,
                       "ref_png": --ref_dir/<名前>.png (無ければ sample 自身)}]})
  masks/<名前>.png     貼り戻しの重み (paste_back_faces.py と同じ式。--mask_parse_dir があれば部位のマスクを掛ける)
                       をフレームの座標で 8 bit にしたもの (焼き付けの範囲を決めるのに使う)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animeface.face_align import paste_back, warp_quad  # noqa: E402
from animeface.image_io import load_image  # noqa: E402
from scripts.paste_back_faces import part_mask  # noqa: E402


def load_full(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1
    return th.from_numpy(a).permute(2, 0, 1)


def find_frame(frames_dir, stem):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(frames_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--quads", required=True, help="元の写真の scripts/align_faces.py の quads.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--names", default="", help="処理する名前 (拡張子なし、カンマ区切り)。空なら quads.json の全部")
    p.add_argument("--supersample", type=int, default=4)
    p.add_argument("--ref_dir", default="", help="config.json の ref_png に書く比較の元 (例 /data/output/aligned)")
    # 貼り戻しの重み (paste_back_faces.py と同じ既定値)
    p.add_argument("--feather", type=float, default=0.08)
    p.add_argument("--mask_parse_dir", default=None)
    p.add_argument("--mask_classes", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,17,18")
    p.add_argument("--mask_dilate", type=int, default=4)
    p.add_argument("--mask_feather", type=float, default=4.0)
    a = p.parse_args()

    with open(a.quads) as f:
        qj = json.load(f)
    size = int(qj["args"]["size"])
    want = set(n for n in a.names.split(",") if n)
    classes = [int(c) for c in a.mask_classes.split(",")]
    for sub in ("aligned", "masks"):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    out_q, names, conds, missing = {}, [], [], []
    for key, q in sorted(qj["images"].items()):
        stem = os.path.splitext(q["source"])[0]
        if want and stem not in want:
            continue
        fp = find_frame(a.frames_dir, stem)
        if fp is None:
            missing.append(stem)
            continue
        frame = load_full(fp).to(device)
        h, w = frame.shape[-2:]
        if [w, h] != list(q["size"]):
            raise SystemExit(f"{fp}: 大きさ {w}x{h} が quads.json の {q['size']} と違う")
        out = warp_quad(frame, q["quad"], size, a.supersample)
        ap = os.path.join(a.out_dir, "aligned", key)
        Image.fromarray(((out.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(ap)
        names.append(key)
        conds.append(load_image(ap, size))
        mask = None
        if a.mask_parse_dir:
            mask = part_mask(os.path.join(a.mask_parse_dir, os.path.splitext(key)[0] + ".png"),
                             classes, a.mask_dilate, a.mask_feather).to(device)
        _, wgt = paste_back(th.zeros_like(frame), th.zeros(3, size, size, device=device), q["quad"], a.feather,
                            mask=mask)
        Image.fromarray((wgt[0].clamp(0, 1) * 255).round().byte().cpu().numpy()).save(
            os.path.join(a.out_dir, "masks", stem + ".png"))
        out_q[key] = {**q, "source": os.path.basename(fp)}
    if want - {os.path.splitext(q["source"])[0] for q in out_q.values()} - set(missing):
        raise SystemExit(f"quads.json に無い名前: {sorted(want - {os.path.splitext(q['source'])[0] for q in out_q.values()})}")
    if missing:
        raise SystemExit(f"フレームが無い: {missing[:5]} ほか {len(missing)} 枚")
    th.save({"names": names, "cond": th.stack(conds).float()}, os.path.join(a.out_dir, "cond.pt"))
    outs = [{"input": k, "sample": os.path.join("aligned", k),
             "ref_png": os.path.join(a.ref_dir, k) if a.ref_dir else os.path.join("aligned", k)} for k in names]
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump({"args": vars(a), "outputs": outs}, f, indent=1, ensure_ascii=False)
    with open(os.path.join(a.out_dir, "quads.json"), "w") as f:
        json.dump({"args": {**qj["args"], "reused_by": vars(a)}, "images": out_q}, f, indent=1)
    print(f"{len(names)} 枚 -> {a.out_dir}")


if __name__ == "__main__":
    main()
