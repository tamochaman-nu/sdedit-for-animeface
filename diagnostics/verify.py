"""
移植先の出力を、元リポジトリ (ilvr-for-anime-faceparts、/orig に読み取り専用でマウント) の出力と比べる回帰確認。
最初の引数で段を選ぶ。元の側のパスは --orig_root からの相対 (既定は採用構成の sample04 の実行)。

  landmarks  ① ランドマークの JSON。点ごとの距離 (px) の最大と平均、検出した顔の数・顔なしの違い
  align      ② 切り出した 256px の画像 (画素の差) と quads.json (quad の頂点の差 px、頭頂・顎先の位置)
  parse      ③ ラベル画像。一致しない画素の割合
  cond       ④ の float の出力 (.pt) どうし。差の最大と、メモリの並び (channels_last) が同じか
  sdedit     ⑤ の出力 (run_dir の samples/)。対応は両方の config.json の outputs[].input で取る
  paste      ⑥ 貼り戻したフレーム。ファイル名で対応させる

  docker compose run --rm verify landmarks --new /data/output/landmarks.json
  docker compose run --rm verify sdedit --new_run /data/output/convert
結果は標準出力と --report (JSON。省略すると比べた新しい側の隣) に書く。差が無ければ終了コード 0、あれば 1。
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ORIG_SCENE = "input/sample04-12119910"
ORIG_ALIGNED = ORIG_SCENE + "_aligned256_head_rotffhq"
ORIG_RUN = "outputs/sample04_stroke/parts2eye/SD"


def load_u8(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def image_diff_summary(pairs):
    """pairs: [(名前, 新しい画像のパス, 元の画像のパス)] -> (要約, 画像ごと)"""
    per, hist, n_px = [], np.zeros(256, dtype=np.int64), 0
    for name, pa, pb in pairs:
        a, b = load_u8(pa), load_u8(pb)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        n_px += a.shape[0] * a.shape[1]
        d = np.abs(a - b)
        hist += np.bincount(d.reshape(-1), minlength=256)
        per.append({"name": name, "max_abs_diff": int(d.max()), "mean_abs_diff": float(d.mean()),
                    "mismatched_pixels": int((d.max(-1) > 0).sum())})
    summary = {
        "num_images": len(per),
        "identical_images": sum(p["max_abs_diff"] == 0 for p in per),
        "max_abs_diff": max(p["max_abs_diff"] for p in per),
        "mean_abs_diff": float(np.mean([p["mean_abs_diff"] for p in per])),
        "mismatched_pixels": sum(p["mismatched_pixels"] for p in per),
        "total_pixels": n_px,
        "mismatched_ratio": sum(p["mismatched_pixels"] for p in per) / n_px,
        "abs_diff_histogram": {int(k): int(v) for k, v in enumerate(hist) if v},
    }
    return summary, per


def print_image_summary(title, s):
    print(f"{title}: 完全に一致 {s['identical_images']} / {s['num_images']} 枚、差の最大 {s['max_abs_diff']}、"
          f"平均 {s['mean_abs_diff']:.6f}、一致しない画素 {s['mismatched_pixels']} / {s['total_pixels']} "
          f"({100 * s['mismatched_ratio']:.4f}%)")
    h = s["abs_diff_histogram"]
    short = {k: v for k, v in h.items() if k <= 3}
    rest = sum(v for k, v in h.items() if k > 3)
    print("  差の分布 (|差|: 値の数):", short, f"(4 以上: {rest}。全体は --report の JSON)" if rest else "")
    return s["max_abs_diff"] == 0


def list_pngs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))


def cmd_landmarks(a, orig):
    with open(a.new) as f:
        new = json.load(f)["images"]
    with open(orig(a.orig)) as f:
        old = json.load(f)["images"]
    assert set(new) == set(old), sorted(set(new) ^ set(old))[:10]
    dists, per, notes = [], [], []
    for n in sorted(old):
        if new[n] is None or old[n] is None:
            if (new[n] is None) != (old[n] is None):
                notes.append(f"{n}: 顔の有無が違う (新 {new[n] is not None}, 元 {old[n] is not None})")
            continue
        if new[n]["num_faces"] != old[n]["num_faces"]:
            notes.append(f"{n}: 検出した顔の数が違う (新 {new[n]['num_faces']}, 元 {old[n]['num_faces']})")
        d = np.linalg.norm(np.asarray(new[n]["landmarks"]) - np.asarray(old[n]["landmarks"]), axis=1)
        bd = np.abs(np.asarray(new[n]["box"][:4]) - np.asarray(old[n]["box"][:4])).max()
        dists.append(d)
        per.append({"name": n, "max_px": float(d.max()), "mean_px": float(d.mean()), "num_points_moved": int((d > 0).sum()),
                    "box_max_abs_diff": float(bd)})
    dd = np.concatenate(dists)
    s = {"num_images": len(per), "max_px": float(dd.max()), "mean_px": float(dd.mean()),
         "identical_points": int((dd == 0).sum()), "total_points": int(dd.size),
         "identical_images": sum(p["max_px"] == 0 for p in per),
         "box_max_abs_diff": max(p["box_max_abs_diff"] for p in per), "notes": notes}
    print(f"① ランドマーク: {s['num_images']} 枚、点ごとの距離 最大 {s['max_px']:.4f} px、平均 {s['mean_px']:.4f} px、"
          f"同じ点 {s['identical_points']} / {s['total_points']}、全点が同じ画像 {s['identical_images']} 枚、"
          f"顔の枠の差の最大 {s['box_max_abs_diff']:.4f} px")
    worst = sorted(per, key=lambda p: -p["max_px"])[:5]
    print("  差の大きい画像:", [(p["name"], round(p["max_px"], 3)) for p in worst])
    for x in notes:
        print("  ", x)
    return s, per, s["max_px"] == 0 and not notes


def cmd_align(a, orig):
    new_dir, old_dir = a.new_dir, orig(a.orig_dir)
    names = list_pngs(old_dir)
    assert names == list_pngs(new_dir), "画像の一覧が違う"
    s, per = image_diff_summary([(n, os.path.join(new_dir, n), os.path.join(old_dir, n)) for n in names])
    ok = print_image_summary("② 切り出した画像", s)
    with open(os.path.join(new_dir, "quads.json")) as f:
        qn = json.load(f)["images"]
    with open(os.path.join(old_dir, "quads.json")) as f:
        qo = json.load(f)["images"]
    assert set(qn) == set(qo)
    qd = max(float(np.abs(np.asarray(qn[n]["quad"]) - np.asarray(qo[n]["quad"])).max()) for n in qo)
    hd = {k: max(abs(qn[n][k] - qo[n][k]) for n in qo) for k in ("head_top", "chin", "qsize", "out_of_frame")}
    s["quad_max_abs_diff_px"] = qd
    s["quads_other_max_abs_diff"] = hd
    print(f"  quads.json: quad の頂点の差の最大 {qd:.6g} px、ほかの値の差の最大 {hd}")
    return s, per, ok and qd == 0


def cmd_parse(a, orig):
    new_dir, old_dir = a.new_dir, orig(a.orig_dir)
    names = list_pngs(old_dir)
    assert names == list_pngs(new_dir), "ラベル画像の一覧が違う"
    per, bad, tot = [], 0, 0
    conf = np.zeros((19, 19), dtype=np.int64)
    for n in names:
        x = np.asarray(Image.open(os.path.join(new_dir, n)), dtype=np.int64)
        y = np.asarray(Image.open(os.path.join(old_dir, n)), dtype=np.int64)
        assert x.shape == y.shape, (n, x.shape, y.shape)
        m = x != y
        np.add.at(conf, (y[m], x[m]), 1)
        bad += int(m.sum())
        tot += m.size
        per.append({"name": n, "mismatched_pixels": int(m.sum()), "mismatched_ratio": float(m.mean())})
    pairs = sorted(((int(conf[i, j]), i, j) for i in range(19) for j in range(19) if conf[i, j]), reverse=True)[:8]
    s = {"num_images": len(per), "mismatched_pixels": bad, "total_pixels": tot, "mismatched_ratio": bad / tot,
         "identical_images": sum(p["mismatched_pixels"] == 0 for p in per),
         "max_image_mismatched_ratio": max(p["mismatched_ratio"] for p in per),
         "top_confusions_orig_to_new": [{"orig": i, "new": j, "pixels": c} for c, i, j in pairs]}
    print(f"③ ラベル: 一致しない画素 {bad} / {tot} ({100 * s['mismatched_ratio']:.4f}%)、完全に一致 "
          f"{s['identical_images']} / {s['num_images']} 枚、1 枚あたりの最大 {100 * s['max_image_mismatched_ratio']:.4f}%")
    if pairs:
        print("  多い食い違い (元のクラス -> 新しいクラス: 画素数):", [(i, j, c) for c, i, j in pairs])
    return s, per, bad == 0


def cmd_cond(a, orig):
    import torch as th

    dn = th.load(a.new_pt, map_location="cpu", weights_only=True)
    do = th.load(a.orig_pt, map_location="cpu", weights_only=True)
    assert list(dn["names"]) == list(do["names"]), "入力の一覧が違う"
    x, y = dn["cond"], do["cond"]
    d = (x.double() - y.double()).abs()
    per = [{"name": n, "max_abs_diff": float(d[i].max())} for i, n in enumerate(dn["names"])]
    s = {"num_images": len(per), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
         "identical_images": sum(p["max_abs_diff"] == 0 for p in per),
         "stride_new": list(x.stride()), "stride_orig": list(y.stride()),
         "channels_last_new": x.is_contiguous(memory_format=th.channels_last),
         "channels_last_orig": y.is_contiguous(memory_format=th.channels_last)}
    print(f"④ float の出力: 完全に一致 {s['identical_images']} / {s['num_images']} 枚、差の最大 {s['max_abs_diff']:.3g}、"
          f"平均 {s['mean_abs_diff']:.3g}。stride 新 {tuple(x.stride())}、元 {tuple(y.stride())}")
    return s, per, s["max_abs_diff"] == 0 and s["stride_new"] == s["stride_orig"]


def cmd_sdedit(a, orig):
    new_run, orig_run = a.new_run, orig(a.orig_run)
    with open(os.path.join(new_run, "config.json")) as f:
        new = {o["input"]: o["sample"] for o in json.load(f)["outputs"]}
    with open(os.path.join(orig_run, "config.json")) as f:
        old = {o["input"]: o["sample"] for o in json.load(f)["outputs"]}
    assert set(new) == set(old), sorted(set(new) ^ set(old))[:10]
    s, per = image_diff_summary([(n, os.path.join(new_run, new[n]), os.path.join(orig_run, old[n])) for n in sorted(old)])
    ok = print_image_summary(f"⑤ SDEdit の出力 ({new_run} vs {a.orig_run})", s)
    return s, per, ok


def cmd_paste(a, orig):
    new_dir, old_dir = a.new_dir, orig(a.orig_dir)
    names = list_pngs(old_dir)
    assert names == list_pngs(new_dir), "画像の一覧が違う"
    s, per = image_diff_summary([(n, os.path.join(new_dir, n), os.path.join(old_dir, n)) for n in names])
    ok = print_image_summary(f"⑥ 貼り戻したフレーム ({new_dir} vs {a.orig_dir})", s)
    return s, per, ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orig_root", default="/orig")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("landmarks")
    s.add_argument("--new", default="/data/output/landmarks.json")
    s.add_argument("--orig", default=ORIG_SCENE + "_landmarks.json")
    s = sub.add_parser("align")
    s.add_argument("--new_dir", default="/data/output/aligned")
    s.add_argument("--orig_dir", default=ORIG_ALIGNED)
    s = sub.add_parser("parse")
    s.add_argument("--new_dir", default="/data/output/parse")
    s.add_argument("--orig_dir", default=ORIG_ALIGNED + "_parse")
    s = sub.add_parser("cond")
    s.add_argument("--new_pt", required=True)
    s.add_argument("--orig_pt", default="/data/output/verify/orig_cond.pt", help="diagnostics/export_orig_cond.py の出力")
    s = sub.add_parser("sdedit")
    s.add_argument("--new_run", default="/data/output/convert")
    s.add_argument("--orig_run", default=ORIG_RUN)
    s = sub.add_parser("paste")
    s.add_argument("--new_dir", default="/data/output/pasted")
    s.add_argument("--orig_dir", default=ORIG_RUN + "/pasted_mask")
    for sp in sub.choices.values():
        sp.add_argument("--report", default="", help="結果の JSON")
    a = p.parse_args()

    def orig(path):
        return path if os.path.isabs(path) else os.path.join(a.orig_root, path)

    summary, per, ok = globals()["cmd_" + a.cmd](a, orig)
    new_side = {"landmarks": lambda: os.path.splitext(a.new)[0], "cond": lambda: os.path.splitext(a.new_pt)[0],
                "sdedit": lambda: a.new_run}.get(a.cmd, lambda: a.new_dir.rstrip("/"))()
    report = a.report or (os.path.join(new_side, f"verify_{a.cmd}.json") if a.cmd == "sdedit"
                          else f"{new_side}_verify_{a.cmd}.json")
    with open(report, "w") as f:
        json.dump({"summary": summary, "per_item": per, "args": vars(a)}, f, indent=1, ensure_ascii=False)
    print("->", report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
