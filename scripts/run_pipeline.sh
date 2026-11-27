#!/usr/bin/env bash
# ① 〜 ⑥ を通しで動かす (pipeline サービスの中身)。段ごとに別の python のプロセスで動かす。
# face_alignment (①) は GPU で FaceAlignment を作るときに torch.backends.cudnn.benchmark=True を立てる
# (face_alignment 1.5.0 の api.py 98 行目) ので、同じプロセスで後の段を動かすと
# cuDNN のカーネルの選び方が変わり、③⑤ の出力が段ごとのサービスで動かしたときとずれうるため。
#
#   docker compose run --rm pipeline [convert の opts]
#
# 入出力はコンテナ内の固定のパス (段ごとのサービスと同じ並び):
#   /data/input                      入力フレーム (.env の HOST_INPUT_DIR)
#   /data/output/landmarks.json      ①
#   /data/output/aligned/            ② (+ quads.json)
#   /data/output/parse/              ③
#   /data/output/convert/            ④⑤ (samples/、refs/、refs_cond/、config.json)
#   /data/output/pasted/             ⑥
#
# 設定の変え方 (既定は採用構成。同じオプションなら後ろが勝つ):
#   [opts]           convert (④⑤) に渡す。例: --t0_ratio 0.4 --seed 1
#                    --out_dir を変えると、⑥ はその出力を読み、<その --out_dir>/pasted に書く (採用構成の結果を上書きしない)
#   LANDMARKS_OPTS、ALIGN_OPTS、PARSE_OPTS、PASTE_OPTS
#                    ①②③⑥ に足す引数 (空白で区切る。値に空白は使えない)。例: -e PASTE_OPTS="--feather 0.1"
#                    入出力のパスは変えない (変えると後の段とつながらない。別の設定を試すときは HOST_OUTPUT_DIR を変える)
#   STAGES           動かす段 (既定: "landmarks align parse convert paste")。例: -e STAGES="convert paste"
#   MODEL_PATH、MODEL_CFG、COLOR_STATS   チェックポイント、cfg、色の写像の統計 (/app からの相対パス)
set -euo pipefail
cd /app

IN=/data/input
OUT=/data/output
MODEL_PATH=${MODEL_PATH:-ddpm/anime-face256-mp128-ema_0.9995_250000.pt}
MODEL_CFG=${MODEL_CFG:-ddpm/cfg/face256-mp128-ema_0.9995_250000.infer.cfg}
COLOR_STATS=${COLOR_STATS:-lists/face_color_stats.json}
STAGES=${STAGES:-landmarks align parse convert paste}

# convert の --out_dir を [opts] から拾い、⑥ の入力と出力先を合わせる (最後の指定が勝つのは argparse と同じ)
CONVERT_DIR=$OUT/convert
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case ${args[i]} in
    --out_dir) CONVERT_DIR=${args[i + 1]:-} ;;
    --out_dir=*) CONVERT_DIR=${args[i]#--out_dir=} ;;
  esac
done
CONVERT_DIR=${CONVERT_DIR%/}
if [ "$CONVERT_DIR" = "$OUT/convert" ]; then PASTE_DIR=$OUT/pasted; else PASTE_DIR=$CONVERT_DIR/pasted; fi

read -ra LANDMARKS_ARGS <<< "${LANDMARKS_OPTS:-}"
read -ra ALIGN_ARGS <<< "${ALIGN_OPTS:-}"
read -ra PARSE_ARGS <<< "${PARSE_OPTS:-}"
read -ra PASTE_ARGS <<< "${PASTE_OPTS:-}"

run() {
  echo "+ $*"
  local t0=$SECONDS
  "$@"
  echo "  ($((SECONDS - t0)) 秒)"
}

for s in $STAGES; do
  echo "=== $s"
  case $s in
    landmarks)
      run python scripts/detect_face_landmarks.py --input_dir "$IN" --out "$OUT/landmarks.json" \
        "${LANDMARKS_ARGS[@]}" ;;
    align)
      run python scripts/align_faces.py --input_dir "$IN" --landmarks "$OUT/landmarks.json" \
        --out_dir "$OUT/aligned" --mode head --rotation ffhq --center_x head "${ALIGN_ARGS[@]}" ;;
    parse)
      run python scripts/parse_faces.py --input_dir "$OUT/aligned" --out_dir "$OUT/parse" "${PARSE_ARGS[@]}" ;;
    convert)
      run python scripts/sdedit_convert.py --model_path "$MODEL_PATH" --cfg "$MODEL_CFG" --color_stats "$COLOR_STATS" \
        --base_samples "$OUT/aligned" --stroke_parse_dir "$OUT/parse" --out_dir "$OUT/convert" "$@" ;;
    paste)
      run python scripts/paste_back_faces.py --run_dir "$CONVERT_DIR" --frames_dir "$IN" \
        --quads "$OUT/aligned/quads.json" --mask_parse_dir "$OUT/parse" --out_dir "$PASTE_DIR" "${PASTE_ARGS[@]}" ;;
    *)
      echo "不明な段: $s (landmarks align parse convert paste)" >&2
      exit 2 ;;
  esac
done
echo "完了: $STAGES"
