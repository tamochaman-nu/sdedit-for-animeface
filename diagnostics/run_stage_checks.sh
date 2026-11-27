#!/usr/bin/env bash
# フェーズ 3 の段ごとの回帰確認。各段の入力に元リポジトリ (/orig) の前段の出力を使い、①③ の差が ②④⑥ に入らないようにする。
# verify サービスの中で動かす (/orig が読み取り専用で見える):
#   docker compose run --rm --entrypoint bash verify diagnostics/run_stage_checks.sh [段 ...]
# 段: landmarks align parse cond paste (省略すると全部)。出力は /data/output/verify/stages/ に書く。
# cond は、元のコードで書き出した ④ の float (/data/output/verify/orig_cond.pt。docker compose run --rm export-orig) と比べる。
set -uo pipefail
cd /app
O=/orig
SCENE=$O/input/sample04-12119910
AL=${SCENE}_aligned256_head_rotffhq
RUN=$O/outputs/sample04_stroke/parts2eye/SD
V=/data/output/verify/stages
MODEL_PATH=${MODEL_PATH:-ddpm/anime-face256-mp128-ema_0.9995_250000.pt}
MODEL_CFG=${MODEL_CFG:-ddpm/cfg/face256-mp128-ema_0.9995_250000.infer.cfg}
mkdir -p "$V"
STAGES=${*:-landmarks align parse cond paste}
status=0
run() { echo "+ $*"; "$@" || status=1; }

for s in $STAGES; do
  echo "=== $s"
  case $s in
    landmarks)  # ①: 入力フレーム -> ランドマーク。元の JSON と比べる
      run python scripts/detect_face_landmarks.py --input_dir /data/input --out $V/landmarks.json
      run python diagnostics/verify.py landmarks --new $V/landmarks.json ;;
    align)      # ②: 元のランドマーク -> 切り出し。元の 256px 画像と quads.json と比べる
      run python scripts/align_faces.py --input_dir /data/input --landmarks ${SCENE}_landmarks.json \
        --out_dir $V/aligned --mode head --rotation ffhq --center_x head
      run python diagnostics/verify.py align --new_dir $V/aligned ;;
    parse)      # ③: 元の切り出し -> ラベル。元のラベルと比べる
      run python scripts/parse_faces.py --input_dir $AL --out_dir $V/parse
      run python diagnostics/verify.py parse --new_dir $V/parse ;;
    cond)       # ④⑤: 元の切り出しとラベル -> ④ の float、⑤ の出力。元の float と samples/ と比べる
      run python scripts/sdedit_convert.py --model_path "$MODEL_PATH" --cfg "$MODEL_CFG" \
        --base_samples $AL --stroke_parse_dir ${AL}_parse --out_dir $V/convert --save_cond_pt $V/cond.pt
      run python diagnostics/verify.py cond --new_pt $V/cond.pt
      run python diagnostics/verify.py sdedit --new_run $V/convert ;;
    paste)      # ⑥: 元の samples/、quads.json、ラベル -> 貼り戻し。元の pasted_mask/ と比べる
      run python scripts/paste_back_faces.py --run_dir $RUN --frames_dir /data/input --quads $AL/quads.json \
        --mask_parse_dir ${AL}_parse --out_dir $V/pasted
      run python diagnostics/verify.py paste --new_dir $V/pasted ;;
    *) echo "不明な段: $s"; status=1 ;;
  esac
done
exit $status
