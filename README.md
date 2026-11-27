# sdedit-for-animeface

実写の顔 (3DGS 用のマルチビュー画像) を、アニメ顔の DDPM で SDEdit によりアニメ調に変換する。
公式の SDEdit ([ermongroup/SDEdit](https://github.com/ermongroup/SDEdit)) を土台に、
ilvr-for-anime-faceparts の変換のうち、いま採用している構成だけを移植した。
公式のコード (`main.py`、`runners/`、`models/` など) は参照用にそのまま残している。

## 変換の流れ

| 段 | 処理 | 実装 |
|---|---|---|
| ① | 68 点ランドマークの検出 (face_alignment: SFD -> 2DFAN4。最大の顔) | `scripts/detect_face_landmarks.py` |
| ② | 切り出し (head 方式: 頭頂 0.037・顎先 0.906 を FFHQ に合わせる。向きは FFHQ の quad、横の中心は目より上の前景の中央、256px。ffhq 方式・回転なしも選べる) | `animeface/face_align.py`、`scripts/align_faces.py` |
| ③ | 顔パーシング (BiSeNet、CelebAMask-HQ の 19 クラス) | `animeface/face_parsing.py`、`scripts/parse_faces.py` |
| ④ | 入力の加工: 部位別ストローク (影 2 段、目の領域を横 1.2・縦 1.8 倍) -> 色の写像 `lab_meanstd_l` (統計 `lists/face_color_stats.json`) | `animeface/face_input_transform.py`、`animeface/color_transfer.py` |
| ⑤ | SDEdit: 間引き 100 ステップ、t0_ratio 0.5 (間引き後の番号 49 = 元の過程の t 494、√ᾱ 0.2436)、K=1、clip_denoised、全入力で同じシード 0 | `animeface/sdedit.py`、`scripts/sdedit_convert.py` |
| ⑥ | 元のフレームへの貼り戻し (quad の逆変換。quad の外縁 8% でぼかし、頭部の部位のマスク (服を除く。4px 膨張、σ 4px) を掛ける) | `animeface/face_align.py`、`scripts/paste_back_faces.py` |

モデルは EDM2 の magnitude-preserving U-Net (`animeface/unet_mp.py`、256px、128ch、`learn_sigma=True` で 6 チャネル出力)、
β は `cosine_shift4` (1000 ステップ)。設定は `ddpm/cfg/face256-mp128-ema_0.9995_250000.infer.cfg`。

### 公式 SDEdit との違い

| 項目 | 公式 SDEdit (`runners/image_editing.py`) | このリポジトリ (`animeface/sdedit.py`) |
|---|---|---|
| モデル | DDPM の UNet (`models/diffusion.py`、3ch 出力) | MPUNetModel (6ch 出力: epsilon と分散の補間係数) |
| β | 線形 | `cosine_shift4` |
| 間引き | なし (`--t` ステップすべて回す) | 100 ステップに間引き、番号 49 -> 0 の 50 ステップ。モデルには元の番号を渡す |
| 開始の t | `--t` (元の番号、ā[t-1] を使う) | `--t0_ratio` (t0 = int(ratio x 100)、間引き後の番号 t0-1 から) |
| 平均 | epsilon から直接 (x0 をクリップしない) | epsilon -> x0 -> [-1, 1] に clamp -> 事後平均 |
| 分散 | 固定 (fixedsmall / fixedlarge) | 学習した値 (LEARNED_RANGE) |
| マスク | 常に使う (編集しない領域を入力に戻す) | 使わない |
| 乱数 | グローバルな乱数 | サンプルごとの Generator。全入力で同じシード (視点が違っても同じノイズ) |
| 入出力 | `.pth` の `[mask, img]`、`torchvision.utils.save_image` | PNG (`center_crop_arr`)、`round()` で 8 bit に丸める |

## 準備

```bash
cp .env.example .env      # パスを手元の環境に合わせる (説明は .env.example のコメント)
mkdir -p output/<シーン名> # HOST_OUTPUT_DIR (入出力のディレクトリは先に作る)
docker compose build
```

- チェックポイント `ddpm/anime-face256-mp128-ema_0.9995_250000.pt` (約 531MB、git 管理外) を置く。
- 重みのキャッシュ (`HOST_CACHE_DIR`、既定は `./.cache`、git 管理外) に、face_alignment の 2DFAN4・s3fd (`torch/hub/checkpoints/`) と
  BiSeNet の `face_parsing/resnet18.pt` を置く。無ければ初回に torch hub と GitHub の公開 URL から取得する。
- コンテナはホストのユーザー (`HOST_UID`、`HOST_GID`) で動くので、出力は root 所有にならない。

## 使い方

すべて `docker compose run --rm <サービス名> [opts]` で動かす。`[opts]` はサービスの既定の引数の後ろに付き、
同じオプションなら上書きになる。既定値は採用構成。

| サービス | 処理 | 例 |
|---|---|---|
| `shell` | 対話シェル | `docker compose run --rm shell` |
| `landmarks` | ① `/data/input` -> `/data/output/landmarks.json` | `docker compose run --rm landmarks --device cpu` |
| `align` | ② -> `/data/output/aligned/` (+ `quads.json`) | `docker compose run --rm align --mode ffhq --out_dir /data/output/aligned_ffhq` |
| `parse` | ③ `/data/output/aligned` -> `/data/output/parse/` | `docker compose run --rm parse --batch_size 8` |
| `convert` | ④ + ⑤ -> `/data/output/convert/` (`samples/`、`refs_cond/`、`config.json`) | `docker compose run --rm convert --t0_ratio 0.4 --seed 1` |
| `paste` | ⑥ -> `/data/output/pasted/` (元の解像度のフレーム) | `docker compose run --rm paste --feather 0.1` |
| `fit-color-stats` | 色の写像の統計の作り直し -> `/data/output/face_color_stats.json` | `docker compose run --rm fit-color-stats --num 2000` |
| `verify` | 回帰確認 (元リポジトリの出力と比べる。段: `landmarks` `align` `parse` `cond` `sdedit` `paste`) | `docker compose run --rm verify sdedit` |
| `export-orig` | 元リポジトリのコードで ④ の float の入力を書き出す (回帰確認用。元のイメージで動く) | `docker compose run --rm export-orig` |

① 〜 ⑥ を通す `pipeline` はフェーズ 4 で足す。それまでは `landmarks` -> `align` -> `parse` -> `convert` -> `paste` の順に動かす。
設定を変えて別の出力先に書くときは、後の段の入力も合わせる (例: `convert --out_dir /data/output/convert_t04` の後に
`paste --run_dir /data/output/convert_t04 --out_dir /data/output/pasted_t04`)。

### .env の主な変数

| 変数 | 用途 | コンテナ内 |
|---|---|---|
| `HOST_INPUT_DIR` | シーンの入力フレーム | `/data/input` (読み取り専用) |
| `HOST_OUTPUT_DIR` | 全段の出力 | `/data/output` |
| `MODEL_PATH`、`MODEL_CFG`、`COLOR_STATS` | チェックポイント、cfg、色の写像の統計 (リポジトリからの相対) | `/app/...` |
| `HOST_CACHE_DIR` | 重みのキャッシュ | `/home/app/.cache` |
| `HOST_REAL_FACE_DIR`、`HOST_ANIME_FACE_DIR` | 統計の作り直しに使う実写の顔 (背景除去済みの FFHQ) とアニメ顔のデータ | `/data/real_face`、`/data/anime_face` (読み取り専用) |
| `HOST_ORIG_REPO`、`ORIG_IMAGE` | 回帰確認で参照する元リポジトリとそのイメージ | `/orig` (読み取り専用) |
| `GPU_ID`、`BASE_IMAGE`、`IMAGE_TAG`、`HOST_UID`、`HOST_GID` | 実行環境 | |

## 回帰確認

元リポジトリの採用構成の実行 (sample04、79 枚) と比べた (手順と結果、移植したファイルの出典は `patches/README.md`)。

- 各段に元の前段の出力を入れると、②④⑤⑥ と色の統計は画素単位で完全に一致する (同じ GPU、PyTorch 2.4.1 + CUDA 12.4)。
- ① face_alignment は実行ごとに数十点が 1〜8 px ずれる (元の環境で動かし直しても同じ)。③ は環境の違いで 0.001% の画素が違う。
- ① 〜 ⑥ を通すと、①③ の差が下流に及び、最終出力は 37 / 79 枚が完全に一致、画素の 1.3% が違う。

段ごとの確認は `docker compose run --rm --entrypoint bash verify diagnostics/run_stage_checks.sh`。

## ライセンス

公式 SDEdit の部分は MIT (`LICENSE`)。**`animeface/unet_mp.py` だけは CC BY-NC-SA 4.0** (NVlabs/edm2 由来) で、
このファイルを含む限り営利目的には使えない。guided-diffusion 由来のファイルは MIT (OpenAI)、`animeface/face_parsing.py` は MIT (anime-face-alignment、元は yakhyo/face-parsing)。
① の face_alignment (pip の依存) は BSD-3-Clause。詳細は `THIRD_PARTY_LICENSES.md`。

## 謝辞と引用

土台は SDEdit の公式実装。拡散過程は [openai/guided-diffusion](https://github.com/openai/guided-diffusion)、
モデルは [NVlabs/edm2](https://github.com/NVlabs/edm2)、ランドマークは [face-alignment](https://github.com/1adrianb/face-alignment)、
顔パーシングは [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) に基づく。

```
@inproceedings{
      meng2022sdedit,
      title={{SDE}dit: Guided Image Synthesis and Editing with Stochastic Differential Equations},
      author={Chenlin Meng and Yutong He and Yang Song and Jiaming Song and Jiajun Wu and Jun-Yan Zhu and Stefano Ermon},
      booktitle={International Conference on Learning Representations},
      year={2022},
}
```
