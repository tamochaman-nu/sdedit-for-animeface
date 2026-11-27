# 公式 SDEdit (ermongroup/SDEdit) からの変更の記録

土台は公式 SDEdit の `66a5e44db6c36d5979c323781675d67117c0fb04`。変換の処理はすべて新規ファイルで足し、
公式の既存ファイル (`main.py`、`runners/`、`models/`、`functions/`、`configs/`、`colab_utils/`) は参照用にそのまま残している。

## 既存ファイルの変更

| ファイル | diff | 内容 |
|---|---|---|
| `.gitignore` | [`gitignore.diff`](gitignore.diff) | シーンの入出力 (`/input/`、`/output/`) とチェックポイント (`/ddpm/*.pt`) を git 管理外にする。`.env` と `.cache` は公式の `.gitignore` (117 行目、54 行目) で無視済み |
| `README.md` | [`readme.diff`](readme.diff) | 公式の README を、このリポジトリの変換・実行方法・公式との違い・ライセンスの説明に書き換えた (公式 SDEdit の引用は残した) |
| `README.md` | [`readme_phase3.diff`](readme_phase3.diff) | フェーズ 3: ①〜④、⑥ の実装と段ごとのサービス、統計の作り直しのデータの変数、回帰確認の結果、face_parsing と face_alignment のライセンスと謝辞を足した |

## 移植したファイルの出典

元リポジトリ: `ilvr-for-anime-faceparts` (tamochaman-nu、ブランチ `dev/ilvr-with-sdedit`、HEAD `87152b3fa55a845b39c5afd9a2abd54600083896`)。
元リポジトリの移植元の多くは未追跡か未コミットの変更を含む (下の「元の状態」) ので、出典は移植元のファイルの sha256 で示す。

### そのままコピーしたもの (移植先と sha256 が同じ)

| 移植先 | 移植元 | 元の状態 | sha256 |
|---|---|---|---|
| `animeface/unet_mp.py` | `guided_diffusion/unet_mp.py` (元は学習フォーク tamochaman-nu/guided-diffusion の `dev/detect-resblock-overflow` `5bb6a5c`) | コミット済み | `19031385ba502c7f5c668bd7a5f7d73e1ed535b5d90750844d1d956d84a18319` |
| `animeface/gaussian_diffusion.py` | `guided_diffusion/gaussian_diffusion.py` (OpenAI guided-diffusion に `cosine_shift` を追加したもの。元リポジトリの `patches/gaussian-diffusion-cosine-shift.diff`) | コミット済み | `857adda9b20a8d0fdd550a42cbe2af4392bad4d47720463b4daa2384ce9698fb` |
| `animeface/respace.py` | `guided_diffusion/respace.py` | コミット済み | `b9a509b28c10a52ca3ee8cc48f40205b1d70aecfc95356b540dc88415368c2de` |
| `animeface/nn.py` | `guided_diffusion/nn.py` (gaussian_diffusion.py が import する) | コミット済み | `d9547fb22b421a16dc079cb82a9d98630100b5e1b2a710d343f4d3f90b8661f3` |
| `animeface/losses.py` | `guided_diffusion/losses.py` (同上) | コミット済み | `0b0f5a5532910fbc7f16d30f2b891d8f1e20aa88881589ce4eacc2cc6ce06026` |
| `ddpm/anime-face256-mp128-ema_0.9995_250000.pt` (git 管理外) | `ddpm/anime-face256-mp128-ema_0.9995_250000.pt` | git 管理外 | `aeb2c2d588ae1e4c41da6ff1e6ed70d1a90a0db6e92a97fd83a5af440f81b392` |
| `ddpm/cfg/face256-mp128-ema_0.9995_250000.infer.cfg` | `ddpm/face256-mp128-ema_0.9995_250000.infer.cfg` (推論用。CHANNEL_MULT だけを埋めた。根拠はファイル冒頭) | 未追跡 | `4cd3812395628edc05c349f60bd253bdc575a82add8e87ea1f06ac5a6b18b892` |
| `ddpm/cfg/face256-mp128-ema_0.9995_250000.cfg` | `ddpm/face256-mp128-ema_0.9995_250000.cfg` (学習時の cfg。上の出典として置く) | 未追跡 | `2212d8a4d771cecd327a23f1c62d9383b45459a79f499bbc9357d8f88bca7310` |
| `lists/face_color_stats.json` | `lists/face_color_stats.json` (④ 色の写像の統計。`scripts/fit_face_color_stats.py` の出力) | 未追跡 | `d471b9d963d1729985e8a4efff32de6613d23d4cd85058cefbe235418cf8d8d1` |
| `lists/face_color_stats_exclude.txt` | `diagnostics/outputs/stability/ref_list16.txt` (上の統計を作るときに実写から除いた 16 枚) | git 管理外 | `582909541fd52689edab15d20c00565cc9da3bbd6bde9396bc754036fc0637c4` |
| `animeface/face_align.py` | `guided_diffusion/face_align.py` (② 切り出しと ⑥ 貼り戻し。ffhq 方式・回転なしも含めて全機能を残した) | 未追跡 | `5043a378fd254dcc8b873d6c28c04f389705e9dce2297c1a01bd8587d9c42f6d` |
| `animeface/face_parsing.py` | **anime-face-alignment** (tamochaman-nu、HEAD `c4360eb`) の `util/face_parsing.py`。HEAD からの変更 (11 行の追加) がステージ済みの作業ツリーのもの | 変更あり (未コミット) | `398a2cd94047797dbc9febe967509c6ed7e4ad481aeea3a8b9b719ad31866b50` |

### import 先と説明文だけを変えたもの

元リポジトリの `guided_diffusion/` を `animeface/` に読み替えた。処理とオプションは変えていない。

| 移植先 (sha256) | 移植元 (元の状態、sha256) | 変えたところ |
|---|---|---|
| `scripts/detect_face_landmarks.py` (`834634b3…`) | `scripts/detect_face_landmarks.py` (未追跡、`44c081e5…`) | docstring の実行方法 (anime-face-alignment のイメージ -> `docker compose run --rm landmarks`) |
| `scripts/align_faces.py` (`1bcf178b…`) | `scripts/align_faces.py` (未追跡、`88ee3337…`) | docstring と import 先 (`guided_diffusion.face_align` -> `animeface.face_align`)。既定の `--mode ffhq` は変えず、採用構成の `--mode head --rotation ffhq --center_x head` は `align` サービスの entrypoint で渡す |
| `scripts/parse_faces.py` (`358085e9…`) | `scripts/parse_faces.py` (未追跡、`726a4a95…`) | docstring、`--afa_root` を除き `animeface.face_parsing` を import |
| `scripts/paste_back_faces.py` (`f2ef6984…`) | `scripts/paste_back_faces.py` (未追跡、`e654b70c…`) | docstring と import 先 |
| `scripts/fit_face_color_stats.py` (`3d07d89d…`) | `scripts/fit_face_color_stats.py` (未追跡、`0b979fc6…`) | docstring と import 先 (`fg_mask` <- `animeface.face_input_transform`、`list_images`・`load_image` <- `animeface.image_io`、`lab_stats` <- `animeface.color_transfer`) |

### 書き写したもの (必要な部分だけを写した。計算は同じ)

| 移植先 | 移植元 (元リポジトリの行番号) | 写さなかった部分 |
|---|---|---|
| `animeface/model.py` | `guided_diffusion/ilvr_eye.py` 38-103 (`read_cfg`、`_bool`、`create_model_from_cfg`、`create_diffusion_from_cfg`、`load_model`)、`guided_diffusion/script_util.py` 8 (`NUM_CLASSES`)、386-424 (`create_gaussian_diffusion`) | script_util.py の `unet.py` に依存する部分 |
| `animeface/image_io.py` | `guided_diffusion/image_datasets.py` 126-143 (`center_crop_arr`)、`guided_diffusion/ilvr_eye.py` 243-293 (`list_images`、`load_image`、`to_uint8_hwc`、`save_png`)、`guided_diffusion/script_util.py` 441-452 (`str2bool`) | image_datasets.py の DataLoader (mpi4py、blobfile)、`list_images` の `{side}` の分岐 |
| `animeface/sdedit.py` | `guided_diffusion/sdedit.py` 27-42 (`sdedit_t0`、`start_info`)、50-113 (`sdedit_ilvr_sample` -> `sdedit_sample`)、`guided_diffusion/ilvr_eye.py` 159-168 (`SampleRNG`) | ILVR (`resizers`)、マスク合成 (`mask`、`x_blend`)、繰り返し (`x_init`、`sdedit_repeat`)、途中経過の保存、`sdedit_t0_from_sqrt_alpha` |
| `animeface/face_input_transform.py` | `guided_diffusion/face_input_transform.py` (未追跡、`de522b80…`) 28 (`FG_THRESH`)、31-36 (`fg_mask`)、39-42 (`to_lab`)、45-47 (`from_lab`)、98-109 (`PART_GROUPS`、`SHADED_PARTS`)、157-167 (`_scale_region`)、185-230 (`stroke_parts`)、233-267 (`transform_input`) | 抽象化 (双方向フィルタ、L* の段階化、XDoG)、汎用のストローク (SLIC)、目の塗り方 `iris`。`stroke_parts` と `transform_input` はその分岐と引数を除いた |
| `animeface/color_transfer.py` | `guided_diffusion/ilvr_ref_transform.py` (コミット済み、`49f33788…`) 30-31、34-36 (`to_lab_pixels`)、39-49 (`lab_stats`)、52-81 (`ColorTransfer`) | 目のマスクの読み込み (`load_masks`) と `ilvr_eye.image_path` の import |
| `scripts/sdedit_convert.py` | `scripts/sdedit_sample.py` 137-271 (`main`)、274-338 (`create_argparser`) | ILVR、マスク合成、色の補正、抽象化、汎用のストローク、瞳の色の固定、`--seed_per_file`、繰り返し (K)、途中経過の保存。既定値は採用構成に変えた。出力の名前は `samples/<入力名>.png` (元は `<入力名>_<オプションの要約>_seed<S>.png`) |

元リポジトリの `guided_diffusion/sdedit.py` と `scripts/sdedit_sample.py` は、HEAD から未コミットの変更を含む
(sha256 はそれぞれ `15cad4d275f9d22fee8cfce9cad7992b1ffa9a6a77d4c3e59f9becddf3126fa0`、`11d89b09c8c2a8fa7edb1b3c550565755496da5a5043275d8fe0922b412ef92e`)。
`ilvr_eye.py` (`2627b891…`)、`script_util.py` (`d47bb38e…`)、`image_datasets.py` (`220d6699…`) はコミット済み。

### 重みのキャッシュ (`.cache/`、git 管理外)

anime-face-alignment の Docker ボリュームからコピーした (取得元は torch hub と GitHub の公開 URL)。

| ファイル | コピー元 | sha256 |
|---|---|---|
| `.cache/torch/hub/checkpoints/2DFAN4-11f355bf06.pth.tar` | ボリューム `anime-face-alignment_torch_cache` の `hub/checkpoints/` | `11f355bf0693120222f5955ce3f9dc8fb5763ebb30a47d7906e509490d32e4aa` |
| `.cache/torch/hub/checkpoints/s3fd-619a316812.pth` | 同上 | `619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543` |
| `.cache/face_parsing/resnet18.pt` | ボリューム `anime-face-alignment_face_parsing_cache` | `18c44e06b77e47da86fe2b48fac78e0ccbd0b5241c829f9487ebf7fd4a6389e6` |

## 新規に書いたもの

| ファイル | 内容 |
|---|---|
| `animeface/__init__.py` | パッケージの説明 |
| `diagnostics/export_orig_cond.py` | 元リポジトリのコードを編集せずに読み込み、④ の float の出力を .pt に書き出す (⑤ の回帰確認の入力) |
| `diagnostics/verify.py` | 回帰確認 (移植先の出力と元の実行の出力の比較。段: landmarks、align、parse、cond、sdedit、paste) |
| `diagnostics/run_stage_checks.sh` | 段ごとの回帰確認 (各段の入力に元の前段の出力を使う) |
| `scripts/run_pipeline.sh` | ① 〜 ⑥ の通し (`pipeline` サービス)。段ごとに別のプロセスで動かす |
| `Dockerfile`、`docker-compose.yml`、`.env.example`、`.dockerignore`、`requirements-animeface.txt` | 実行環境 |
| `THIRD_PARTY_LICENSES.md` | 移植したファイルのライセンス (CC BY-NC-SA 4.0 の `unet_mp.py`、OpenAI と anime-face-alignment の MIT、face_alignment の BSD-3) |
| `patches/` | この記録と、既存ファイルの diff |

## 回帰確認の結果

### フェーズ 2: ⑤ SDEdit (2026-11-27)

入力は、元リポジトリのコードで書き出した ④ の float の出力 (`diagnostics/export_orig_cond.py`)。書き出した float を 8 bit に
丸めたものは、元の実行の `refs_cond/` の 79 枚と完全に一致した (元の実行の後に更新されたコードでも ④ は同じ)。

| 比較 | 結果 |
|---|---|
| 移植した ⑤ (新しいイメージ `sdedit-for-animeface:latest`) vs 元の実行の `samples/` | 79 / 79 枚が完全に一致 (差の最大 0) |
| 移植した ⑤ (元のイメージ `ilvr-anime-faceparts:latest`) vs 同上 | 79 / 79 枚が完全に一致 |
| 元の `scripts/sdedit_sample.py` を元のイメージで再実行 vs 同上 | 79 / 79 枚が完全に一致 (元の実行自体が再現できる) |

同じ GPU (RTX 4090)、torch 2.4.1+cu124、cuDNN 9.1.0、バッチ 16 (名前順)。

注意: ④ の出力は numpy の HWC を permute したテンソルなので、メモリの並びが channels_last のまま ⑤ のモデルに入る。
これを `.contiguous()` で NCHW に並べ替えると、cuDNN の畳み込みのカーネルが変わり、2.2% の画素が 1〜5 階調ずれた
(差の分布: 1 が 117578、2 が 129、3 が 20、4 が 2、5 が 1)。並びを保てば完全に一致する。

### フェーズ 3: ①〜④、⑥ (2026-11-27)

**段ごと** (`diagnostics/run_stage_checks.sh`。各段の入力に元の前段の出力を使う):

| 段 | 入力 | 結果 |
|---|---|---|
| ① ランドマーク | 入力フレーム 79 枚 | 5372 点のうち 5330 点が一致。ずれた 42 点 (21 枚) は 1 px か 5〜7.8 px の飛び (最大 7.81 px、平均 0.020 px)。顔の枠の差は最大 0.034 px |
| ② 切り出し | 元の `landmarks.json` | 256px 画像 79 / 79 枚と `quads.json` が完全に一致 |
| ③ 顔パーシング | 元の ② の画像 | 一致しない画素 57 / 5,177,344 (0.0011%)。完全に一致 38 / 79 枚 |
| ④ 入力の加工 | 元の ② の画像と ③ のラベル | float の出力が 79 / 79 枚で完全に一致 (メモリの並び channels_last も同じ)。続く ⑤ も 79 / 79 枚で一致 |
| ⑥ 貼り戻し | 元の `samples/`、`quads.json`、ラベル | 79 / 79 枚が完全に一致 (元の `pasted_mask/` は既定値 feather 0.08、膨張 4px、σ 4px で作られていた) |
| 統計の作り直し | FFHQ・アニメ顔のデータ | `lists/face_color_stats.json` と平均・標準偏差・分位点がすべて一致 |

**① と ③ の差の原因**:
- ① face_alignment は実行ごとに結果が少し変わる (`cudnn.benchmark = True` を設定するため、と考えられる)。
  移植先で 2 回動かした結果どうしでも 32 点 (最大 6 px)、元のイメージ (anime-face-alignment:latest) で元のスクリプトを
  動かし直した結果と元の JSON でも 37 点 (最大 6 px) がずれた。座標はヒートマップの最大の位置を整数にしたものなので、
  ずれは 1 マス分 (元のフレームで数 px) ずつ飛ぶ。差は移植や環境ではなく、この揺らぎによる。
- ③ BiSeNet はどちらの環境でも毎回同じ結果になる (移植先で 2 回、元のイメージで再実行しても差 0)。57 画素の差は
  環境の違い (元: PyTorch 2.14.1 + CUDA 12.6、移植先: PyTorch 2.4.1 + CUDA 12.4) による。

**通し** (移植先のサービスだけで ① -> ⑥。入力フレーム 79 枚):

| 段 | 元の出力との比較 |
|---|---|
| ① | 5322 / 5372 点が一致 (ずれた 50 点、最大 7.81 px、平均 0.026 px) |
| ② | 69 / 79 枚が完全に一致 (残り 10 枚は ① の差による。quad の頂点の差は最大 4.10 px、一辺の差は最大 1.62 px) |
| ③ | 一致しない画素 0.053%。完全に一致 34 / 79 枚 |
| ⑤ | 37 / 79 枚が完全に一致。一致しない画素 5.9% (ほとんどは 1 階調。②③ の差で部位の平均色がわずかに変わり、SDEdit の結果に及ぶ) |
| ⑥ 最終出力 | 37 / 79 枚が完全に一致。一致しない画素 1.26%、差の最大 173、平均 0.017 |

### フェーズ 4: 通し (`pipeline`) と入口の確認 (2026-11-27)

入力は `input/<シーン名>/` に置く (`input/sample04-12119910/` は元リポジトリの `input/sample04-12119910/` と 79 枚とも sha256 が同じ)。

| 確認 | 結果 |
|---|---|
| `pipeline` (sample04、79 枚、4 分 16 秒) と元の出力 | ① 5323 / 5372 点が一致 (最大 5 px)。② 68 / 79 枚、⑤ 37 / 79 枚、⑥ 37 / 79 枚が完全に一致 (⑥ の一致しない画素 1.09%)。フェーズ 3 の通しと同程度 |
| 同じ入力で `pipeline` を 2 回 | ① の 51 点 (最大 7.81 px) が違い、②③⑤⑥ はどれも 69 / 79 枚が一致。② が同じ 69 枚は ③〜⑥ も同じで、揺れは ① だけから来る |
| 段ごとのサービスで ②〜⑥ (`[opts]` で出力先を変え、`pipeline` の ① を入力) と `pipeline` の出力 | ②③⑤⑥ と `quads.json` が 79 / 79 枚で完全に一致 |
| 別のシーン (sample02、183 枚。`.env` の `HOST_INPUT_DIR`、`HOST_OUTPUT_DIR` だけを変更) | `pipeline` が最後まで動いた (9 分 6 秒)。1 枚 (`Cam4_frame_00037.jpg`) は顔が見つからず ② 以降で飛ばされ、`pasted/` は 182 枚 |
| `[opts]` の上書き | `pipeline --t0_ratio 0.4 --out_dir /data/output/convert_t04` で開始が番号 39 (t 394、√ᾱ 0.3260) になり、⑥ は `convert_t04/pasted/` に書かれた (採用構成の `convert/`、`pasted/` は変わらない)。`PASTE_OPTS="--feather 0.1"` が ⑥ の引数の最後に付いた。`align --mode ffhq --out_dir ...`、`landmarks --out ...`、`parse --input_dir ... --out_dir ...`、`convert --base_samples ... --out_dir ...`、`paste --run_dir ... --quads ...` も指定どおりに動いた。`STAGES=foo` は終了コード 2 で止まる |

sample04 の別の名前のシーン (`sample04-27243460`、`sample04-28602580`) は、入力フレームが sample04-12119910 と同じだったので、
別のシーンの確認には sample02 を使った。

sample02 の採用構成での見た目 (`output/sample02/pasted/grid.jpg`): 顔の全体が写るフレームは変換できているが、顔の一部が
フレームの外に出るフレーム (27 枚で顔が 2 個以上検出され、うち 22 枚はランドマークの信頼度が 1.0 未満) では、① が最大の枠を
選ぶため顔以外の位置に貼られるものがある。採用構成の ① の挙動 (元の `detect_face_landmarks.py` と同じ) で、移植では変えていない。
