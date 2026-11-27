# 第三者のライセンス

ルートの `LICENSE` (MIT、Ermon Group) は公式 SDEdit のもの。移植したファイルの一部は別のライセンスに従う。
出典 (コミットと sha256) は `patches/README.md` の表を参照。

| ファイル | ライセンス | 著作権者・出典 |
|---|---|---|
| `animeface/unet_mp.py` | **CC BY-NC-SA 4.0** | NVIDIA (NVlabs/edm2 の `training/networks_edm2.py` を改変。ファイル冒頭の表記) |
| `animeface/gaussian_diffusion.py`、`respace.py`、`nn.py`、`losses.py`、`animeface/model.py` と `animeface/image_io.py` の一部 | MIT | OpenAI (openai/guided-diffusion。jychoi118/ilvr_adm を経由) |
| `animeface/face_parsing.py` | MIT | tamochaman-nu (anime-face-alignment の `util/face_parsing.py`)。BiSeNet の構造は yakhyo/face-parsing (MIT、Copyright (c) 2024 Yakhyokhuja Valikhujaev) の `models/bisenet.py`・`models/resnet.py` を移植したもの (ファイル冒頭の表記) |
| face_alignment (pip の依存。同梱しない) | BSD-3-Clause | Adrian Bulat (1adrianb/face-alignment)。① で使う |

## CC BY-NC-SA 4.0 (`animeface/unet_mp.py`)

このファイルだけが CC BY-NC-SA 4.0 (http://creativecommons.org/licenses/by-nc-sa/4.0/) に従う。非営利の利用に限られ、
このファイルを改変して配布するときは同じライセンスにする必要がある。このファイルを含む限り、このリポジトリの変換を
営利目的に使うことはできない。ファイルは変更せずにコピーしている。

## MIT (OpenAI)

```
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MIT (anime-face-alignment、`animeface/face_parsing.py`)

```
MIT License

Copyright (c) 2026 tamochaman-nu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

BiSeNet の構造の元 (yakhyo/face-parsing) も MIT で、著作権表示は `Copyright (c) 2024 Yakhyokhuja Valikhujaev`
(`animeface/face_parsing.py` の冒頭に記載)。

## 同梱しないもの

- 重み (アニメ顔 DDPM のチェックポイント、face_alignment の 2DFAN4・s3fd、BiSeNet の resnet18.pt) は含めない。
- 学習・統計に使った画像 (FFHQ、アニメ顔のデータ) は含めない。`lists/face_color_stats.json` はそれらの色の統計 (平均・標準偏差・分位点) だけを持つ。
