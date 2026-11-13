# MiniMax-H3 — RunPod Serverless Worker (ComfyUI 版)

音声つき動画生成モデル [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) を、RunPod Serverless 上で API として動かすワーカーです。

ComfyUI 本体の MiniMax H3 ネイティブサポート（Comfy-Org/ComfyUI #15224、0.30.0+）と、Comfy-Org が配布している量子化済みチェックポイントを使います。ワーカーは handler の中で ComfyUI を別プロセスとして起動し、ワークフローを組み立てて投げます。クライアントには ComfyUI のノードグラフを見せず、`prompt` / `image_url` / `seconds` のような単純なフィールドだけを受け取ります。

- T2V（テキストから）、I2V（先頭・終端フレームを指定）、R2V（参照画像最大 9 枚）
- 映像と音声（32kHz ステレオ）を同時に生成し、音声つき mp4 を S3 互換ストレージにアップロードして URL を返す
- モデルの重みは Docker イメージに入れず、RunPod の Model Cache から読む

リクエストとレスポンスの形は [`API.md`](API.md) を参照してください。

## モデル（Model Cache 用の痩せフォーク）

`Comfy-Org/MiniMax-H3`（447 GB）は Model Cache にそのまま載らないため、必要なファイルだけを抜き出したリポジトリを用意しています。

| 用途 | リポジトリ | サイズ |
| --- | --- | ---: |
| T2V / I2V | [`fkyyy/MiniMax-H3-comfy-slim`](https://huggingface.co/fkyyy/MiniMax-H3-comfy-slim) | 約 41 GB |
| R2V | [`fkyyy/MiniMax-H3-comfy-r2v-slim`](https://huggingface.co/fkyyy/MiniMax-H3-comfy-r2v-slim) | 約 41 GB |

T2V / I2V 用の中身:

| ファイル | サイズ |
| --- | ---: |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.53 GB |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.61 GB |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 4.85 GB |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors`（`video_vae: "int8"` 用） | 2.62 GB |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.56 GB |
| `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`（8 ステップ蒸留 LoRA） | 1.82 GB |

R2V 用は、本体が `minimax_h3_ref2va_pruned_int8_convrot.safetensors` に、LoRA が `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors` に替わり、テキストエンコーダと VAE は共通です。

> **注意:** 上の公開リポジトリは、今後もずっと公開し続けることを保証するものではありません。予告なく非公開・削除・内容の変更をする場合があります。長く安定して使いたい場合は、[`tools/make_comfy_slim_fork.py`](tools/make_comfy_slim_fork.py) で自分の Hugging Face アカウントに同じ構成のリポジトリを作り、そちらを使ってください。
>
> ```sh
> pip install -U huggingface_hub hf_transfer
> export HF_TOKEN=hf_...   # 書き込み権限のあるトークン
> python tools/make_comfy_slim_fork.py --dst <your-hf-name>/MiniMax-H3-comfy-slim --public
> python tools/make_comfy_slim_fork.py --workflow r2v --dst <your-hf-name>/MiniMax-H3-comfy-r2v-slim --public
> ```
>
> 1 ファイルずつダウンロード→アップロード→削除するので、ローカルのディスクは最大のファイル 1 つ分（約 20 GB）あれば足ります。

## デプロイ

### T2V / I2V エンドポイント

1. RunPod コンソールで `Serverless → New Endpoint → GitHub Repo` を選び、このリポジトリを接続します（Branch: `main`、Dockerfile: `./Dockerfile`）。
2. 次のように設定します。
   - GPU: H100 80GB（CUDA 13.0 に対応したドライバのもの。PyTorch の cu130 版を使うため）
   - Model: `fkyyy/minimax-h3-comfy-slim`（**小文字で入力**。下の「Model 欄は小文字で」を参照）
   - Container Disk: 20 GB
   - Execution Timeout: 1800 秒以上（最初のジョブはモデルの読み込みが乗るため）
   - Max Workers: 1、Active Workers: 0
   - Data Center: 米国・EU・英国・韓国を外す（下の「ライセンス」を参照）
   - 環境変数: 下の表の `S3_*` 一式
3. Model Cache のダウンロードが終わると使えるようになります。

### R2V エンドポイント

同じイメージで、もう 1 つエンドポイントを作ります。違うのは次の 2 点だけです。

- Model: `fkyyy/minimax-h3-comfy-r2v-slim`
- 環境変数: `WORKER_MODE=r2v`

R2V は I2V より時間がかかるため、`COMFY_INFERENCE_TIMEOUT_S` の既定は 1800 秒です。エンドポイントの Execution Timeout もそれ以上にしてください。

### Model 欄は小文字で

Model Cache は、Model 欄に入力した文字列そのままでキャッシュのフォルダ名を作ります。Hugging Face 上の表記（大文字混在）で入力すると、ワーカーがキャッシュを見つけられないことがあります。Model 欄は小文字にし、自分のフォークを使う場合は環境変数 `HF_REPO` にも同じ小文字の名前を設定してください。

また、Model Cache は Model 欄を設定した時点のリポジトリを取り込み、同じ名前を入れ直しても取り直しません。自分でフォークを作る場合は、アップロードが全部終わってからエンドポイントを作ってください。

## 環境変数

| 変数 | 既定値 | 説明 |
| --- | --- | --- |
| `S3_ENDPOINT` | — | S3 互換ストレージのエンドポイント（R2 / B2 など。AWS S3 なら空） |
| `S3_REGION` | `auto` | リージョン |
| `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | — | アップロード用の認証情報 |
| `S3_BUCKET_DEFAULT` | — | アップロード先のバケット |
| `S3_PUBLIC_BASE_URL` | — | 動画の公開 URL のベース（R2 の公開ドメインなど）。未設定だと、返る URL がブラウザで開けないことがあります |
| `WORKER_MODE` | `i2v` | `i2v`（T2V / I2V）または `r2v` |
| `HF_REPO` | モードに応じた公開フォーク | Model 欄と同じ、小文字のリポジトリ名 |
| `MAX_SECONDS` | `10` | 受け付ける最大の秒数（モデルの上限は 15） |
| `MIN_SECONDS` | `5` | 受け付ける最小の秒数（1 以上）。モデルの学習範囲は 5〜15 秒で、5 未満は範囲外の実験用。品質は保証されません |
| `TURBO_MODE` | `1` | I2V / T2V で 8 ステップ蒸留 LoRA を使う |
| `DEFAULT_STEPS` | `4`（Turbo 時） | `steps` を省略したときのステップ数 |
| `R2V_TURBO_DEFAULT` | `0` | R2V で `turbo` を省略したときの値 |
| `ATTENTION_DEFAULT` | `default` | `default` / `sage` / `ck_int8` |
| `VIDEO_VAE_DEFAULT` | `fp16` | `fp16` / `int8` |
| `MAX_BATCH_SIZE` | `4` | 1 ジョブの `batch` の最大件数 |
| `COMFY_INFERENCE_TIMEOUT_S` | `600`（R2V は `1800`） | 1 本あたりの生成の上限秒数 |
| `COMFY_DISABLE_COMPILER` | `0` | ComfyUI のモデルコンパイラを無効にする |

## 速度とコストの目安

H100 での実測（ワーカーが温まった状態、768p・5 秒・8 ステップ・`attention: "ck_int8"`）:

| モード | 処理時間 |
| --- | ---: |
| T2V | 約 66 秒 |
| I2V | 約 75 秒 |
| R2V（Turbo、参照画像 2 枚） | 約 72〜81 秒 |

冷えた状態からの起動（ComfyUI の起動とモデルの読み込み）には、別に数分かかります。複数本を作るときは、ジョブを並列に投げずに 1 つのジョブの `batch` にまとめると、ワーカーが余分に起動しません。

`steps` を 4 にすると時間は約 6 割になります。`attention` の `sage` / `ck_int8` は量子化 attention なので、画質と音声への影響は同じシードで比べてから使ってください。

## ファイル構成

```
src/
  handler.py       RunPod のエントリポイント。ComfyUI をサブプロセスで起動し、ジョブごとに投げる
  config.py        環境変数ベースの設定
  model_link.py    Model Cache のスナップショット → ComfyUI の models/ へシンボリックリンク
  workflow.py      ワークフロー（API 形式）を Python の dict で組み立てる
  comfy_client.py  ComfyUI の HTTP / WebSocket クライアント（/prompt, /history, /view）
  validator.py     入力検証
  image.py         キーフレーム・参照画像の取得
  security.py      画像 URL の SSRF 対策
  storage.py       S3 互換ストレージへのアップロード
  watchdog.py      Pod で動かすとき（SERVE_MODE=api）の自己終了
  errors.py        エラー型
comfy_nodes/
  attention_backend/        リクエストごとに attention の実装を切り替えるカスタムノード
tools/
  make_comfy_slim_fork.py   Model Cache 用の痩せフォークを作るスクリプト
Dockerfile                  comfy-cli で ComfyUI を入れる。モデルは焼き込まない
API.md                      リクエスト / レスポンスの仕様
```

## ライセンス

- このリポジトリのコードは [GPL-3.0](LICENSE) です。`comfy_nodes/` のカスタムノードは ComfyUI（GPL-3.0）のモジュールを直接使っているため、リポジトリ全体を GPL-3.0 にしています。
- モデルの重みは [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) に従います。このライセンスは米国・EU・英国・韓国でのデプロイを対象外としているので、エンドポイントのデータセンターからこれらの地域を外してください。商用利用などの条件は、利用前にライセンス本文を確認してください。
- 8 ステップ蒸留 LoRA は [lightx2v](https://huggingface.co/lightx2v) によるもので、Comfy-Org が ComfyUI 形式に変換したものを使っています。利用条件は配布元を確認してください。

## 参考

- ComfyUI の MiniMax H3 対応: https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui
- ComfyUI 公式チュートリアル: https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- モデル配布元: https://huggingface.co/Comfy-Org/MiniMax-H3
