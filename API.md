# MiniMax-H3 (ComfyUI) Worker API

RunPod Serverless エンドポイントのリクエスト／レスポンス契約です。

エンドポイント: `POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run`（非同期）または `/runsync`
認証: `Authorization: Bearer <RUNPOD_API_KEY>`

---

## リクエスト

### 入力フィールド

| フィールド | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `prompt` | string | — | **必須**。2000文字以内 |
| `image_url` | string | — | 先頭キーフレームの URL。指定すると I2V |
| `image_base64` | string | — | 先頭キーフレーム（raw base64 または `data:` URL）。`image_url` との併用不可 |
| `last_image_url` | string | — | 終端キーフレームの URL。単独指定も可 |
| `last_image_base64` | string | — | 終端キーフレーム。`last_image_url` との併用不可 |
| `seconds` | int | `5` | **5〜15**（エンドポイントの `MAX_SECONDS` で更に絞られる） |
| `fps` | int | `24` | `24` のみ。MiniMax-H3 は 24fps 固定 |
| `steps` | int | `4`（`TURBO_MODE` 既定オン時） | 1〜60。NFE（サンプラーのステップ数）。8 にすると毛並みなど細部が少しシャープになり、時間は約1.7倍 |
| `seed` | int | `-1` | `-1` でランダム |
| `resolution` | string | — | プリセット名（下表）。指定すると `width`/`height` を上書き |
| `width` / `height` | int | `960` / `544` | 32の倍数、各128以上、積が `1032192` 以内 |
| `mute` | bool | `false` | `true` で無音 mp4 |
| `video_vae` | string | `"fp16"`（環境変数 `VIDEO_VAE_DEFAULT` で変更可） | `"fp16"` / `"int8"`。`"int8"` は Comfy-Org の INT8+ConvRot video VAE でデコードする（ComfyUI 0.36 の専用カーネルで速い）。Model Cache のスナップショットに INT8 VAE が無い場合はエラー |
| `attention` | string | `"default"`（環境変数 `ATTENTION_DEFAULT` で変更可） | `"default"` / `"sage"` / `"ck_int8"`。attention の計算方式（下の注意を参照） |
| `negative_prompt` | string | `""` | **受理されるが無視される**（guidance 蒸留済み） |
| `cfg_scale` | float | `1.0` | **受理されるが無視される**（同上） |
| `image_strength` | float | `1.0` | **受理されるが無視される**（キーフレームに強度つまみが無い） |
| `output` | object | `{}` | 出力先設定（下表） |
| `batch` | array | — | バッチジョブ。他のフィールドは全アイテム共通の既定値になる |

`negative_prompt` / `cfg_scale` / `image_strength` は検証（型・範囲）を通したうえで捨てられ、送った場合はレスポンスの `meta.ignored_params` に列挙されます。

### `output`

| フィールド | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `bucket` | string | `S3_BUCKET_DEFAULT` | アップロード先バケット |
| `prefix` | string | `"minimax-h3-comfy/"` | キーの前置き。実キーは `<prefix>YYYY/MM/DD/<job_id>.mp4` |
| `public` | bool | `true` | `false` かつ `ttl_seconds > 0` で presigned URL |
| `ttl_seconds` | int | `0` | presigned URL の有効秒数 |

### 解像度プリセット

| 名前 | 解像度 | 備考 |
| --- | --- | --- |
| `768p` | 1344×768 | ネイティブ。1,032,192 px = モデル上限ちょうど |
| `768p-portrait` | 768×1344 | |
| `720p` / `720p-portrait` | 1344×768 / 768×1344 | 他ワーカー互換のエイリアス |
| `square` | 992×992 | |
| `480p` | 960×544 | |
| `480p-portrait` | 544×960 | |

### フレーム数の丸め

`num_frames`（レスポンスでは `meta.frames`）は video VAE がデコードできる `17n + 5` に丸められ、結果の尺が 5〜15 秒に収まるよう調整されます（`src/workflow.py` の `frame_length()`）。

| `seconds` | `meta.frames` | 実尺 |
| --- | --- | --- |
| 5 | 124 | 5.17 s |
| 8 | 192 | 8.00 s |
| 10 | 243 | 10.13 s |
| 15 | 345 | 14.38 s |

---

## レスポンス

### 成功（単発）

```json
{
  "status": "completed",
  "video_url": "https://cdn.example.com/minimax-h3-comfy/2026/09/18/abc123.mp4",
  "meta": {
    "mode": "i2v",
    "seconds": 5,
    "fps": 24,
    "frames": 124,
    "seed": 42,
    "steps": 8,
    "cfg_scale": 1.0,
    "width": 960,
    "height": 544,
    "model": "minimax-h3-comfy-turbo-8step",
    "has_audio": true,
    "audio_sampling_rate": 32000,
    "attention": "default",
    "video_vae": "fp16",
    "ignored_params": []
  },
  "timing": {
    "preprocess_s": 0.03,
    "inference_s": 184.2,
    "upload_s": 0.9,
    "total_s": 185.1
  }
}
```

### 成功（バッチ）

```json
{
  "status": "completed",
  "results": [
    { "status": "completed", "video_url": "...", "meta": {...}, "timing": {...} },
    { "status": "failed", "error": {...} }
  ],
  "meta": { "model": "minimax-h3-comfy-turbo-8step" }
}
```

`results[i]` は `batch[i]` に対応します（順序は保持されます）。

### 失敗

RunPod のジョブ自体が `"status": "FAILED"` になり、トップレベルの `error` に文字列、`output` に構造化された詳細が入ります:

```json
{
  "status": "FAILED",
  "error": "ValidationError: seconds must be between 5 and 10.",
  "output": {
    "status": "failed",
    "error_detail": {
      "type": "ValidationError",
      "message": "seconds must be between 5 and 10.",
      "details": { "hint": "...", "trace_id": "..." }
    }
  }
}
```

`error` を dict で返すと runpod SDK の `/job-done` 送信が 400 で弾かれて結果ごと失われるため、文字列にしています。

`type`: `ValidationError` / `DownloadError` / `InferenceError` / `UploadError` / `InternalError`。
`InternalError` のときだけ `details.traceback` が付きます。`InferenceError` はワーカー内部での ComfyUI 実行失敗（起動失敗、グラフ検証エラー、ノード実行時エラー、タイムアウト）を包みます。

---

## 例

### Text-to-Video

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {
        "prompt": "A red fox trotting through a snowy pine forest, snow crunching underfoot",
        "resolution": "480p",
        "seconds": 5
      }}'
```

### 先頭＋終端キーフレーム

```json
{
  "input": {
    "image_url": "https://example.com/start.png",
    "last_image_url": "https://example.com/end.png",
    "prompt": "the camera pushes in as the light shifts to dusk",
    "resolution": "768p",
    "seconds": 8,
    "seed": 42
  }
}
```

### 無音

```json
{ "input": { "prompt": "...", "mute": true } }
```

### バッチ

```json
{
  "input": {
    "resolution": "480p",
    "seconds": 5,
    "batch": [
      { "prompt": "scene one", "seed": 1 },
      { "prompt": "scene two", "seed": 2 }
    ]
  }
}
```

---

## R2V（Reference-to-Video）エンドポイント

同じイメージを `WORKER_MODE=r2v` でデプロイした**別エンドポイント**が、複数の参照画像から動画を作る R2V を受け付けます（Model フィールドは `fkyyy/minimax-h3-comfy-r2v-slim`）。i2v エンドポイントに `ref_images` を送ると `ValidationError` になります。

### 追加・変更されるフィールド（r2v エンドポイントのみ）

| フィールド | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `ref_images` | array | `[]` | 参照画像、最大 **9 枚**。要素は `{"url": "..."}` / `{"base64": "..."}`、または文字列（`http(s)://` なら URL、それ以外は raw base64 / `data:` URL）。各 10MB まで |
| `turbo` | bool | `false` | `true` で ref2v Turbo LoRA（lightx2v 8-step v1.1 768p）を使う。速いが参照の効きが弱まる |
| `steps` | int | turbo 時 `8` / 通常 `20` | 1〜60 |
| `ref_image_size` | string | `"match"` | `match` は参照を生成サイズの面積に縮小。`max` は短辺 2048px まで保持（数倍遅くなることがある） |
| `scheduler` | string | `"simple"` | `simple` / `beta` / `normal`。公式テンプレートの注記では参照の多いプロンプトで `beta`/`normal` が良いことが多い |

- `image_url` / `last_image_url` などのキーフレームは**使えません**（R2V ノードに先頭・終端フレーム入力が無いため）。前のクリップから続けたい場合は、その最終フレームを参照画像の 1 枚として渡し、プロンプトで指示します。
- i2v エンドポイントに `turbo` / `ref_image_size` / `scheduler` を送った場合は無視され、`meta.ignored_params` に入ります。

### プロンプトでの参照の書き方

`ref_images` の順に `<Picture 1>` 〜 `<Picture 9>`（1 始まり）で参照します。

```json
{
  "input": {
    "prompt": "<Picture 1> the female reporter sneaks behind <Picture 2> the bear mascot down a concrete corridor.",
    "ref_images": [
      {"url": "https://example.com/reporter.png"},
      {"url": "https://example.com/bear.png"}
    ],
    "resolution": "768p-portrait",
    "seconds": 10
  }
}
```

### レスポンスの追加 meta

`mode` は `"r2v"`。加えて `ref_count`、`turbo`、`ref_image_size`、`scheduler` が入ります。`model` は `minimax-h3-comfy-r2v`（turbo 時は `-turbo-8step` 付き）。

---

## 注意

- **`timing.inference_s` には、ComfyUI 側のグラフ実行全体（サンプリング + デコード + mux）が含まれます。** 動画のエンコードは ComfyUI の `CreateVideo` / `SaveVideo` ノードが行い、このワーカー自身は行いません。
- **`steps` はサンプラーのステップ数（NFE）そのままです。** 内部で +1 などの変換はしません。
- **`attention` について。** そのジョブだけ attention の計算方式を差し替えます（自前のカスタムノード `comfy_nodes/attention_backend`。ComfyUI の `optimized_attention_override` を使うので、デプロイし直さずにリクエストごとに切り替えられます）。`"sage"` は SageAttention 1.0.6（Triton 実装、QK を INT8 で計算）、`"ck_int8"` は ComfyUI 0.36 に同梱の Comfy Kitchen INT8 attention です。`"default"` は PyTorch SDPA。どちらも量子化 attention なので、画質と音声への影響は未検証です。速度と品質を同じ seed で比べてから既定値を決めてください。使えない環境（パッケージの import 失敗や、GPU が非対応など）では、黙って既定に戻さずジョブがエラーになります。
- **音声を切っても速くなりません。** MiniMax-H3 は映像 latent と音声 latent を同一の packed sequence で同時にデノイズします。`mute` で省けるのは audio VAE のデコードと mux だけです。
- **cold start は分単位です。** ComfyUI プロセスの起動 + Model Cache スナップショットのリンクに加え、最初のジョブは重みの実ロード時間も乗ります。ポーリングのタイムアウトは余裕を持たせてください。
- **`TURBO_MODE`（既定オン）はデプロイ時の設定で、リクエストからは切り替えられません。** オフにすると 20 ステップ前後のベース挙動になりますが、この経路は実地未検証です。
