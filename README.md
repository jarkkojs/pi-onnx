# pi-provider-onnx-community

Runs [Hugging Face onnx-community](https://huggingface.co/onnx-community) models locally inside
the Pi coding agent using [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers).

Implements a chat provider and several tool calls:

- `onnx_embed({ texts: string[] })`: array of vectors (and dimensionality).
- `onnx_classify({ text, labels? })`: top-K labels with scores; when `labels`
  is provided, runs zero-shot classification.
- `onnx_transcribe({ path, language?, task? })`: transcript text and segments.

## Install

```sh
pi install npm:pi-provider-onnx-community
```

## Configure

Copy `example-config.json` from this package as a starting point:

```sh
cp example-config.json ~/.pi/agent/onnx-community.json
```

### Top-level

| Field          | Type                                   | Default                           | Notes                                                     |
| -------------- | -------------------------------------- | --------------------------------- | --------------------------------------------------------- |
| `cacheDir`     | `string \| null`                       | `null` (HF default)               | Forwarded to `env.cacheDir`.                              |
| `device`       | `"cpu" \| "webgpu" \| "wasm" \| "gpu"` | `"cpu"`                           | onnxruntime execution provider hint.                      |
| `defaultDtype` | `Dtype`                                | `"q4"`                            | Per-model `dtype` overrides this.                         |
| `models`       | `ModelEntry[]`                         | `[Qwen2.5-Coder-0.5B-Instruct]`   | Each entry becomes a `onnx-community/<id>` chat model.    |
| `discovery`    | object                                 | enabled, limit 50                 | Append popular `onnx-community/*` models from the HF Hub. |
| `tools`        | object                                 | `embed` only                      | Toggles for `onnx_embed` / `_classify` / `_transcribe`.   |

`Dtype` is one of `"fp32"`, `"fp16"`, `"q8"`, `"int8"`, `"uint8"`, `"q4"`, `"bnb4"`, `"q4f16"`.

### `models[]`

| Field           | Type     | Default        | Notes                                                |
| --------------- | -------- | -------------- | ---------------------------------------------------- |
| `id`            | `string` | —              | Hugging Face repo path (`onnx-community/` prefixed). |
| `name`          | `string` | `id`           | Display name shown in the model picker.              |
| `contextWindow` | `number` | —              | Context window size in tokens.                       |
| `maxTokens`     | `number` | `1024`         | Default `max_new_tokens` for completions.            |
| `dtype`         | `Dtype`  | `defaultDtype` | Quantization for this model only.                    |

Only `id` is required; the `onnx-community/` prefix is added if missing.

Example:

```json
{
  "id": "onnx-community/Qwen3-0.6B-ONNX",
  "name": "Qwen3-0.6B (ONNX, q4)",
  "contextWindow": 32768,
  "maxTokens": 2048,
  "dtype": "q4"
}
```

### `discovery`

| Field          | Type            | Default                                                     | Notes                                   |
| -------------- | --------------- | ----------------------------------------------------------- | --------------------------------------- |
| `enabled`      | `boolean`       | `true`                                                      | Append discovered models to `models[]`. |
| `limit`        | `number`        | `50`                                                        | Per pipeline tag.                       |
| `pipelineTags` | `PipelineTag[]` | `["text-generation", "image-text-to-text", "any-to-any"]`   | Hugging Face pipeline tags to scan.     |

### `tools.embed`

| Field       | Type                        | Default                             | Notes                          |
| ----------- | --------------------------- | ----------------------------------- | ------------------------------ |
| `enabled`   | `boolean`                   | `true`                              | Toggles `onnx_embed`.          |
| `model`     | `string`                    | `onnx-community/all-MiniLM-L6-v2`   | Any feature-extraction model.  |
| `pooling`   | `"mean" \| "cls" \| "none"` | `"mean"`                            | Pooling strategy.              |
| `normalize` | `boolean`                   | `true`                              | L2-normalize output vectors.   |

### `tools.classify`

| Field     | Type      | Default                                                            | Notes                                |
| --------- | --------- | ------------------------------------------------------------------ | ------------------------------------ |
| `enabled` | `boolean` | `false`                                                            | Toggles `onnx_classify`.             |
| `model`   | `string`  | `onnx-community/distilbert-base-uncased-finetuned-sst-2-english`   | Classifier or NLI model (zero-shot). |
| `topK`    | `number`  | `5`                                                                | Maximum labels returned.             |

### `tools.transcribe`

| Field      | Type                            | Default                       | Notes                                |
| ---------- | ------------------------------- | ----------------------------- | ------------------------------------ |
| `enabled`  | `boolean`                       | `false`                       | Toggles `onnx_transcribe`.           |
| `model`    | `string`                        | `onnx-community/whisper-tiny` | Any ASR model.                       |
| `language` | `string \| null`                | `null`                        | Default language hint (e.g. `"en"`). |
| `task`     | `"transcribe" \| "translate"`   | `"transcribe"`                | Default ASR task.                    |

## Limitations

- No tool calling support for ONNX chat models.
- Tokens are approximated from the tokenizer.
- First call to a model blocks while weights download.
- `onnx_transcribe` shells out to `ffmpeg` (must be on `PATH`) to decode the
  input audio file to a `Float32Array` before inference.

## License

`pi-provider-onnx-community` is licensed under `MIT`. See [LICENSE](LICENSE)
for more information.
