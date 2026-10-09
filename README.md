# Evaluating LLM Creativity through Long-Tail Performance

This repository contains the codebase for the paper **Evaluating LLM Creativity through Long-Tail Performance**. It provides:

1. efficient, general computation of n-gram novelty and Creativity Index (CI); and
2. a configurable framework for generating and evaluating model performance across tasks, model backends, creativity conditions, and promoting methods.


## Installation

Clone with the embedded upstream repositories:

```bash
git clone --recurse-submodules <repository-url>
cd longtail-creativity
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Install only the optional components that you need:

```bash
pip install -e '.[hf]'        # Hugging Face inference
pip install -e '.[vllm]'      # vLLM inference
pip install -e '.[api]'       # OpenAI-compatible APIs
pip install -e '.[ci]'        # CI tokenization and local count server
pip install -e '.[reward]'    # Hugging Face reward-model evaluation
pip install -e '.[datasets]'  # Hugging Face datasets
```

## Quick start

This example shows how to run one model on one task under a specific creativity setting, evaluate task performance, and then evaluate novelty with CI and n-gram novelty.

Install the Hugging Face and CI dependencies:

```bash
pip install -e '.[hf,ci]'
```

Create `quickstart-task.jsonl` with one task:

```json
{"id": "story-1", "prompt": "Write a short story about a lighthouse that remembers every ship it has seen."}
```

Create `quickstart.yaml`:

```yaml
model:
  backend: hf
  model_id: Qwen/Qwen2.5-0.5B-Instruct

task:
  source: local
  path: quickstart-task.jsonl
  prompt_field: prompt
  id_field: id
  answer_field: null

taxonomy:
  transformation: inherent
  depth: conceptual
  focus: outcome
  reference: historical
  prompt_template: "{prompt}"

generation:
  do_sample: true
  temperature: 0.7
  top_p: 0.95
  max_tokens: 128
  n: 1

run:
  output_dir: outputs/quickstart
  seed: 42
  batch_size: 1
```

When the configuration omits `method`, generation uses the unmodified `direct` baseline (i.e., the vanilla model). Run generation:

```bash
longtail-creativity generate quickstart.yaml
```

The default and recommended evaluation path uses a self-hosted Infini-gram Mini server. Download/build the engine and obtain an index, then start the server in a separate terminal:

```bash
longtail-creativity serve-index \
  --engine-root /path/to/infini-gram-mini \
  --index-dir /path/to/index
```

Startup loads the index into RAM by default. Wait for `Index ready` and verify `http://<server-host>:5555/health`. The server writes its address to `runtime/ci-server.address`. In another terminal, compute CI and n-gram novelty; no endpoint option is needed for this default path:

```bash
longtail-creativity novelty \
  --input outputs/quickstart/outputs.jsonl \
  --window 0 \
  --cache runtime/infini-gram.sqlite
```

The per-example metrics are written to `outputs/quickstart/novelty_metrics.jsonl`, with an aggregate summary in `outputs/quickstart/novelty_metrics_summary.json`.

The same command can evaluate input/prompt novelty while retaining the same task IDs and tokenization settings:

```bash
longtail-creativity novelty \
  --input outputs/quickstart/outputs.jsonl \
  --output outputs/quickstart/input_novelty_metrics.jsonl \
  --text-field prompt \
  --window 0 \
  --cache runtime/infini-gram.sqlite
```

## Repository layout

```text
longtail_creativity/
  backends/          model inference backends
  evaluation/        task-scoring interfaces and generic scorers
  novelty/           CI, n-gram novelty, clients, batching, and count server
  likelihood.py      input and conditional-output perplexity evaluation
  normalization.py   final-answer extraction and reasoning separation
  config.py          validated YAML configuration
  generation.py      generation orchestrator
  methods.py         external method plugin interface
  tasks.py           dataset-agnostic task loading
  taxonomy.py        paper-taxonomy condition handling
configs/
  example.yaml       infrastructure-neutral example configuration
scripts/
  launch_ci_server.sbatch
lm-evaluation-harness/          upstream submodule
third_party/creativity_index/   upstream submodule
tests/
```

## Dataset format and sources

No dataset is committed to this repository. A local JSONL file can use the following minimal schema:

```json
{"id": "example-1", "prompt": "...", "answer": "..."}
```

Field names are configurable. Hugging Face datasets can be loaded directly by setting `task.source: huggingface`, `task.path`, `task.split`, and the field mapping in the run YAML.

Datasets used or supported during the project can be obtained from their original distribution pages:

| Dataset | Download/source |
| --- | --- |
| WritingPrompts Curated | [Hugging Face](https://huggingface.co/datasets/euclaise/WritingPrompts_curated) |
| Annotated Alpaca prompts | [Hugging Face](https://huggingface.co/datasets/agentlans/alpaca-prompts-annotated) |
| Arena-Hard | [GitHub](https://github.com/lmarena/arena-hard-auto) |
| StoryGen | [Meta AI download](https://dl.fbaipublicfiles.com/doc-storygen/story_annotation_v2.json) |
| CS4 | [GitHub](https://github.com/anirudhlakkaraju/cs4_benchmark) |
| DeepMath-Creative | [GitHub](https://github.com/DeepMathLLM/DeepMath/tree/main/DeepMath-Creative-data) |
| IFEval | [Hugging Face](https://huggingface.co/datasets/google/IFEval) |
| AIME 1983–2024 | [Hugging Face](https://huggingface.co/datasets/gneubig/aime-1983-2024) |
| MATH | [GitHub](https://github.com/hendrycks/math) |
| BIG-Bench Hard | [GitHub](https://github.com/suzgunmirac/BIG-Bench-Hard) |
| CommonsenseQA | [Hugging Face](https://huggingface.co/datasets/tau/commonsense_qa) |
| GPQA | [Hugging Face](https://huggingface.co/datasets/Idavidrein/gpqa) |


## Generate model outputs

Copy and edit [`configs/example.yaml`](configs/example.yaml). The taxonomy block records the experimental condition. Its optional `prompt_template` must contain `{prompt}`; exact reproduction prompts can therefore be supplied without hard-coding experiment-specific language in the framework.

```bash
longtail-creativity generate configs/example.yaml
```

Outputs are written to the configured runtime directory. Re-running with `run.resume: true` skips completed task IDs.

Generation preserves each untouched backend response in `raw_output` and puts the scoreable response in `final_answer` and `text`. By default, complete `<think>...</think>` and `<analysis>...</analysis>` blocks are separated from the final answer. Extraction is configurable without a model-name registry:

```yaml
normalization:
  plugin: reasoning_tags  # or identity / package.module:NormalizerClass
  options:
    strip_tokens: ["<end_of_turn>"]
    final_answer_pattern: "FINAL ANSWER:\\s*(.*)"
```

Each output records normalization provenance and any extracted reasoning separately. Task scoring and novelty evaluation prefer `final_answer`, so reasoning text is not silently included in outcome metrics.


```yaml
model:
  backend: api
  model_id: provider/model-name
  options:
    base_url: https://provider.example/v1
    api_key_env: PROVIDER_API_KEY
```

## Add a generation method

Concrete improvement methods are deliberately not bundled. Implement `longtail_creativity.methods.GenerationMethod` in another package and reference it in YAML:

```yaml
method:
  plugin: my_package.methods:MyMethod
  options:
    parameter: value
```

This keeps the shared framework independent of the paper's method-specific prompts and checkpoints.

## Evaluate task performance

```bash
longtail-creativity evaluate configs/example.yaml \
  --outputs outputs/example/outputs.jsonl \
  --scorer exact_match
```

Use `numeric_match` for simple numeric answers or provide an external scorer as `package.module:ScorerClass`.

For reference-free open-ended tasks such as creative writing, install the reward-model dependencies and select the batched `reward_model` scorer:

```bash
pip install -e '.[reward]'

longtail-creativity evaluate configs/example.yaml \
  --outputs outputs/example/outputs.jsonl \
  --scorer reward_model \
  --scorer-options '{"batch_size": 4, "max_length": 4096}'
```

The default example checkpoint is `Skywork/Skywork-Reward-V2-Llama-3.1-8B`. The evaluator loads it once, formats each normalized `(prompt, final_answer)` pair as user and assistant turns, and runs batched sequence-classification inference. It does not require dataset answers. The resulting scalar reward is written as `score`; higher is better. Length-limited or missing final answers remain unscored.

Any compatible Hugging Face reward model with one scalar output logit can be selected without changing code:

```bash
longtail-creativity evaluate configs/example.yaml \
  --outputs outputs/example/outputs.jsonl \
  --scorer reward_model \
  --scorer-options '{"model_id": "organization/reward-model", "device": "cuda", "dtype": "bfloat16", "batch_size": 8}'
```

Supported options include `model_id`, `tokenizer_id`, `revision`, `device`, `device_map`, `dtype`, `attn_implementation`, `batch_size`, `max_length`, `trust_remote_code`, `tokenizer_options`, and `model_options`. External scorers can also override `score_batch()` and set `requires_answer = False` for other reference-free evaluators.

## Creativity Index and n-gram novelty

The implementation first finds the longest reference-corpus match beginning at every token. Exact-match suffix reuse makes the number of corpus count queries linear in text length, assuming the count service has substring-consistent semantics. A heap-based sweep computes per-token familiarity horizons in `O(N log N)` local time. CI and every requested n-gram novelty score reuse this same longest-match array, so adding n values requires no additional corpus queries.

Self-hosted Infini-gram Mini is the default backend. In our testing, a warmed local index was substantially faster than the rate-limited public API. Once a server has written `runtime/ci-server.address`, score an `outputs.jsonl` file without specifying an endpoint:

```bash
longtail-creativity novelty \
  --input outputs/example/outputs.jsonl \
  --cache runtime/infini-gram.sqlite \
  --resume
```

Use `--text-field prompt` to evaluate input novelty and `--sample-index` to choose among multiple generations. The SQLite cache is optional but strongly recommended. With `--resume`, completed task/variant IDs are skipped. Results are appended by one writer; failed or stalled endpoints are quarantined and their in-flight item is retried on a healthy endpoint. If no healthy endpoint remains, the command exits with the already written progress intact. Network failures are never interpreted as zero corpus occurrences, which prevents silent novelty inflation. `--per-item-timeout` and `--watchdog-interval` control stall detection. Length-limited generations are skipped by default; use `--include-length-truncated` only when partial outputs are intentional. Use a distinct, stable `--cache-namespace` for each reference corpus; it also protects resumed runs from mixing incompatible evaluation settings.

### Local count server

For high-throughput evaluation, download and build [Infini-gram Mini](https://github.com/xuhaoxh/infini-gram-mini), then obtain or construct an index following its upstream instructions:

```bash
longtail-creativity serve-index \
  --engine-root /path/to/infini-gram-mini \
  --index-dir /path/to/index \
  --port 5555
```

The index is loaded into RAM before the address file is written. RAM loading is enabled by default for retrieval throughput; `--no-load-to-ram` is available for constrained environments but is not the recommended performance setting.

One or more servers can be used in parallel:

```bash
longtail-creativity novelty \
  --input outputs/example/outputs.jsonl \
  --servers http://host-a:5555 http://host-b:5555 \
  --cache runtime/local-index.sqlite
```

Use `--server-address-file` to select a different address file. The file may contain multiple URLs, one per line. If sufficient CPU/RAM is unavailable for a local index, explicitly fall back to a public service:

```bash
longtail-creativity novelty \
  --input outputs/example/outputs.jsonl \
  --service infini-gram \
  --cache runtime/public-api.sqlite \
  --resume
```

The public endpoint is convenient but rate limited, so it is intended as a fallback rather than the default high-throughput evaluation path.

The Slurm entry point contains no partition, account, hostname, absolute environment path, or server ID. Supply site-specific resources when submitting it:

```bash
sbatch \
  --export=ALL,INFINI_GRAM_MINI_ROOT=/path/to/engine,CI_INDEX_DIR=/path/to/index \
  scripts/launch_ci_server.sbatch
```

## Likelihood and perplexity tailness

The optional vLLM evaluator computes full-input perplexity, content-only input perplexity, and conditional-output perplexity from generation records:

```bash
longtail-creativity likelihood \
  --input outputs/example/outputs.jsonl \
  --model organization/scoring-model \
  --apply-chat-template \
  --max-model-length 32768
```

For instruct models, `--apply-chat-template` renders a user/assistant conversation. Otherwise, use `--plain-template` with exactly one `{prompt}` and `{answer}` placeholder. Token boundaries are derived by longest-common-prefix comparison against the complete tokenized sequence, avoiding BPE boundary errors from separately tokenizing prompt and answer. Add `--truncate-output-tokens K` to report an additional length-controlled conditional perplexity over the first `K` scored output tokens.

## Embedded upstream repositories

- [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) is retained for benchmark compatibility and reproduction work.
- [`third_party/creativity_index`](https://github.com/GXimingLu/creativity_index) is retained as the upstream implementation associated with the original Creativity Index paper.
