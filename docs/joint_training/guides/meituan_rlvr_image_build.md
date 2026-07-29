# Meituan RLVR Image Build and Environment Handoff

本文是当前 `codex/rebuttal-rlvr` 分支交给美团同事的镜像构建清单。当前直接目标是：加载 ordinary-SFT / offline-WDL-SFT 权重，在 H20 上运行相同的纯 GRPO。主镜像覆盖数学 parser/scorer、代码任务 online reward 和三套 pinned harness 源码；BigCodeBench 全量 task-library 环境与 Python 3.12/Torch 训练 ABI 冲突，必须使用单独的 evaluator image，不能为了“一镜到底”污染训练环境。

## 一页结论

1. **首选上游**：美团内部镜像仓库中、以不可变 digest 引用的 `linux/amd64` `nvidia/cuda:12.6.3-devel-ubuntu22.04` 等价镜像。必须是 `devel`，因为需要编译 `flash-attn`；不要用只有 runtime 的 vLLM serving 镜像。
2. **当前分支的兼容基线**：Python `3.12.13`、PyTorch `2.9.1+cu126`、vLLM `0.12.0`、FlashInfer `0.5.3`、FlashAttention `2.8.1`。H20 的 driver 由宿主提供；容器 CUDA runtime 保持 `12.6`。
3. **数学 scorer**：`math-verify==0.9.0`、`latex2sympy2-extended==1.11.0`、`pylatexenc==2.10`。formal RLVR 强制使用 `recipe/joint_training/custom_reward_function_latex_verify.py`。这条数学 RLVR 路径在镜像 build + H20 gate 通过后可用。
4. **代码 scorer**：EvalPlus、BigCodeBench、LiveCodeBench 不能只写在文档里或在 worker 启动后安装。三个 pinned harness 源码必须进入训练镜像；KodCode 还要求 `firejail`、`firejail-profiles` 和 system Python 的 `pytest`。主镜像只承诺 parser、harness import 和基本 sandbox prerequisites，不承诺 BigCodeBench 全量 library execution。当前 code-task formal 入口还被第 3.2 节的路径不一致、LCB index receipt、真实 Firejail 行为，以及 KodCode runner 暴露 test source 的安全问题阻塞。
5. **运行时禁止安装**：Hope job 内不执行 `apt-get`、`pip install`、`git clone` 或在线 dataset/model download。`run.hope` 保持 `with_requirements=false`，正式作业绑定 image digest。
6. **DolphinFS 只放持久化内容**：`ROOT` 是总安全边界；Hub dataset 放 `$DATASET_ROOT`，独立 evaluator 资产放 `$EVALUATOR_ASSET_ROOT`，模型放 `$MODEL_ROOT`，checkpoint、日志、offline W&B、receipt 和 registry 放 `$STATE_ROOT`。四个目录都必须是 `ROOT` 的严格子目录。Ray、vLLM、ZMQ、`TMPDIR` 放 pod-local `/tmp`。模型目录必须是 flat files，不能使用 Hugging Face symlink cache layout。
7. **不能原样复制旧 Dockerfile**：当前本地 `verl-harness:latest` 能运行，但其构建不是严格可复现；现有 Dockerfile 还有浮动 Git ref、`uv:latest`、未 pin Python 依赖和 cuDNN 注释与实际环境不一致的问题。
8. **推荐两镜像**：本文件的 cu126 training image 用于 FSDP/vLLM、数学 RLVR 和 online code reward；完整 BigCodeBench offline eval 使用独立 Python 3.10 evaluator image。两者通过 generation JSONL、benchmark asset receipt 和 repo commit 对接，不共享可写 checkpoint mount。

参考构建文件是 `docker/Dockerfile.meituan-rlvr.cu126`，其专用 context filter 是相邻的 `.dockerignore`，dependency conflict gate 是 `docker/verify_meituan_rlvr_pip_check.py`。这些文件是构建候选，不代表已经获得 H20 admission；必须完成本文的 build gate 和 H20 runtime gate 后，才能把 digest 写入 formal manifest。

## 0. 代码仓库、分支和平台位置

本次镜像和实验入口来自下面这一对同名分支：

| 内容 | 远程仓库 | 分支 / 固定方式 |
|---|---|---|
| verl superproject | `https://github.com/AlexJJ009/verl-v0.7.git` | `codex/rebuttal-rlvr` |
| recipe submodule | `https://github.com/AlexJJ009/verl-recipe.git` | 远程协作分支也是 `codex/rebuttal-rlvr`；formal checkout 以 superproject gitlink 为准 |

当前作者侧 worktree 是 `/data-1/code/worktrees/verl-rebuttal-rlvr`，这个路径只用于本地 provenance，不得复制成美团路径。美团侧用 `ROOT` 限定同事自己的 DolphinFS 安全边界，再分别绑定代码、dataset、模型和运行状态。按本次同事给出的目录规范，推荐：

```bash
export VERL_GIT_URL=${VERL_GIT_URL:-https://github.com/AlexJJ009/verl-v0.7.git}
export RECIPE_GIT_URL=${RECIPE_GIT_URL:-https://github.com/AlexJJ009/verl-recipe.git}
export STORAGE_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/chenzehao07
export ROOT="$STORAGE_ROOT"
export REPO_SUBPATH=wdl/WDL-SFT/verl-rebuttal-rlvr
export REPO_ROOT="$ROOT/$REPO_SUBPATH"
export DATASET_ROOT="$ROOT/huggingface/dataset/RLdataset"
export EVALUATOR_ASSET_ROOT="$ROOT/huggingface/evaluator_assets/rebuttal_rlvr"
export MODEL_ROOT="$ROOT/huggingface.co"
export STATE_ROOT="$ROOT/wdl/WDL-SFT/state/rebuttal_rlvr"
```

`STATE_ROOT` 必须位于 clean Git checkout 外。Hugging Face credential 只放在
联网 staging host 的 operator-only secret path，并且该路径必须位于
`ROOT` 外；不能把 credential 放进 `STATE_ROOT` 或挂载给 worker。完整
dataset 当前是 private；其不可变 revision、受控下载、校验和 public gate
见 `docs/joint_training/guides/rebuttal_rlvr_hf_dataset_handoff.md`。

在有 Git 网络或内部 mirror 访问能力的 preparation/builder 环境中准备 checkout；正式 Hope worker 网络隔离，禁止在 worker 启动时 `git clone`：

```bash
set -euo pipefail
: "${EXPECTED_REPO_COMMIT:?set the delivered superproject commit from the handoff message}"
: "${EXPECTED_RECIPE_COMMIT:?set the delivered recipe gitlink from the handoff message}"
for fetch_url in "$VERL_GIT_URL" "$RECIPE_GIT_URL"; do
  case "$fetch_url" in
    http://*@*|https://*@*)
      echo "HTTP(S) Git URLs must not embed credentials" >&2
      exit 2
      ;;
  esac
done
test ! -e "$REPO_ROOT"
mkdir -p "$(dirname "$REPO_ROOT")"
git clone --no-checkout --single-branch --branch codex/rebuttal-rlvr \
  "$VERL_GIT_URL" "$REPO_ROOT"
git -C "$REPO_ROOT" checkout --detach "$EXPECTED_REPO_COMMIT"
# Override the public .gitmodules URL before the first submodule fetch. This is
# mandatory when RECIPE_GIT_URL points to an internal mirror.
git -C "$REPO_ROOT" config submodule.recipe.url "$RECIPE_GIT_URL"
git -C "$REPO_ROOT" submodule update --init --recursive recipe
test -e "$REPO_ROOT/recipe/.git"
test "$(git -C "$REPO_ROOT/recipe" rev-parse --show-toplevel)" = \
  "$(realpath "$REPO_ROOT/recipe")"

REPO_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
RECIPE_COMMIT=$(git -C "$REPO_ROOT/recipe" rev-parse HEAD)
RECIPE_GITLINK=$(git -C "$REPO_ROOT" ls-tree HEAD recipe | awk '{print $3}')
VERL_FETCH_URL=$(git -C "$REPO_ROOT" remote get-url origin)
RECIPE_FETCH_URL=$(git -C "$REPO_ROOT/recipe" remote get-url origin)
test "$RECIPE_COMMIT" = "$RECIPE_GITLINK"
test "$REPO_COMMIT" = "$EXPECTED_REPO_COMMIT"
test "$RECIPE_COMMIT" = "$EXPECTED_RECIPE_COMMIT"
test -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)"
printf 'VERL_FETCH_URL=%s\nRECIPE_FETCH_URL=%s\nREPO_COMMIT=%s\nRECIPE_COMMIT=%s\n' \
  "$VERL_FETCH_URL" "$RECIPE_FETCH_URL" "$REPO_COMMIT" "$RECIPE_COMMIT"
```

若平台要求全部访问内部服务，构建同事必须在执行前把 `VERL_GIT_URL`、`RECIPE_GIT_URL` 同时设为内部 mirror；只替换 superproject URL 不够，因为 `.gitmodules` 默认仍指向公开 recipe URL。mirror URL 不得嵌入 credential，实际两项 fetch URL 和 mirror snapshot/commit receipt 一起归档。无论使用公开源还是内部 mirror，分支名、superproject commit 和 recipe gitlink 都不能改变。镜像构建必须从 `cd "$REPO_ROOT"` 后的 repo root 发起，因为 Dockerfile-specific `.dockerignore` 和三份 probe 的 `COPY docker/...` 都按 repo root 解析。最终 manifest 写入相同的 `REPO_SUBPATH`、构建后返回的 `REPO_COMMIT`，并附 recursive submodule status receipt；不能只写 branch name 代替 commit。

## 1. 当前训练入口真正消费的环境

### 1.1 纯 GRPO / verl / vLLM

| 层 | 要求 | 当前基线 | 说明 |
|---|---|---:|---|
| OS / arch | Ubuntu 22.04, `linux/amd64` | 22.04 / x86_64 | H20 worker 使用 NVIDIA Container Runtime |
| CUDA toolkit | CUDA devel | 12.6.3 | 需要 `nvcc` 构建 FA2；宿主 driver API 可以更新，但必须被 receipt 记录 |
| Python | CPython | 3.12.13 | 使用固定版本 `uv` 创建 `/opt/venv` |
| PyTorch | CUDA 12.6 wheel | `2.9.1+cu126` | 当前分支和已有 runtime profile 的兼容基线 |
| vLLM | rollout engine | `0.12.0` | 当前 launcher 使用 vLLM V1 |
| FlashInfer | vLLM attention backend | `0.5.3` | `VLLM_ATTENTION_BACKEND=FLASHINFER` |
| FlashAttention | FSDP attention backend | `2.8.1` | trainer 显式使用 `flash_attention_2`；必须在最终 Torch 安装后编译 |
| Ray | worker orchestration | `2.54.1` | 满足 repo 的 `ray>=2.41` 与 vLLM 的 `ray>=2.48` |
| Transformers | tokenizer/model config | `4.57.6` | 必须 `<5` |
| NCCL | distributed collectives | `2.27.5` | 最终以实际 wheel/runtime probe 为准 |
| cuDNN | PyTorch runtime | `9.10.2.21` | 这是当前镜像实测值；旧 Dockerfile 注释声称 9.16，但后续 resolver 将其替换 |

verl 的核心 Python 依赖至少包括：

```text
accelerate==1.13.0
codetiming==1.4.0
datasets==4.8.4
dill==0.4.1
hydra-core==1.3.2
numpy==1.26.4
pandas==3.0.2
peft==0.18.1
pyarrow==23.0.1
pybind11==3.0.3
pylatexenc==2.10
ray[default]==2.54.1
torchdata==0.11.0
tensordict==0.10.0
transformers==4.57.6
wandb==0.25.1
packaging==25.0
tensorboard==2.20.0
liger-kernel==0.7.0
```

这些是当前本地镜像的实测版本，不等价于完整 transitive lock。正式镜像构建成功后仍需导出 `python -m pip freeze --all`、保存 wheel/source hashes，并与 image digest 一起归档。

`verl` 本身来自当前 clean、commit-pinned 的 repo checkout，不从 PyPI 安装另一个 upstream `verl` package，也不在本镜像里复制一份会与 mounted checkout 漂移的源码。Hope worker 必须挂载已校验的 repo commit + recipe gitlink，从该目录启动并把它放进 `PYTHONPATH`；image receipt 与 source receipt 是两份独立、同时必需的证据。

### 1.2 当前 GRPO 不需要的宽功能组件

本次 launcher 是 FSDP + vLLM 的 text-only 单机 8-GPU GRPO，不选择 Megatron、TransformerEngine、Apex、DeepEP、SGLang 或 TRL。参考 Dockerfile 因此不安装它们，减少编译失败和 dependency drift。

若同一镜像还要支持 Megatron/DeepEP，再加一个独立的 `full` target，并固定当前本地镜像中实际解析到的 commit：

| 组件 | 当前实际 source identity |
|---|---|
| NVIDIA Apex | `270523af2387b737428ddcf34ca39c3b4c6d1113` |
| TransformerEngine | `769ed778341a32c8c593fda391700c0a80f65f1f` |
| Megatron-LM | `0d7e02bd26d3eb347ad8d1bef625002babfa6df7` |
| mbridge | `90c4633a6cdcfe5d29723d7b145d32f6f5e73303` |
| DeepEP | tag `v1.2.1`, local version `1.2.1+9af0e0d` |

不要继续使用 Apex/mbridge HEAD 或 TransformerEngine 的浮动 branch 名称。

## 2. 数学 parser、scorer 与 Math-7

### 2.1 镜像依赖

| package | 固定版本 | 用途 |
|---|---:|---|
| `math-verify` | `0.9.0` | LaTeX parse + semantic equivalence |
| `latex2sympy2-extended` | `1.11.0` | parser / normalization backend |
| `pylatexenc` | `2.10` | LaTeX text handling |
| `pandas`, `pyarrow` | 见核心表 | parquet dataset / receipt validation |
| `numpy`, `transformers`, `vllm` | 见核心表 | offline generation and aggregation |

当前 formal RLVR launcher 会把 reward path/name 固定为：

```text
recipe/joint_training/custom_reward_function_latex_verify.py
compute_score_latex_verify
```

scorer 的 output contract 是一个顺序正确且唯一的 `<think>...</think><answer>...</answer>`；答案从 `<answer>` 内的 `\boxed{...}` 提取。生成被截断、没有 EOS、boxed 在 `<answer>` 外或 format 错误时，不能得到正奖励。

### 2.2 不进入镜像、但 worker 必须看见的资产

```text
$DATASET_ROOT/data/math/train_rl_format.parquet
$DATASET_ROOT/data/math7/AIME-2025/aime-2025_with_system_prompt.parquet
$DATASET_ROOT/data/math7/MATH-500/math500-test_with_system_prompt.parquet
$DATASET_ROOT/data/math7/AMC23/amc23-test_with_system_prompt.parquet
$DATASET_ROOT/data/math7/AQUA/aqua-test_with_system_prompt.parquet
$DATASET_ROOT/data/math7/gsm8k/gsm8k-test_with_system_prompt.parquet
$DATASET_ROOT/data/math7/MAWPS/mawps-test_with_system_prompt.parquet
$DATASET_ROOT/data/math7/SVAMP/svamp-test_with_system_prompt.parquet
```

镜像不应复制这些 dataset；它们由 receipt/hash 管理，作为只读输入挂载。当前期望行数分别为 7,500（训练原始行数）、30、500、40、254、1,319、355、300；正式 worker 会重新读取 parquet 并核对 receipt。

## 3. 代码 parser、scorer 与 official evaluator

### 3.1 统一输出 parser

当前代码任务严格接受：

````text
<think>non-empty reasoning</think>
<answer>
```python
...
```
</answer>
````

正式训练/eval 默认不从全文、裸代码或 `<answer>` 外的 fenced block 回退提取。parser 在 `recipe/on_policy_wdl_sft/code_task/code_extraction.py`，formal scorer 在 `official_aligned_reward.py`。旧 `custom_reward_function_code.py` 只作为 local/G1 smoke fallback，不能产生 official benchmark 结论。

### 3.2 official harness source pins

| benchmark / route | source | 固定 identity | 本项目用途 |
|---|---|---|---|
| HumanEval+ / MBPP+ | `evalplus/evalplus` | commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`；package line `0.3.1` | online reward + offline eval |
| BigCodeBench | `bigcode-project/bigcodebench` | commit `09dd993f46c3fbf3a799465bb96d524edcb0b199`；package line `0.2.5` | official harness + project safety sanitizer + local execution；必须报告 sanitizer count |
| LiveCodeBench | `LiveCodeBench/LiveCodeBench` | commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` | release_v5 online subset + official offline eval |
| KodCode-Light-RL-10K | project `kodcode_exec` | repo commit + image digest | project-local pytest runner under Firejail；当前 hidden-test confidentiality 未通过 |
| DeepCoder / explicit smoke rows | project local runner | repo commit + image digest | local-only；不是 official score |

参考镜像把三份源码放在：

```text
/opt/code-eval/official_site/evalplus
/opt/code-eval/official_site/bigcodebench
/opt/code-eval/LiveCodeBench/lcb_runner
```

不要对 evaluator 执行普通的 `pip install <repo>`：EvalPlus、BigCodeBench 和 LiveCodeBench 的完整 dependency metadata 会重新解析 Torch/vLLM，破坏训练 ABI。参考 Dockerfile 按固定 commit 拉取源码，只把源码树复制到最终 runtime stage，并显式安装已审核的最小 runtime dependencies。

**当前 formal code-task 路径尚未统一。** 镜像默认是 `/opt/code-eval/...`；`check_official_scorer_dependencies.py` 仍硬编码 `/data-1/code_eval_envs/...`；code-task 的 Meituan adapter 又默认 `$LGX/verl-exp/code_eval_envs/...`。在 checker 改为读取 `CODE_EVAL_OFFICIAL_SITE` / `LCB_REPO_DIR` 并显式绑定 `DATASET_ROOT` / `EVALUATOR_ASSET_ROOT` / `STATE_ROOT`，或形成等价的只读 mount + hash receipt 前，不能宣称 code-task formal launch ready。这个 blocker 不影响本次数学 GRPO。

**当前 KodCode test 还不是真正隐藏的。** `official_aligned_reward.py` 把 base64 编码后的 candidate 与 test source 一起写进 Firejail 内执行的 `runner.py`；candidate 与 test 又在同一 Python interpreter 中运行。candidate 可以通过 `sys.argv[0]` 读取 `runner.py`、恢复 test source，再针对测试作答。`--private`、`--private-tmp` 和 seccomp 能限制部分宿主访问，但不能对正在执行该文件的 interpreter 隐藏 `runner.py`。因此当前实现只能作为 project-local execution reward，不能称为 formal hidden-test reward。必须先把 candidate 与 test materialization 拆到彼此不可读的隔离边界，再让本文件提供的 confidentiality probe 由红转绿。这个 blocker 同样不影响数学 GRPO。

2026-07-28 在旧 `verl-harness:latest` 的普通 Docker security context 中实测，带 `code_reward_sandbox=firejail` 标签的 candidate 仍可读取 `/root` 下的 host-home canary；另一个 candidate 也成功从 `sys.argv[0]` 恢复随机 test secret。前者说明当前本机没有 Firejail containment 正证据，后者说明即使平台以后启用 mount isolation，当前 runner 结构仍会泄漏 tests。美团 AFO 的 namespace/capability policy 可能不同，所以必须在真实 worker 重跑，但不能从本机结果推断 AFO 已安全。

镜像内还需要：

```text
appdirs==1.4.4
e2b==1.11.1
fire==0.7.1
gradio-client==2.5.0
multipledispatch==1.0.0
pqdm==0.2.0
rich==14.3.3
tempdir==0.7.1
termcolor==3.3.0
tree-sitter==0.26.0
tree-sitter-python==0.25.0
wget==3.2
timeout-decorator==0.5.0
pytest>=8,<9
```

KodCode 必须同时具备：

```text
/usr/bin/firejail
firejail profiles
/usr/bin/python3 -m pytest
```

“二进制存在”不等于 sandbox 可用。必须在真实 AFO worker 中实际运行含 `--profile=pip --private --private-dev --private-tmp --seccomp=socket` 的行为测试。基本 containment probe 会创建 host-home canary，并证明同一 candidate 在 unsandboxed control 中能读到、在 Firejail 中读不到；这只证明该项 mount isolation，不证明 test source 已隐藏。若平台禁止 namespace/seccomp，不能设置 `KODCODE_ALLOW_UNSANDBOXED=1` 进入 formal training；应改用独立 remote sandbox service。

### 3.3 代码评测资产

Hub 上的 1+4 个 code parquet 仍属于 `$DATASET_ROOT` 的 exact dataset
tree；不属于该 Hub bundle 的 evaluator cache/data 放在独立、只读的
`$EVALUATOR_ASSET_ROOT`。不要把 evaluator 资产写进 `$DATASET_ROOT`，否则
会破坏 17-file download receipt：

```text
$EVALUATOR_ASSET_ROOT/evalplus/...
$EVALUATOR_ASSET_ROOT/bigcodebench/BigCodeBench-v0.1.4.jsonl
$EVALUATOR_ASSET_ROOT/livecodebench_cache/snapshots/<snapshot>/test.jsonl ... test5.jsonl
$EVALUATOR_ASSET_ROOT/livecodebench_cache/index/release_v5_input_output.sqlite
$EVALUATOR_ASSET_ROOT/livecodebench_cache/index/release_v5_input_output.receipt.json
$DATASET_ROOT/data/code/verl_rl/kodcode_light_rl_10k_train_rl_format_author_signature_v2.parquet
$DATASET_ROOT/data/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet
$DATASET_ROOT/data/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet
$DATASET_ROOT/data/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet
$DATASET_ROOT/data/code/verl_rl/online_full_bigcodebench/official_bigcodebench_val.parquet
```

当前本机 `release_v5_input_output.sqlite` 的已验证 receipt 是 `row_count=880`、`size_bytes=3634851840`、SHA256 `2f049e91c20f55b3967655c2828f4188cef4bc13108fd3a6d0407046375954b4`。复制到 DolphinFS 后必须重新计算 SHA256，并通过 `LCB_INPUT_OUTPUT_INDEX`、`LCB_INPUT_OUTPUT_INDEX_SHA256`、`LCB_INPUT_OUTPUT_INDEX_RECEIPT` 同时绑定路径、内容和 receipt；任何一项缺失都 fail closed。

worker 必须设置 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。缺资产时 fail closed，不能从 `/root/.cache`、公网或另一位用户的 cache 临时补齐。

执行生成代码存在安全边界：不挂载 Docker socket、SSH/HF/W&B credentials 或宿主敏感目录；evaluator 数据/源码只读；每个 candidate 使用独立 `/tmp`；容器禁网。BigCodeBench/LiveCodeBench 的 local execution 不能当成强隔离，完整离线评测优先放在独立 evaluator job，而不是具有 checkpoint 写权限的 trainer process。

本项目会在 BigCodeBench official harness 前把命中 `os.kill` / `pkill` / `killall` 等模式的生成样本替换为安全失败 stub，再使用 `--execution=local`。因此报告必须写成“official harness + project safety sanitizer + local execution”，并同时给出 `unsafe_sample_count`；不能把该数值无修饰地称为 official benchmark result。

### 3.4 BigCodeBench 全量依赖必须拆镜像

BigCodeBench pinned commit 的 `Requirements/requirements-eval.txt` 有 74 个非空 requirement entries（71 个 canonical package names），包含 TensorFlow/Keras、GeoPandas/GDAL、SciPy/Numba、scikit-image/scikit-learn、OpenCV、librosa、Django/Flask、Selenium、NLTK、Pillow、lxml、openpyxl 等 task libraries。它同时固定 `numpy==1.21.2`、`pandas==2.0.3`、`matplotlib==3.7.0`、`opencv-python-headless==4.9.0.80`、`sympy==1.12`、`numba==0.55.0`、`tensorflow==2.11.0`。这些版本不能与主镜像的 Python 3.12、Torch 2.9.1、NumPy 1.26.4、Pandas 3.0.2 和 OpenCV 4.11 共用一个 venv。

参考 Dockerfile 会把下面两份原始 manifest 保留在训练镜像中，但**不会安装全量 task libraries**：

```text
/opt/code-eval/manifests/bigcodebench-requirements-eval.txt
  sha256=a4d01fb12cbce5223b51f982265cb7975bea770b758cd85cc91b803d3293e39f
/opt/code-eval/manifests/bigcodebench-Evaluate.Dockerfile
  sha256=a0d56d681c9eeaf0c6d780309aab31b6df422bad21ad1a3de2c3da0598e05b12
```

完整 BigCodeBench evaluator image 应从内部、digest-pinned 的 Python 3.10 slim/devel 等价镜像构建，按 commit `09dd993f46c3fbf3a799465bb96d524edcb0b199` 的 `Docker/Evaluate.Dockerfile` 和 `Requirements/requirements-eval.txt` 安装。不能原样使用其公网 `ADD`、GitHub `main` 或 raw URL；全部改为 pinned source stage + 内部 apt/PyPI snapshot。若内部安全扫描拒绝这些旧 pins，状态应记录为 `official-unavailable`，不能静默升级依赖后仍声称与 pinned official environment 等价。该 evaluator 只挂载只读 generation JSONL/benchmark assets 和独立 output dir，不挂载 trainer checkpoint 写目录、credentials 或 Docker socket。

因此有三个不同的 readiness 结论：

- 数学 GRPO：使用本 cu126 training image，待 build/H20/实验 admission gates。
- code online reward：parser/harness/basic sandbox 依赖在 training image 内，但仍受路径、LCB receipt、Firejail platform gate 和 KodCode test-source confidentiality 阻塞。
- BigCodeBench full offline eval：另需 Python 3.10 evaluator image 的独立 build/runtime receipt；training image 的 harness import 通过不能替代它。

## 4. 上游镜像怎么选

### 4.1 推荐顺序

1. **最佳**：美团内部已经安全扫描、且能以 digest 引用的 CUDA 12.6.3 devel Ubuntu 22.04 镜像。它可以没有 Python/PyTorch，参考 Dockerfile 从这一层安装。`uv` helper image 也必须镜像到内部 registry 并以 digest 引用。
2. **可接受**：内部已有 Python 3.12 + PyTorch 2.9.1+cu126 的 training/devel 镜像，但必须证明它包含 CUDA compiler/headers，并且没有预装冲突的 vLLM/SGLang/Torch。构建后仍跑全部 gate。
3. **迁移现有镜像**：若美团允许导入 OCI archive，可在 tmux 中 `docker save` 当前 `verl-harness`，校验 archive SHA256 后推入内部 registry，再增加 evaluator layer。它最快，但会继承本文第 5 节列出的 dependency debt；不能跳过审计。

不推荐：

- `vllm/vllm-openai` serving image：面向推理服务，通常没有完整 CUDA devel toolchain，且 Torch/CUDA matrix 由 vLLM image 决定。
- CUDA 12.8/12.9 或其他 NGC PyTorch 月度镜像：会改变当前已经验证的 cu126 runtime contract。
- 仅用 `ubuntu:22.04` 再手工拼 NVIDIA userspace libraries：更难证明 NCCL/cuDNN/driver compatibility。

### 4.2 Base image digest

本文无法替你填写美团内部 registry 的 digest。构建者必须准备：

```text
REPLACE_MEITUAN_CUDA_BASE@sha256:<digest>
REPLACE_MEITUAN_UV_IMAGE@sha256:<digest>
INTERNAL_UV_MANAGED_CPYTHON_MIRROR/
```

分别作为 `CUDA_BASE_IMAGE`、`UV_IMAGE` 和 `UV_PYTHON_INSTALL_MIRROR` 传入。CPython mirror 必须提供与 `uv python install --mirror` 兼容的 CPython 3.12.13 artifact。resolved base/uv digest、Python artifact identity、内部 apt/PyPI/PyTorch snapshot identity、最终 image digest、Dockerfile SHA256、evaluator commits 和完整 package freeze 都写入 build receipt。禁止 formal build 只记录 mutable tag。

## 5. 旧 `verl-harness` 的实测状态与已知债务

2026-07-28 在本机只读检查的 image ID 是：

```text
sha256:c9d525a1f4b33267bd00be60fe00693338253537cac78151e4c55a6d3a7e5708
```

它不是 registry digest，不能直接填入 Meituan manifest。实测核心版本：

```text
Python 3.12.13
torch 2.9.1+cu126
vllm 0.12.0
flashinfer-python 0.5.3
flash-attn 2.8.1
ray 2.54.1
transformers 4.57.6
math-verify 0.9.0
latex2sympy2-extended 1.11.0
firejail 0.9.66
```

已知问题：

- `evalplus`、`timeout-decorator`、official BigCodeBench package 和 LiveCodeBench source 没有 baked into image；当前代码任务依赖外部 `PYTHONPATH`。
- vLLM 0.12.0 wheel metadata 要求 Torch `2.9.0` / torchvision `0.24.0` / torchaudio `2.9.0`，而本分支基线是 Torch `2.9.1`。这是当前项目的有意兼容覆盖，不是 upstream-supported exact pair。
- 旧镜像还出现过无关的 `cupy`/OpenCV/Numpy 和 TRL/tyro `pip check` 冲突。本次 lean image 不安装未使用的 `qwen_vl_utils`、CuPy、TRL，且 pin `opencv-python-headless==4.11.0.86`。
- 旧 Dockerfile 使用 `uv:latest`、Apex HEAD、mbridge HEAD、浮动 Python dependency；不能据此声称 byte-for-byte rebuild。
- 旧 Dockerfile 先安装 cuDNN 9.16，后续 vLLM/Torch resolver 又落回 9.10.2.21。参考 Dockerfile 直接采用实测值，不保留错误注释。

若坚持 upstream-supported vLLM wheel matrix，可以把 Torch 改成 `2.9.0`、torchvision `0.24.0`、torchaudio `2.9.0`。这会改变当前 runtime contract，必须重新完成 H20 calibration、两臂 worker evidence 和 G4 attestation；不能只在其中一个 arm 上改。

## 6. Dockerfile 构建说明

### 6.1 构建前准备

1. 由平台 owner 给出可信 image/apt/PyPI/PyTorch/uv-Python/Git prefix allowlist。将 `CUDA_BASE_IMAGE` 和 `UV_IMAGE` 映射到 allowlisted 美团内部 registry 的不可变 digest；参考 Dockerfile 故意不提供公网 `FROM` 默认值。Docker 在执行任何 `RUN` gate 前就会解析/pull `FROM`，所以 image-builder 必须先用平台 policy 校验这两个引用；`FORMAL_BUILD=1` 中的 prefix/digest 检查是第二道 fail-closed gate，不是网络防火墙。
2. 准备内部 apt mirror、PyPI snapshot、PyTorch cu126 wheel snapshot，以及 uv-managed CPython 3.12.13 mirror。内部 Python package snapshot 必须包含 NVIDIA cu12 wheels、vLLM、FlashInfer 和 FlashAttention sdist/build dependency。
3. 确认 image builder 能访问三个 evaluator 的内部 Git mirror。固定 commit 不能只固定 branch/tag；不要在 worker 运行时下载。
4. Dockerfile 假设内部 mirror 在 builder 网络内无需把 credential 写进 URL。若必须认证，改用 BuildKit secret/SSH mount 或预置 source stage；不得把 token 放进 `--build-arg`、Docker history 或最终 layer。
5. 镜像中不复制模型、dataset、checkpoint 或任何 secret。模型与数据由 `$ROOT` 下的 hash receipt 管理。

示例：

```bash
DOCKER_BUILDKIT=1 docker build \
  --platform=linux/amd64 \
  --file docker/Dockerfile.meituan-rlvr.cu126 \
  --build-arg FORMAL_BUILD=1 \
  --build-arg TRUSTED_IMAGE_PREFIX='REGISTRY/' \
  --build-arg TRUSTED_APT_PREFIX='INTERNAL_APT_MIRROR/' \
  --build-arg TRUSTED_PYPI_PREFIX='INTERNAL_PYPI_SNAPSHOT/' \
  --build-arg TRUSTED_PYTORCH_PREFIX='INTERNAL_PYTORCH_CU126_SNAPSHOT/' \
  --build-arg TRUSTED_UV_PYTHON_PREFIX='INTERNAL_UV_PYTHON_MIRROR/' \
  --build-arg TRUSTED_GIT_PREFIX='INTERNAL_GIT/' \
  --build-arg CUDA_BASE_IMAGE='REGISTRY/CUDA@sha256:REPLACE' \
  --build-arg UV_IMAGE='REGISTRY/astral-sh/uv@sha256:REPLACE' \
  --build-arg APT_MIRROR='INTERNAL_APT_MIRROR/ubuntu' \
  --build-arg UV_PYTHON_INSTALL_MIRROR='INTERNAL_UV_PYTHON_MIRROR/' \
  --build-arg PYPI_INDEX_URL='INTERNAL_PYPI_SNAPSHOT/simple' \
  --build-arg PYTORCH_INDEX_URL='INTERNAL_PYTORCH_CU126_SNAPSHOT/simple' \
  --build-arg EVALPLUS_GIT_URL='INTERNAL_GIT/evalplus.git' \
  --build-arg BIGCODEBENCH_GIT_URL='INTERNAL_GIT/bigcodebench.git' \
  --build-arg LIVECODEBENCH_GIT_URL='INTERNAL_GIT/LiveCodeBench.git' \
  --tag REGISTRY/verl-rebuttal-rlvr:cu126-candidate .
```

这里的 internal index/mirror 应绑定不可变 snapshot 或在 build receipt 中记录仓库 snapshot ID。只有 mutable package index + top-level version pins 仍不足以 byte-for-byte 重建 transitive dependency；构建者还需归档实际下载的 wheels/sdists 及 SHA256。

`TRUSTED_*_PREFIX` 必须以 `/` 结尾，不是由实验提交者自行发明的字符串；它们由美团 image-builder/registry owner 提供并进入 build receipt。Dockerfile 同时设置 `UV_NO_CONFIG=1`，清空 `UV_INDEX`、uv/pip extra-index 与 find-links，固定 `UV_INDEX_STRATEGY=first-index`，并把 pip config 指向 `/dev/null`，避免 base image 注入第二个 resolver source。formal build 还断言 `TARGETARCH=amd64`；未使用 BuildKit 或构建到其他架构会直接失败。

必须从 repo root 运行上述命令。`docker/Dockerfile.meituan-rlvr.cu126.dockerignore` 只把三个 probe 源文件送入 BuildKit context，避免把接近 1 GB 的 recipe worktree 和任何实验资产发送给 image builder；不要删除该文件或改用不支持 Dockerfile-specific ignore 的 legacy builder。

构建完成后以 registry 返回的 digest 为准：

```bash
docker push REGISTRY/verl-rebuttal-rlvr:cu126-candidate
docker image inspect REGISTRY/verl-rebuttal-rlvr@sha256:REPLACE
```

### 6.2 Dependency integrity

当前 contract 对 vLLM wheel metadata 有三项已知 override，所以不能把裸 `pip check` 的非零退出码简单忽略，也不能要求它无条件全绿。Dockerfile 会执行：

```bash
python /usr/local/libexec/verify_meituan_rlvr_pip_check.py --self-test
python /usr/local/libexec/verify_meituan_rlvr_pip_check.py
```

这个 gate：

1. 保存完整 `pip check` 输出；
2. 只允许 vLLM 0.12.0 对 Torch/torchvision/torchaudio 2.9.0 的三项已登记 metadata mismatch；
3. 强制三项 observed version 正好是 Torch `2.9.1+cu126`、torchvision `0.24.1+cu126`、torchaudio `2.9.1+cu126`；
4. 对 injected extra conflict 做负测，证明 verifier 会变红；
5. 任何其他 missing/conflict 或 compatibility matrix 漂移都失败；
6. 继续跑真实 GPU rollout，因为 metadata allowlist 不能证明 ABI 可用。

若采用 upstream-supported Torch 2.9.0 matrix，则 `pip check` 必须全绿，不再使用上述 allowlist。

## 7. Build gate：无 GPU 也必须通过

### 7.1 版本和 import

```bash
python - <<'PY'
import importlib.metadata as md
import torch, vllm, flashinfer, flash_attn, ray, transformers
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

expected = {
    "torch": "2.9.1+cu126",
    "vllm": "0.12.0",
    "flashinfer-python": "0.5.3",
    "flash-attn": "2.8.1",
    "ray": "2.54.1",
    "transformers": "4.57.6",
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
}
observed = {name: md.version(name) for name in expected}
assert observed == expected, (observed, expected)
assert torch.version.cuda == "12.6", torch.version.cuda
print(observed)
PY
```

### 7.2 official evaluator source identity

```bash
PYTHONPATH=/workspace/verl:/opt/code-eval/official_site:/opt/code-eval/LiveCodeBench \
python - <<'PY'
import importlib

names = (
    "evalplus.evaluate",
    "evalplus.gen.util",
    "evalplus.eval._special_oracle",
    "bigcodebench.evaluate",
    "bigcodebench.gen.util",
    "bigcodebench.eval._special_oracle",
    "lcb_runner.benchmarks.code_generation",
    "lcb_runner.evaluation.compute_code_generation_metrics",
)
for name in names:
    module = importlib.import_module(name)
    print(name, module.__file__)
    assert module.__file__.startswith("/opt/code-eval/")
PY

(cd /opt/code-eval/LiveCodeBench && \
  PYTHONPATH=/opt/code-eval/official_site:/opt/code-eval/LiveCodeBench \
  python -c 'import lcb_runner.runner.custom_evaluator')

test "$(cat /opt/code-eval/revisions/evalplus)" = \
  26d6d00bb1fd0fa37f39c99d5290da67891d1c5e
test "$(cat /opt/code-eval/revisions/bigcodebench)" = \
  09dd993f46c3fbf3a799465bb96d524edcb0b199
test "$(cat /opt/code-eval/revisions/livecodebench)" = \
  28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24

test "$(sha256sum /opt/code-eval/manifests/bigcodebench-requirements-eval.txt | awk '{print $1}')" = \
  a4d01fb12cbce5223b51f982265cb7975bea770b758cd85cc91b803d3293e39f
```

这些 import 只证明 harness path 可加载，不证明 71 个 BigCodeBench task-library pins 已安装；后者只在第 3.4 节的独立 evaluator image 验收。

### 7.3 parser/scorer behavior

数学至少验证：正确 boxed fraction 得正分；正确答案但无 EOS 得负分；boxed 在 `<answer>` 外得负分。

代码至少验证：

- strict parser 接受带非空 `<think>`、唯一 `<answer>` 和 Python fence 的输出；
- 缺 `<think>`、缺 fence、重复 answer tag 时为 `format_error` / extraction failure；
- KodCode pass sample 在 Firejail 内通过；host-home canary 在 unsandboxed control 中可读、在 Firejail 中不可读；
- candidate 读取当前 executable `runner.py`、恢复随机 test secret 的攻击必须失败。当前实现会被这个 probe 正确拦截，因此 code-task formal gate 仍是红色；
- EvalPlus、BigCodeBench、LiveCodeBench 的 required imports 全部成功。

当前 `tests/on_policy_wdl_sft/test_code_task_reward_and_metrics.py` 有两个 test helper 仍只生成 `<answer>`、没有 `<think>`，因此在 strict format contract 下得到 `format_error`。在修复该 test drift 前，不要把整文件的两个失败错误归因于镜像或 Firejail；镜像 gate 应使用符合当前 contract 的 sample，并把该 repo-test drift 单独记录。

### 7.4 离线与 receipt

```bash
python - <<'PY'
import os

expected = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "offline",
}
assert {key: os.environ.get(key) for key in expected} == expected
PY

python -m pip freeze --all > /path/to/build-receipt/pip-freeze.txt
python /usr/local/libexec/verify_meituan_rlvr_pip_check.py \
  > /path/to/build-receipt/pip-check.txt 2>&1
```

verifier 命令返回非零就停止交付。不要用 `|| true` 丢掉退出码。

## 8. H20 runtime gate：必须在真实 AFO worker 通过

仅 import 成功不够。候选镜像在一个实际 8×H20 worker 上至少完成。宿主 NVIDIA driver 必须与审批后的 H20 profile 一致；当前已观测 profile 是 `590.48.01`，不能仅凭“高于 CUDA 12.6 理论最低版本”放行。

1. `nvidia-smi`：恰好 8 张 H20，记录 driver version 与 CUDA driver API version。
2. `torch.cuda.device_count()==8`，在每张卡分配 tensor。
3. 单进程和 8-rank NCCL `all_reduce`；若以后跨节点，再加 2-node NCCL probe。
4. 加载本次真实 Qwen checkpoint tokenizer/config。
5. `VLLM_ATTENTION_BACKEND=FLASHINFER` 下做一个 prompt、`n=1` 的真实 generation；生成后通过 vLLM `collective_rpc` 逐个 TP worker 检查已初始化的 `attn_groups` 与 metadata builders 都是 FlashInfer。日志只作补充证据。
6. 先做 FlashAttention BF16 kernel forward/backward；随后用真实 launcher 的 bounded training smoke 证明 Transformers `attn_implementation=flash_attention_2` + FSDP integration。前者不能替代后者。
7. 读取训练 parquet 和全部 Math-7 parquet；运行 strict math scorer behavior cases。
8. 若本次验收 code-task，再在相同 AFO security context 下运行 Firejail basic containment 与 hidden-test confidentiality gate；该项不是数学 RLVR admission 的组成部分，且当前 confidentiality gate 已知会失败。
9. 运行 `CONFIG_ONLY=1` 的 rebuttal RLVR resolved-config gate，随后才做 bounded GPU calibration。

候选镜像内置了 GPU/Firejail behavior probes。数学 RLVR image admission 必须在真实 worker 中执行下面三组 GPU probe，并依赖 `set -o pipefail` 保留失败退出码：

```bash
set -euo pipefail
PROBE=/usr/local/libexec/smoke_meituan_rlvr_h20.py
RECEIPT_DIR=${RECEIPT_DIR:?set a persistent receipt directory under ROOT}
INIT_MODEL_PATH=${INIT_MODEL_PATH:?set the real flat Qwen checkpoint path}
REPO_ROOT=${REPO_ROOT:?set the immutable mounted repo path}
EXPECTED_DRIVER=${EXPECTED_DRIVER:?copy from the approved H20 profile}
EXPECTED_CUDA_DRIVER_API=${EXPECTED_CUDA_DRIVER_API:?copy from the approved H20 profile}
mkdir -p "$RECEIPT_DIR"

python "$PROBE" inventory --expected-gpus 8 --expected-gpu-name H20 \
  --expected-driver "$EXPECTED_DRIVER" \
  --expected-cuda-driver "$EXPECTED_CUDA_DRIVER_API" \
  2>&1 | tee "$RECEIPT_DIR/h20-inventory.log"

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  "$PROBE" distributed --expected-world-size 8 \
  2>&1 | tee "$RECEIPT_DIR/nccl-fa2.log"

VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_LOGGING_LEVEL=DEBUG \
  python "$PROBE" vllm --model "$INIT_MODEL_PATH" --tensor-parallel-size 8 \
  2>&1 | tee "$RECEIPT_DIR/vllm-flashinfer.log"
```

若验收 code-task，再单独执行下面两条。第一条是 image/platform prerequisite，目标 worker 上必须通过；第二条是 formal scorer gate，在当前 commit 上预期非零，不属于数学 RLVR admission 命令：

```bash
set -euo pipefail
RECEIPT_DIR=${RECEIPT_DIR:?set a persistent receipt directory under ROOT}
REPO_ROOT=${REPO_ROOT:?set the immutable mounted repo path}

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
  python /usr/local/libexec/smoke_meituan_code_firejail.py --containment-only \
  2>&1 | tee "$RECEIPT_DIR/kodcode-firejail-containment.log"

# Run this separately after the first command so its expected nonzero status is
# captured rather than hidden by a shell continuation.
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
  python /usr/local/libexec/smoke_meituan_code_firejail.py \
  2>&1 | tee "$RECEIPT_DIR/kodcode-hidden-test-confidentiality.log"
```

`inventory` 对 GPU count/name、NVIDIA driver、CUDA driver API/runtime、PyTorch、vLLM、FlashInfer、FlashAttention、NCCL package/runtime 做 exact comparison，并执行逐卡 allocation；这些 expected package defaults 已固定为本文基线，修改参数等同于修改 runtime contract。`distributed` 在 8 ranks 上实际执行 NCCL `all_reduce` 与 FlashAttention BF16 forward/backward。`vllm` 加载真实 flat checkpoint并生成非空输出，再对每个 TP worker 的已初始化 engine state 做 RPC introspection，要求 backend 为 `FlashInferBackend`、metadata builder 为 `FlashInferMetadataBuilder`；它不再用独立 selector 调用冒充 engine 证据。

Firejail `--containment-only` probe 先证明正常实现通过，再用同一个 host-home canary 做 unsandboxed-readable / Firejail-unreadable 对照。无参数的 formal code probe 还会生成随机 test secret，并要求读取 executable `runner.py` 的 candidate 无法恢复它；当前 scorer 会泄漏该 secret，所以该命令必须非零退出。修复 scorer 后，只有这个命令真实转绿才能解除 hidden-test blocker。不要用 `--containment-only` 的绿灯替代 formal code gate。

前三组 GPU probe 覆盖第 1–5 项以及第 6 项的 raw FlashAttention kernel evidence；它们不证明真实 FSDP model integration，后者仍需 bounded training smoke。Firejail 两种模式分别提供 basic containment 和 code-only confidentiality evidence。这些 probe 不替代 dataset/model receipt、Math-7/scorer cases、`CONFIG_ONLY=1` resolved config 和当前平台 admission schemas。构建同事仍需逐项返回命令、退出码和完整 log；不能用 `validate_inputs.py` 的六字段版本 probe 代替这些行为证据。

runtime receipt 至少记录：

```text
image digest
NVIDIA driver
CUDA driver API
torch.version.cuda
PyTorch
vLLM
FlashInfer
FlashAttention
NCCL
GPU model/count
Dockerfile hash
repo commit + recipe gitlink
evaluator source commits
dataset/model receipt hashes
```

当前 `validate_inputs.py` 已强制比对 NVIDIA driver、CUDA driver/runtime、PyTorch、vLLM 和 FlashInfer；新的 inventory probe 还 exact-check FlashAttention/NCCL，distributed/vLLM probes 再提供 kernel/collective/backend behavior。仍需保存完整 preflight log，不能用现有六字段 probe 代替。

## 9. Meituan/AFO 运行约束

- `run.hope` 使用 1 worker、8×H20、1,920,000 MB memory、128 vcores、512 GiB shm；变更资源需要重新形成 profile/receipt。
- `afo.docker.image.name` 填不可变 image reference；manifest 同时记录 digest。
- `with_requirements=false`；worker 不做依赖安装。
- 镜像默认 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`WANDB_MODE=offline`。Meituan worker 永远不执行 `wandb sync`；成功训练通过 release gate 后，把 offline run、日志、checkpoint 和 SHA-256 manifest 交给联网机器，再由人手工处理后续同步。
- `RAY_TMPDIR`、`TMPDIR`、`VLLM_CONFIG_ROOT`、`VERL_ZMQ_IPC_DIR` 放 `/tmp/rebuttal_rlvr/...`。
- `ROOT` 只定义总安全边界；manifest 分别绑定其严格子目录 `DATASET_ROOT`、`EVALUATOR_ASSET_ROOT`、`MODEL_ROOT`、`STATE_ROOT`。代码 checkout 位于 `$ROOT/$REPO_SUBPATH`，Hub dataset 位于 `$DATASET_ROOT`，evaluator cache/data 位于 `$EVALUATOR_ASSET_ROOT`，模型位于 `$MODEL_ROOT`，checkpoint/eval output/log/offline-W&B/receipt/registry 位于 `$STATE_ROOT`；formal worker 丢弃继承的 `/data-1` 或其他用户路径。
- DolphinFS 不支持 symlink/hardlink。模型使用 flat directory；dataset 必须由联网 staging host 上的 guarded downloader 通过 `大流量 -> [BW] 香港非家宽 leaf` 拉取并核验，不能在 worker 内执行裸 `hf download`。route receipt 必须是 schema v2，并绑定 runtime selector membership、controller projection、每个 redirect hostname 和包含 server/port 的 leaf identity fingerprint；模型转移同样使用 flat directory，并校验每个 weight shard hash。
- formal worker 校验 clean repo commit、recipe gitlink、image digest、model/data/scorer receipts；任一漂移 fail closed。

## 10. 给同事的最终交付包

构建前交付：

```text
docker/Dockerfile.meituan-rlvr.cu126
docker/Dockerfile.meituan-rlvr.cu126.dockerignore
docker/verify_meituan_rlvr_pip_check.py
docker/smoke_meituan_rlvr_h20.py
docker/smoke_meituan_code_firejail.py
本文件
docs/joint_training/guides/rebuttal_rlvr_hf_dataset_handoff.md
verl/recipe remote URL + `codex/rebuttal-rlvr` branch
若走内网：`VERL_GIT_URL` + `RECIPE_GIT_URL` + mirror snapshot identity
`REPO_SUBPATH` + repo commit + recipe gitlink + recursive submodule receipt
`ROOT` + `DATASET_ROOT` + `EVALUATOR_ASSET_ROOT` + `MODEL_ROOT` + `STATE_ROOT` 的 manifest 值及 path receipt
三个 evaluator commit / 内部 mirror URL
CUDA base image internal digest
uv helper image internal digest + uv-managed CPython mirror identity
apt / PyPI / PyTorch immutable snapshot identity
platform-owned TRUSTED_*_PREFIX allowlist
WDL-SFT / ordinary-SFT flat model directories及逐文件 hash manifest
数学训练集、Math-7、代码 evaluator assets receipts
目标 registry/repository、builder push 权限和 H20 worker security profile
```

构建后同事返回：

```text
final image reference + digest
base image digest
Dockerfile SHA256
pip/H20/Firejail probe script SHA256
pip freeze --all
pip check 原始输出和 allowlist 判定
CPU build-gate log
8×H20 runtime-gate log
NCCL / FlashInfer / FA2 / Firejail behavior logs
evaluator source revision files
内部 package/source snapshot IDs 和 wheel/sdist SHA256 manifest
若验收 BigCodeBench full offline：独立 Python 3.10 evaluator image digest + pip freeze/check + task-library smoke
```

这些材料齐全后，才能把新 image digest 写入 `platform/hope_rebuttal_rlvr` 的 H20 profile / calibration receipts。镜像构建成功本身不是训练 admission，也不能替代两臂使用同一 digest 和同一 scientific config 的公平性检查。当前数学 RLVR 除 build/H20 evidence 外，仍需 G0 authoritative ordinary-SFT/WDL-SFT checkpoint + model-path receipts，以及并行 Hope 路线的 G3/G4/G5；code-task 还额外需要第 3.2 节的路径统一、LCB receipt、Firejail platform evidence、KodCode test-source confidentiality 修复与独立 BigCodeBench evaluator receipt。
