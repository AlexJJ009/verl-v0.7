# 服务器迁移后的环境问题排查记录

**迁移时间**：2026-03-06
**迁移内容**：verl joint training 项目（feature/joint-training 分支）
**目标服务器**：`pm-1782`，8x NVIDIA H800，PyTorch 2.6.0+cu124，CUDA 12.4

---

## 问题一：joint model 目录不存在导致 HFValidationError

### 现象

脚本运行后立刻崩溃，错误如下：

```
huggingface_hub.errors.HFValidationError: Repo id must be in the form
'repo_name' or 'namespace/repo_name':
'/data-1/.cache/huggingface/QwenJoint-1.7B'.
Use `repo_type` argument if needed.

OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/data-1/.cache/huggingface/QwenJoint-1.7B'.
```

完整调用栈止于 tokenizer 加载：

```
File "/data-1/verl07/verl/verl/trainer/main_ppo.py", line 308, in run
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
  File "/data-1/verl07/verl/verl/utils/tokenizer.py", line 61, in hf_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
  ...
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': ...
```

### 根因分析

错误信息看起来像是 repo ID 格式问题，实际根因是：

**`/data-1/.cache/huggingface/QwenJoint-1.7B` 目录不存在。**

HuggingFace transformers 的 `cached_file` 在收到路径参数时，会先判断它是否为本地已存在的目录。若目录**不存在**，则把它当作 repo ID 进行格式校验，路径中的多级 `/` 导致校验失败，抛出 `HFValidationError`。

| 场景 | transformers 行为 |
|------|-----------------|
| 路径存在（本地目录） | 直接读取文件，不做 repo ID 校验 |
| 路径不存在 | 视为 repo ID，尝试从 HF Hub 下载，校验失败时抛出 `HFValidationError` |

### 缺失的原因

项目在旧服务器上运行过 `prepare_joint_weights.py`，生成了 `QwenJoint-1.7B` 目录。迁移时只迁移了代码，**模型权重没有随代码一起迁移**（体积 7.6GB），导致新服务器上该目录不存在。

此外，base model `Qwen3-1.7B-Base` 也不在新服务器上，需要先下载。

### 解决步骤

**Step 1：下载 base model**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='Qwen/Qwen3-1.7B-Base',
    local_dir='/data-1/.cache/huggingface/Qwen3-1.7B-Base',
)
```

或等价的 Python 脚本：

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B-Base',
                  local_dir='/data-1/.cache/huggingface/Qwen3-1.7B-Base')
"
```

**Step 2：生成 joint model**

```bash
# Run inside the Docker container (verl-harness)
python -m verl.models.joint_model.prepare_joint_weights \
    --base_model_path /data-1/.cache/huggingface/Qwen3-1.7B-Base \
    --output_path /data-1/.cache/huggingface/QwenJoint-1.7B \
    --fusion_lambda 0.5
```

生成结果（约 7.6GB）：

```
/data-1/.cache/huggingface/QwenJoint-1.7B/
├── config.json                    # model_type=qwen_joint，含 auto_map
├── configuration_joint_qwen3.py   # 从源码复制
├── modeling_joint_qwen3.py        # 从源码复制
├── model.safetensors              # 7.6GB，sub_models.0.* + sub_models.1.*
├── tokenizer.json
├── tokenizer_config.json
└── chat_template.jinja
```

### 预防措施

在 launch script (`run_joint_grpo_qwen3_1.7b.sh`) 中加入了自动检测逻辑：

```bash
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/data-1/.cache/huggingface/Qwen3-1.7B-Base"}
MODEL_PATH=${MODEL_PATH:-"/data-1/.cache/huggingface/QwenJoint-1.7B"}

if [ ! -d "$MODEL_PATH" ]; then
    echo "Joint model not found at $MODEL_PATH. Preparing from base model..."
    if [ ! -d "$BASE_MODEL_PATH" ]; then
        echo "ERROR: Base model not found at $BASE_MODEL_PATH"
        exit 1
    fi
    python3 -m verl.models.joint_model.prepare_joint_weights \
        --base_model_path "$BASE_MODEL_PATH" \
        --output_path "$MODEL_PATH" \
        --fusion_lambda 0.5
fi
```

---

## 问题二：flash_attn ABI 不兼容导致 ImportError

### 现象

进入训练主循环后，每个 Ray worker 初始化模型时报错：

```
ImportError: /data-1/.cache/conda/envs/verl07/lib/python3.10/
site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so:
undefined symbol:
_ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11
char_traitsIcESaIcEEE
```

调用栈：

```
File "modeling_joint_qwen3.py", line 31, in __init__
    super().__init__(config)
  File "transformers/modeling_utils.py", in __init__
    self.config._attn_implementation_internal = \
        self._check_and_adjust_attn_implementation(...)
  File "transformers/modeling_flash_attention_utils.py", in _lazy_imports
    from flash_attn import flash_attn_func, flash_attn_varlen_func
  File "flash_attn/flash_attn_interface.py", line 15, in <module>
    import flash_attn_2_cuda as flash_attn_gpu
ImportError: ... undefined symbol: _ZN3c105Error...
```

### 根因分析

这是 **C++ ABI 不匹配**问题，与 flash_attn 版本号无关。

| | 旧服务器（编译 flash_attn 时的环境）| 新服务器（当前 PyTorch）|
|---|---|---|
| GCC C++ ABI | `_GLIBCXX_USE_CXX11_ABI=1`（cxx11abiTRUE） | `_GLIBCXX_USE_CXX11_ABI=0`（cxx11abiFALSE） |
| PyTorch 版本 | 可能为 2.4.x 或 2.5.x | **2.6.0** |

未定义的符号 `_ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE` 是 c10 库（PyTorch 核心）中 `c10::Error::Error(SourceLocation, std::__cxx11::string)` 的 mangled name。其中 `std::__cxx11::basic_string` 是 CXX11 ABI 的符号，但当前 PyTorch 2.6.0 使用旧 ABI，不包含此符号。

验证方式：

```bash
# 查看 PyTorch 编译标志
python -c "import torch; print(torch.__config__.show())" | grep CXX_FLAGS
# 输出：-D_GLIBCXX_USE_CXX11_ABI=0  ← 旧 ABI

# 查看已安装的 flash_attn
pip show flash-attn
# Version: 2.7.4.post1  ← 从旧服务器迁移来，用 cxx11abiTRUE 编译
```

**迁移 conda 环境时直接复制了 `.so` 文件，而该 `.so` 与新环境的 PyTorch 版本 ABI 不匹配。**

### 解决方案

由于不能重新编译 flash_attn（耗时长，且 ABI 版本需要完全匹配），改为卸载 flash_attn，使用 PyTorch 内置的 SDPA（Scaled Dot Product Attention）。

**H800 上的 SDPA 性能说明**：PyTorch 2.6 在 Hopper 架构 GPU（H100/H800）上，SDPA 会自动选用 cuDNN FlashAttention 后端，性能与独立安装的 flash_attn 相当，无需额外安装。

```bash
pip uninstall flash-attn -y
```

卸载后，transformers 自动 fallback 到 SDPA：

```python
# 验证
from transformers import Qwen3ForCausalLM, AutoConfig
config = AutoConfig.from_pretrained('/data-1/.cache/huggingface/Qwen3-1.7B-Base')
model = Qwen3ForCausalLM(config)
print(model.config._attn_implementation)  # 输出: sdpa
```

### 注意事项

卸载 flash_attn 后不要忘记清除 transformers 的 module cache，否则旧的缓存文件可能掩盖问题：

```bash
rm -rf /data-1/.cache/huggingface/modules/transformers_modules/QwenJoint_hyphen_1_dot_7B
```

---

## 问题三：verl 硬编码 flash_attention_2 作为默认值

### 现象

卸载 flash_attn 后再次运行，仍然报错，且错误信息变化了：

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used
due to the following error: the package flash_attn seems to be not installed.
Please refer to the documentation of https://huggingface.co/docs/transformers/
perf_infer_gpu_one#flashattention-2 to install Flash Attention 2.
```

与问题二的区别：

| | 问题二 | 问题三 |
|---|---|---|
| flash_attn 状态 | 已安装但 ABI 损坏 | 未安装 |
| 错误类型 | `.so` undefined symbol | "FA2 has been toggled on" |
| 错误来源 | `flash_attn/__init__.py` | `transformers/modeling_utils.py` |

### 根因分析

verl 的 `fsdp_workers.py` 在加载模型时显式指定了 attention 实现方式，默认值硬编码为 `flash_attention_2`：

```python
# verl/workers/fsdp_workers.py line 388
attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")

# line 463
actor_module = actor_module_class.from_pretrained(
    pretrained_model_name_or_path=local_path,
    ...
    attn_implementation=attn_implementation,  # 显式传入 "flash_attention_2"
)
```

当 `attn_implementation` 被**显式**设为 `"flash_attention_2"` 时，transformers 不会做自动 fallback，而是直接抛出 ImportError（"FA2 has been toggled on"）。

问题一中卸载 flash_attn 后，transformers 的**自动检测**逻辑能够 fallback 到 SDPA，但 verl 绕过了这个自动检测，直接传入了固定值。

### 解决方案

在 launch script 的 Hydra 参数中，通过 `override_config.attn_implementation` 覆盖 verl 的默认值：

```bash
# run_joint_grpo_qwen3_1.7b.sh 中的 Model 配置段
+actor_rollout_ref.model.override_config.attn_implementation=sdpa \
```

`override_model_config` 在 verl 中的读取路径：

```
Hydra 参数
  actor_rollout_ref.model.override_config.attn_implementation=sdpa
    → self.config.model.override_config (fsdp_workers.py line 908)
      → override_model_config.get("attn_implementation", "flash_attention_2")
```

Actor、rollout、ref 三个 worker 都读取 `actor_rollout_ref.model.override_config`，因此一处修改覆盖所有 worker。

---

## 迁移到新服务器的完整 Checklist

```
□ 模型权重（不随代码自动迁移）
  □ base model: /data-1/.cache/huggingface/Qwen3-1.7B-Base
  □ joint model: /data-1/.cache/huggingface/QwenJoint-1.7B

□ 数据集
  □ /data-1/dataset/gsm8k/train.parquet
  □ /data-1/dataset/gsm8k/test.parquet

□ 环境依赖
  □ Docker 容器 verl-harness 已启动（docker run --rm --gpus all -v /data-1/verl07/verl:/workspace/verl -v /data-1:/data-1 --ipc=host verl-harness bash）
  □ flash_attn 2.8.1 预装在镜像中，无需额外安装

□ 路径相关
  □ /data-1/.cache/huggingface/  存在
  □ /data-2/checkpoints/JointTraining/GRPO/  存在（或可创建）
  □ RAY_TMPDIR=/data-1/ray_tmp  可写

□ 快速验证命令（运行前确认）
  python -c "
  from transformers import AutoModelForCausalLM
  m = AutoModelForCausalLM.from_pretrained(
      '/data-1/.cache/huggingface/QwenJoint-1.7B',
      trust_remote_code=True
  )
  print('OK, attn:', m.config._attn_implementation)
  "
```

---

## 问题触发时间线

```
10:14  脚本首次运行
10:14  问题一：QwenJoint-1.7B 不存在 → HFValidationError，立即崩溃

       修复：下载 Qwen3-1.7B-Base，运行 prepare_joint_weights.py
10:24  QwenJoint-1.7B 生成完毕（7.6GB）

10:29  脚本第二次运行
10:29  问题二：flash_attn_2_cuda.so ABI 不兼容 → ImportError (undefined symbol)
       同时出现警告：FA2 dtype 要求 float16/bfloat16 但模型以 float32 初始化

       修复：pip uninstall flash-attn，清理 module cache

       脚本第三次运行
       问题三：verl 默认 attn_implementation="flash_attention_2" → ImportError (FA2 toggled on)

       修复：在 Hydra 参数中加入 +actor_rollout_ref.model.override_config.attn_implementation=sdpa
```
