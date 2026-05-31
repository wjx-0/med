# Qwen2.5-1.5B QLoRA SFT

This folder contains the local-development / server-experiment SFT setup for the CURE-Med reproduction.

## Data

Convert the original CURE-Med SFT JSONL files into LLaMA-Factory OpenAI messages format:

```bash
cd /Users/wjx/Downloads/java/llm/cure-med/med
python sft/convert_sft_to_llamafactory.py
```

The converted files are written to:

```txt
sft/llamafactory_data/
```

Expected counts:

```txt
Spanish: 852
Yoruba: 1328
Spanish + Yoruba: 2180
```

## Server Setup

On the server, keep the project at:

```txt
/root/med
```

Install LLaMA-Factory separately:

```bash
cd /root
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Train

Smoke test:

```bash
cd /root/LLaMA-Factory
llamafactory-cli train /root/med/sft/configs/qwen2.5_1.5B_qlora_spanish_smoke.yaml
```

Spanish full 1 epoch:

```bash
llamafactory-cli train /root/med/sft/configs/qwen2.5_1.5B_qlora_spanish.yaml
```

Spanish + Yoruba 1 epoch:

```bash
llamafactory-cli train /root/med/sft/configs/qwen2.5_1.5B_qlora_spanish_yoruba.yaml
```

## Inference

Run SFT LoRA inference on the server:

```bash
cd /root/med/sft/inference
python qwen2.5_1.5B_instruct_sft_lora_inference.py
```

For the Spanish adapter, use:

```txt
Spanish
-1
16
```

The default adapter and output paths are:

```txt
adapter: /root/med/saves/qwen2.5_1.5B_qlora_spanish
output:  /root/med/outputs/sft/qwen2.5_1.5B_qlora_spanish/generations
```

For a different adapter, pass explicit paths:

```bash
python qwen2.5_1.5B_instruct_sft_lora_inference.py \
  --adapter_dir /root/med/saves/qwen2.5_1.5B_qlora_spanish_yoruba \
  --output_dir /root/med/outputs/sft/qwen2.5_1.5B_qlora_spanish_yoruba/generations
```

## Local Evaluation

After downloading `generations/` from the server to local `med/outputs/sft/...`, run the shared judge scripts against the SFT output directory:

```bash
cd /Users/wjx/Downloads/java/llm/cure-med/med/baseline_inference/qwen_models
conda activate med-eval

python evaluating_logic_of_generated_solutions.py --language Spanish --run_dir ../../outputs/sft/qwen2.5_1.5B_qlora_spanish
python evaluating_language_of_generated_solutions.py --language Spanish --run_dir ../../outputs/sft/qwen2.5_1.5B_qlora_spanish --answer_only
python compute_logic_accuracy.py --run_dir ../../outputs/sft/qwen2.5_1.5B_qlora_spanish
python compute_language_accuracy.py --run_dir ../../outputs/sft/qwen2.5_1.5B_qlora_spanish --answer_only
```

Baseline Spanish reference:

```txt
Logic: 15.24
Language: 58.10
```
