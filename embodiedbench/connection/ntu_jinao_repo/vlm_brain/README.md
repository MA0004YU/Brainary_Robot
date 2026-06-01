# VLM Brain: Qwen3-VL Skill Blueprint Loop

Stage IV connects Qwen3-VL to the existing skill blueprint executor and predictor_v1 feedback. It does not start Isaac Sim.

## Install

```bash
pip install -r vlm_brain/requirements_vlm.txt
```

If `Qwen3VLForConditionalGeneration` is unavailable, install the latest Transformers source:

```bash
pip install git+https://github.com/huggingface/transformers
```

## Dry Run

Construct the generation prompt without loading Qwen3-VL:

```bash
python vlm_brain/run_vlm_inference.py \
  --config vlm_brain/configs/qwen3vl_8b_infer.json \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --task "pick the cube and place it on the target" \
  --output_json vlm_brain/outputs/dry_run_blueprint.json \
  --dry_run true \
  --validate false
```

## Download Model

```bash
python vlm_brain/download_model.py \
  --model_name Qwen/Qwen3-VL-8B-Instruct
```

Weights are cached by Hugging Face and must not be stored in this repository.

## VLM Inference

```bash
python vlm_brain/run_vlm_inference.py \
  --config vlm_brain/configs/qwen3vl_8b_infer.json \
  --image /path/to/scene.png \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --task "pick the cube and place it on the target" \
  --output_json vlm_brain/outputs/generated_blueprint.json \
  --validate true
```

Outputs include `raw_response.txt`, `parsed_blueprint.json`, `validation_report.json`, and `generated_blueprint.json` if valid.

## Predictor Bridge

```bash
python vlm_brain/predictor_bridge.py \
  --blueprint vlm_brain/examples/sample_blueprint.json \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --output_feedback vlm_brain/outputs/mock_predictor_feedback.json \
  --mock true
```

With a trained predictor:

```bash
python vlm_brain/predictor_bridge.py \
  --blueprint vlm_brain/outputs/generated_blueprint.json \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --predictor_checkpoint skill_performance_predictor/outputs/predictor_v1/best_model.pt \
  --predictor_data_dir skill_performance_predictor/data/predictor_v1 \
  --output_feedback vlm_brain/outputs/predictor_feedback.json
```

## Closed Loop

```bash
python vlm_brain/vlm_predictor_loop.py \
  --config vlm_brain/configs/qwen3vl_8b_infer.json \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --task "pick the cube and place it on the target" \
  --output_dir vlm_brain/outputs/loop_dry_run \
  --dry_run true \
  --mock_predictor true \
  --max_refine_iters 1
```

Dry run saves `initial_blueprint.json`, `predictor_feedback_iter0.json`, optional `revised_blueprint_iter1.json`, and `loop_summary.json`.

## SFT Dataset

```bash
python vlm_brain/build_vlm_sft_dataset.py \
  --episodes_jsonl /path/to/episodes.jsonl \
  --vlm_inputs_jsonl /path/to/vlm_inputs.jsonl \
  --predictor_feedback_dir /path/to/feedbacks \
  --output_train vlm_brain/data/sft_train.jsonl \
  --output_val vlm_brain/data/sft_val.jsonl
```

Large JSONL files and images are ignored by Git. The dataset stores image paths rather than copying images.

## LoRA Dry Run

```bash
python vlm_brain/train_qwen3vl_lora.py \
  --config vlm_brain/configs/qwen3vl_lora_train.json \
  --train_jsonl vlm_brain/data/sft_train.jsonl \
  --val_jsonl vlm_brain/data/sft_val.jsonl \
  --dry_run true
```

The Stage IV skeleton validates config/data and reports sample counts. Full training is a later step.
