## Finetune Huggingface model with Unsloth (Windows)

### Environment setup

##### Create a new conda environment and activate it.

```
conda create -n env1
conda activate env1
```

##### Install the following libraries

```
pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
pip install triton-windows
pip install --no-deps bitsandbytes accelerate transformers xformers peft trl
pip install sentencepiece protobuf datasets huggingface_hub
```

- Check your CUDA driver version by running `nvidia-smi` in cmd prompt. Then install pytorch library compatible with your driver.

- `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`

- Load python environment in console and verify CUDA availability

```python
import torch 
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```
---

### Execution

<blockquote>
If the kernel dies or restarts at any time, simply rerun the steps.
</blockquote>

#### Step 1: Load model with 4-bit quantization

Make sure that `MODEL_NAME` below is a base model version from huggingface
```python
from unsloth import FastModel
MODEL_NAME = "google/medgemma-4b-it"
MAX_SEQ_LENGTH = 512

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect: bfloat16 for Ampere+, float16 for older GPUs
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)
```

#### Step 2: Configuring LoRA Adapters

```python
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)
```

#### Number of trainable parameters (should be ~ 1%)

```
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.4f}%)")
```

#### Step 3: Dataset preparation (testing with FineTome-100k dataset)

```python
from datasets import load_dataset

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
print(f"Dataset size: {len(dataset):,} examples")
```

#### Chat Templates: The Secret Sauce 

```python
from unsloth.chat_templates import get_chat_template,standardize_data_formats
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3", # Change to 'llama3' for llama 3 models
)
dataset = standardize_data_formats(dataset)
```

#### Step 4: Formating dataset
```python
def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)
```

#### Step 5: Train model

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)
```

#### Step 7: Train Only on Responses (*template again must match the model type - gemma*)
A crucial optimization: we only compute loss on assistant responses, not user questions
```python
from unsloth.chat_templates import train_on_responses_only
# For Gemma 3 model
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
```

```python
# For llama 3 model
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

#### Step 8: Training Time - To resume run `trainer.train(resume_from_checkpoint = True)`
```python
trainer_stats = trainer.train()
print(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f}s")
```
<blockquote>
You’ll see a progress bar showing:

- Current step
- Training loss (should decrease over time)
- Estimated time remaining
</blockquote>

#### Step 9: Run model via Unsloth native inference.
```python
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    }]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    tokenize = True,
    return_tensors = "pt",
    return_dict = True,
)
outputs = model.generate(
    **inputs.to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)
```

#### Step 10: Saving Your Model

##### Option 1: Save LoRA Adapters
```python
model.save_pretrained("gemma_3_lora") # Local saving
tokenizer.save_pretrained("gemma_3_lora")

# model.push_to_hub("HF_ACCOUNT/gemma_3_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma_3_lora", token = "YOUR_HF_TOKEN") # Online saving
```

##### To test: Load the LoRA adapters we just saved for inference, set False to True:
```python

from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name = "gemma_3_lora", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    load_in_4bit = True,
)

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-3?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    tokenize = True,
    return_tensors = "pt",
    return_dict = True,
)

from transformers import TextStreamer
_ = model.generate(
    **inputs.to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
```
##### Option 2: Saving to float16 for VLLM (save it in the folder gemma-3-finetune)

```python
model.save_pretrained_merged("gemma-3-finetune", tokenizer)
```
##### Option 3: Saving to float16 for VLLM (save it in the folder gemma-3-finetune)

```python
model.save_pretrained_merged("gemma-3-finetune", tokenizer)
```
Upload / push to your Hugging Face account
```python
model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-3-finetune", tokenizer,
        token = "YOUR_HF_TOKEN"
    )
```

##### Option 4: GGUF / llama.cpp Conversion
```python
model.save_pretrained_gguf(
    "gemma_3_finetune",
    tokenizer,
    quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
)
```
Push to GGUF to your Hugging Face account
```python
model.push_to_hub_gguf(
    "HF_ACCOUNT/gemma_3_finetune",
    tokenizer,
    quantization_method = "Q8_0", # Only Q8_0, BF16, F16 supported
    token = "YOUR_HF_TOKEN",
)
```
