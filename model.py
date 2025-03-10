from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model


model_name = "openvla/openvla-7b"
model = AutoModelForVision2Seq.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

lora_config = LoraConfig(
  r = 32,
  lora_alpha = 16,
  target_modules = ["q_proj", "v_proj"],
  lora_dropout = 0.05,
  bias = "none",
  task_type = "SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)