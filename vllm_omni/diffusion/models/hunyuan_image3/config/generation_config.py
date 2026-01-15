generation_config = {
  "bot_task": "image",
  "diff_guidance_scale": 5.0,
  "diff_infer_steps": 50,
  "disable_compile": True,
  "do_sample": True,
  "drop_think": False,
  "eos_token_id": [
    127957
  ],
  "flow_shift": 3.0,
  "max_length": 12800,
  "moe_impl": "eager",
  "pad_token_id": 128009,
  "sequence_template": "pretrain",
  "temperature": 0.6,
  "top_k": 1024,
  "top_p": 0.95,
  "use_system_prompt": "None"
}
#self.model.generation_config
#/usr/local/lib/python3.12/dist-packages/transformers/generation/configuration_utils.py class GenerationConfig(PushToHubMixin)