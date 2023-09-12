from dataclasses import dataclass

@dataclass
class RunConfig:
    save_steps = 500
    prior_loss_weight = 1.0
    with_prior_preservation = True
    freeze_model = "crossattn_kv"
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    revision = None
    tokenizer_name = None
    iv_modifier_tokens = None
    iv_initializer_tokens = None
    output_dir = None
    seed = 42
    resolution = 512
    iv_train_batch_size = 4
    ft_train_batch_size = 1
    iv_max_train_steps = 200
    ft_max_train_steps = 800
    iv_gradient_accumulation_steps = 1
    ft_gradient_accumulation_steps = 2
    gradient_checkpointing = False
    iv_lr = 0.0005
    ft_lr = 1e-05
    scale_lr = True
    lr_scheduler = "constant"
    lr_warmup_steps = 0
    dataloader_num_workers = 0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 0.01
    adam_epsilon = 1e-08
    push_to_hub = False
    hub_token = None
    hub_model_id = None
    logging_dir = "logs"
    mixed_precision = "no"
    allow_tf32 = False
    report_to = "tensorboard"
    train_prompt = None
    num_validation_images = 1
    local_rank = -1
    checkpointing_steps = 500
    checkpoints_total_limit = None
    enable_xformers_memory_efficient_attention = False
    image_path = None
    mask_image = None
    semantic_dict = None
    addition_tokens = None
    lambda_factor = 0
