from enum import Enum

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class TuningTactics(Enum):
    LORA_TUNING = 'lora_tuning'
    P_TUNING = 'p_tuning'
    PROMPT_TUNING = 'prompt_tuning'
    PREFIX_TUNING = 'prefix_tuning'


class PluginPEFT(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "replace_model",
    }

    def replace_model(self, ctx: TrainContext):
        from peft import LoraConfig, get_peft_model, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig 
        tuning_tactics = ctx.cfg.get("tuning_tactics")
        if tuning_tactics == TuningTactics.LORA_TUNING:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "key", "value"]
            )
        elif tuning_tactics == TuningTactics.LORA_TUNING:
            peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        elif tuning_tactics == TuningTactics.PROMPT_TUNING:
            peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        elif tuning_tactics == TuningTactics.PREFIX_TUNING:
            peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

        peft_model = get_peft_model(ctx.model, peft_config)
        for name, param in peft_model.named_parameters():
            if "classify" in name or "crf" in name:
                param.requires_grad = True
            elif "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        peft_model.print_trainable_parameters()
        
        ctx.model = peft_model
