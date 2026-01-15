from peft import LoraConfig, get_peft_model
from transformers import BlipForQuestionAnswering, BlipProcessor


def build_blip_with_lora(
    model_name: str = "Salesforce/blip-vqa-base",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
):
    """
    Returns: (processor, lora_model)
    """
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)

    # LoRA config (generic). BLIP module naming varies by version.
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_cfg)
    return processor, model
