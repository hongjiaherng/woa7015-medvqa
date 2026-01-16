from peft import LoraConfig, get_peft_model
from transformers import BlipForQuestionAnswering, BlipProcessor


def build_blip_with_lora(
    model_name: str = "Salesforce/blip-vqa-base",
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
):
    """
    Builds a BLIP model with LoRA adapters customized for MedVQA (SLAKE).
    Returns: (processor, lora_model)
    """
    processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
    model = BlipForQuestionAnswering.from_pretrained(model_name)

    # 1. TARGET MODULES
    # The 'Salesforce/blip-vqa-base' model contains these linear layers:
    # {'query', 'key', 'value', 'dense', 'decoder', 'projection', 'qkv', 'fc1', 'fc2'}
    #
    # We add "key" to the list to fully adapt the attention pattern.
    # - "query", "key", "value": Text Decoder Attention.
    #    - Q & K determine WHERE to look (Attention Scores).
    #    - V determines WHAT to extract.
    #    - Fine-tuning all three helps the model align medical words with specific image regions.
    #                     Essential for learning medical terminology (e.g., "pneumonia").
    # - "qkv": Vision Encoder Attention.
    #                     Essential for adapting to medical images (X-ray/CT) vs natural images.
    target_modules = ["query", "key", "value", "qkv"]

    # 2. TASK TYPE
    # Supported PEFT TaskTypes include:
    # - SEQ_CLS (Sequence Classification)
    # - SEQ_2_SEQ_LM (Sequence-to-Sequence, e.g., T5)
    # - CAUSAL_LM (Causal Language Modeling, e.g., GPT, LLaMA)
    # - TOKEN_CLS (Token Classification, e.g., NER)
    # - QUESTION_ANS (Extractive QA, e.g., SQuAD - selects start/end index)
    # - FEATURE_EXTRACTION (Getting embeddings)
    #
    # SELECTION: None (Custom/Multimodal)
    # Why? BLIP is Multimodal (Image + Text). Standard types like CAUSAL_LM
    # often enforce a text-only input pipeline, which drops 'pixel_values'.
    # Setting None ensures PEFT adds adapters without breaking the image inputs.
    task_type = None

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return processor, model
