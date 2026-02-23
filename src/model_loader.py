"""모델 로드 유틸리티."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str = "ibm-granite/granite-3.1-3b-a800m-instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    HuggingFace 모델과 토크나이저를 로드.
    Jetson AGX Orin 64GB 환경 최적화.
    """
    print(f"[ModelLoader] Loading model: {model_name}")
    print(f"[ModelLoader] Device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # 모델 메모리 사용량 출력
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[ModelLoader] GPU Memory - Allocated: {mem_alloc:.2f} GB, Reserved: {mem_reserved:.2f} GB")

    return model, tokenizer

