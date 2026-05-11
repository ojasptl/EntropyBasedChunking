import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def initialize_llm():
    print("Loading LLM (Qwen2.5-3B-Instruct)...")
    llm_model_name = "Qwen/Qwen2.5-3B-Instruct"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True, low_cpu_mem_usage=True)

    if device == "cpu":
        llm_model = llm_model.to(device)

    llm_model.config.pad_token_id = llm_tokenizer.pad_token_id
    llm_model.eval()
    return llm_model, llm_tokenizer


def generate_answer(query: str, context: str, llm_model, llm_tokenizer) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a precise technical assistant. "
            "Answer questions directly based ONLY on the provided context. "
            "Be concise and start with the direct answer."
        )},
        {"role": "user", "content": (
            f"Context:\n{context[:3000]}\n\n"
            f"Question: {query}\n\n"
            "Instructions:\n"
            "- Answer the exact question asked\n"
            "- Start with the direct answer in the first sentence\n"
            "- Only include relevant supporting details"
        )},
    ]

    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs, max_new_tokens=512, temperature=0.3, do_sample=True,
            top_p=0.9, repetition_penalty=1.1,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id)

    return llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)