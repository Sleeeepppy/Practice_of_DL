from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map = "cuda",
    torch_dtype = "auto",
    trust_remote_code = True
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

prompt = "Writting a sentence to discribe yourself.<|im_start|>assistant"
# input prompt to tokenizer
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
# generate the text
generate_text = model.generate(input_ids=input_ids, max_new_tokens=20)
print(tokenizer.decode(generate_text[0]))