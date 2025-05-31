from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
model_name = "/mnt/data/Qwen-7B-Chat"
prompt = "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道。请简要解释 "

tokenizer = AutoTokenizer.from_pretrained(
     model_name,
     trust_remote_code=True
 )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
).eval()

inputs = tokenizer(prompt, return_tensors="pt").input_ids

streamer = TextStreamer(tokenizer)

outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)