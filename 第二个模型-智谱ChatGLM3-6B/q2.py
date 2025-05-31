from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
model_name = "/mnt/data/chatglm3-6b"
prompt = "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上。请简要解释 "

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