import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, WhisperProcessor, WhisperForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
from scipy.signal import resample

from transformers import LlamaForCausalLM, AutoTokenizer
from llava_gpt_neox import LlavaGPTNeoXForCausalLM
from PIL import Image
device='cuda'

llm = 'LearnItAnyway/llava-polyglot-ko-1.3b-hf'
tokenizer = AutoTokenizer.from_pretrained(llm)
model = LlavaGPTNeoXForCausalLM.from_pretrained(llm).half().to(device)


image_file = f"example.jpg"
from transformers import CLIPImageProcessor
image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
image = Image.open(image_file).convert('RGB')
image = image_processor(image, return_tensors='pt')['pixel_values'][0].half().cuda()
model.gpt_neox.vision_tower.config.im_patch_token = tokenizer.encode("<im_patch>")[0]
model.gpt_neox.vision_tower.config.use_im_start_end=False

def greedy_search(
    inputs, max_new_length, image=image
):
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    pkv = model(input_ids=inputs, images=image.unsqueeze(0), use_cache=True)['past_key_values']
    counter = 0
    while True:
        outputs = model(input_ids=inputs, past_key_values=pkv, use_cache=True)

        pkv = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # update generated ids, model inputs, and length for next step
        inputs = torch.cat([inputs, next_tokens[:, None]], dim=-1)
        counter += 1
        if next_tokens[0].item()==eos_token_id  or counter >= max_new_length:
            break
    return inputs

# Define function for generating response
def generate_response(input_text):
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)[-512*3:]
    with torch.no_grad():
        outputs = greedy_search(inputs, max_new_length=200, )
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response
im_tokens = ''.join(['<im_patch>' for _ in range(256)])

input_text = f'{im_tokens}이미지 물체의 이름이 뭐야?' 
input_text = f"{tokenizer.bos_token}### USER: {input_text}\n### ASSISTANT: "
response = generate_response(input_text)
print(response)

input_text = f'{im_tokens}이미지 속 물체의 색은?' 
input_text = f"{tokenizer.bos_token}### USER: {input_text}\n### ASSISTANT: "
response = generate_response(input_text)
print(response)
