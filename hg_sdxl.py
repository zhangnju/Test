from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import time
import torch
import numpy as np

def elapsed_time(pipeline, prompt, nb_pass=2, num_inference_steps=10):
    # warmup
    images = pipeline(prompt, num_inference_steps=10).images
    
    start = time.time()
    for _ in range(nb_pass):
        pipeline(prompt, guidance_scale=0.0, num_inference_steps=num_inference_steps,width=512,height=512)
    end = time.time()
    return (end - start) / (nb_pass*num_inference_steps)
np.random.seed(112)    

#pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16, variant="fp16")
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16, variant="fp16")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

latency = elapsed_time(pipe, prompt)
Throughput = 1 / latency
print('inference latency %.3f s'%latency)
print("Throughput: {:.3f} fps".format(Throughput))