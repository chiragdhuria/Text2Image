import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import numpy as np

# Create the tkinter app
app = tk.Tk()
app.geometry("532x632")
app.title("Text2Image Synthesizer")
ctk.set_appearance_mode("dark")

# Create the CTkEntry widget with app (master) as its parent
prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

# Create the CTkLabel widget with app (master) as its parent
lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

# Initialize other components
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

# Bypass the safety checker
def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker

def is_nsfw_prompt(prompt_text):
    # A simple check for potentially NSFW content in the prompt
    nsfw_keywords = ["nude", "nudity", "sex", "porn", "explicit", "nsfw"]
    return any(keyword in prompt_text.lower() for keyword in nsfw_keywords)

def normalize_image(image):
    # Ensure no NaNs or infinities
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    # Clip values to [0, 1] and scale to [0, 255]
    image = np.clip(image, 0, 1)
    image = (image * 255).round().astype(np.uint8)
    return image

def generate():
    prompt_text = prompt.get()
    if is_nsfw_prompt(prompt_text):
        print("NSFW content detected in the prompt. Please use a different prompt.")
        return

    try:
        with autocast(device):
            result = pipe(prompt_text, guidance_scale=8.5)
        
        if 'images' in result:
            image = result['images'][0]
            image = normalize_image(np.array(image))
            pil_image = Image.fromarray(image)
            pil_image.save('generatedimage.png')
            img = ctk.CTkImage(pil_image, size=(512, 512))
            lmain.configure(image=img)
            lmain.image = img  # Keep a reference to avoid garbage collection
        else:
            print("No images generated. Please try again with a different prompt.")
    except Exception as e:
        print(f"Error generating image: {e}")

# Create the CTkButton widget with app (master) as its parent
trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

# Start the tkinter main loop
app.mainloop()
