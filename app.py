import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import gradio as gr

# Carica il modello
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

def edit_multi_image(img_main, img_outfit, img_pose, prompt, steps, guidance):
    # Almeno l'immagine principale deve esserci
    if img_main is None:
        return None

    # Converti tutte le immagini presenti
    images = []
    for img in [img_main, img_outfit, img_pose]:
        if img is not None:
            images.append(img.convert("RGB"))

    # Se c'è solo la principale, funziona comunque
    with torch.autocast(device):
        result = pipe(
            prompt=prompt,
            image=images,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )

    return result.images[0]

# Interfaccia Gradio
ui = gr.Interface(
    fn=edit_multi_image,
    inputs=[
        gr.Image(type="pil", label="Immagine principale (obbligatoria)"),
        gr.Image(type="pil", label="Outfit (opzionale)"),
        gr.Image(type="pil", label="Posa (opzionale)"),
        gr.Textbox(label="Prompt di modifica"),
        gr.Slider(5, 50, value=24, step=1, label="Inference Steps"),
        gr.Slider(0, 10, value=4.0, step=0.1, label="Guidance Scale"),
    ],
    outputs=gr.Image(label="Risultato"),
    title="Qwen Image Edit – Multi‑Image (Pose / Outfit)",
    description=(
        "Carica una o più immagini: soggetto principale, outfit e posa. "
        "Il modello combinerà gli elementi secondo il prompt."
    ),
)

ui.launch(server_name="0.0.0.0", server_port=7860)
