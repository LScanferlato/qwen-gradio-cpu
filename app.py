import torch
from diffusers import DiffusionPipeline
from PIL import Image
import gradio as gr
import subprocess

# Carica il modello Qwen (usa la pipeline generica)
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

def edit_multi_image(img_main, img_outfit, img_pose, prompt, steps, guidance):
    if img_main is None:
        return None

    images = []
    for img in [img_main, img_outfit, img_pose]:
        if img is not None:
            images.append(img.convert("RGB"))

    # Se c'è solo una immagine → singolo input, altrimenti lista
    image_input = images[0] if len(images) == 1 else images

    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompt,
            image=image_input,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )

    return result.images[0]

# Preset
def preset_ritratto():
    return (
        "Migliora il ritratto: pelle naturale, luce morbida, "
        "contrasto equilibrato, resa realistica e dettagliata."
    )

def preset_fashion():
    return (
        "Stile fashion editoriale: colori vividi, outfit valorizzato, "
        "illuminazione da studio, look moderno e pulito."
    )

def preset_prodotto():
    return (
        "Foto prodotto professionale: sfondo pulito, luce uniforme, "
        "alta nitidezza, resa commerciale e dettagli chiari."
    )

# Monitor GPU
def gpu_status():
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True)
        return out
    except Exception as e:
        return f"Errore nel leggere nvidia-smi: {e}"

with gr.Blocks(title="Qwen Image Edit – Multi Image + Preset + GPU Monitor") as ui:
    gr.Markdown("## Qwen Image Edit – Multi‑Image con Preset e Monitor GPU")

    with gr.Row():
        img_main = gr.Image(type="pil", label="Immagine principale (obbligatoria)")
        img_outfit = gr.Image(type="pil", label="Outfit (opzionale)")
        img_pose = gr.Image(type="pil", label="Posa (opzionale)")

    prompt = gr.Textbox(label="Prompt di modifica")

    with gr.Row():
        btn_ritratto = gr.Button("Preset Ritratto")
        btn_fashion = gr.Button("Preset Fashion")
        btn_prodotto = gr.Button("Preset Prodotto")

    btn_ritratto.click(fn=preset_ritratto, outputs=prompt)
    btn_fashion.click(fn=preset_fashion, outputs=prompt)
    btn_prodotto.click(fn=preset_prodotto, outputs=prompt)

    steps = gr.Slider(5, 50, value=24, step=1, label="Inference Steps")
    guidance = gr.Slider(0, 10, value=4.0, step=0.1, label="Guidance Scale")

    output_img = gr.Image(label="Risultato")

    run_btn = gr.Button("Genera immagine")
    run_btn.click(
        fn=edit_multi_image,
        inputs=[img_main, img_outfit, img_pose, prompt, steps, guidance],
        outputs=output_img,
    )

    gr.Markdown("### Monitor GPU")
    gpu_btn = gr.Button("Mostra stato GPU")
    gpu_out = gr.Textbox(label="Output nvidia-smi")
    gpu_btn.click(fn=gpu_status, outputs=gpu_out)

ui.launch(server_name="0.0.0.0", server_port=7860)
