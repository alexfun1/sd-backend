import gradio as gr
import helper as h
import diffuser as d


default_prompt = "score_9, score_8_up, score_7_up, masterpiece, 8K, realistic, Expressiveh, mature woman, 45 yo, short dark brown hair, pixie haircut, curvy, slightly chubby, large breasts, wide hips, thick thigh, hourglass figure, red high heel sandals sitting on bar stool, wide spread legs, pubic hair <lora:add-detail-xl:1>"
default_neg_prompt = "ugly, deformed, noisy, blurry, low contrast, text, 3d, cgi, render, anime,  big forehead, long neck,watermark,extra fingers, mutated body, mutated palm, deformed hands, 4 fingers, deformed fingernails"
  

iface = gr.Blocks()
with iface:
    with gr.Row():
        with gr.Column():
            models = gr.Dropdown(
                h.load_models_from_dir(),
                label="Select Model",
                #value="Model 1"
            )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                value=default_prompt
            )
            neg_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Enter your negative prompt here...",
                value=default_neg_prompt
            )
            width = gr.Slider(
                minimum=512,
                maximum=1024,
                step=64,
                label="Width",
                value=512
            )
            height = gr.Slider(
                minimum=512,
                maximum=1024,
                step=64,
                label="Height",
                value=512
            )
            num_images = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                label="Number of Images",
                value=1
            )
            submit = gr.Button("Submit")
        with gr.Column():
            galery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery",
                height=300
            )
        submit.click(
            fn=d.txt2img,
            inputs=[models, prompt, neg_prompt, width, height, num_images],
            outputs=galery,
            api_name="generate",
            show_progress=True,
        )
try:
    iface.launch()
except KeyboardInterrupt:
    d.flush()