import gradio as gr
import diffuseit as di
import helpers as h

def txt2img_tab():
    txt2img = gr.Blocks()
    with txt2img:
        with gr.Row():
            with gr.Column():
                with gr.Row():            
                    models = gr.Dropdown(label="Select Model", choices=["pocsd15", "pocsdxl"], value="pocsd15")
                    img_gen_qty = gr.Slider(label="Number of Images", minimum=1, maximum=4, value=1, step=1)
                with gr.Row():
                    with gr.Column():
                        width = gr.Slider(label="Width", minimum=512, maximum=1024, value=512, step=64)
                        height = gr.Slider(label="Height", minimum=512, maximum=1024, value=512, step=64)
                    with gr.Column():
                        orientation = gr.Radio(label="Orientation", choices=["Landscape", "Portrait"], value="Landscape")
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt")
                    negative_prompt = gr.Textbox(label="Negative Prompt")
                with gr.Row():
                    submit = gr.Button("Run", size="sm")
            with gr.Column():
                gallery = gr.Image(label="Results", show_label=True)
        orientation.change(
            fn=h.set_orientation,
            inputs=[orientation, width, height],
            outputs=[width, height],
        )
        submit.click(
            fn=di.txt2img,
            inputs=[prompt, negative_prompt,models],
            outputs=gallery,
            api_name="text_to_image",
            show_progress=True,
            queue=True,
        )
    return txt2img

def txt2img_chat_tab():
    txt2img_chat = gr.Blocks()
    with txt2img_chat:
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(placeholder="Enter your message here...")
        #submit = gr.Button("Send")
        with gr.Sidebar():
            models = gr.Dropdown(label="Select Model", choices=["pocsd15", "pocsdxl"], value="pocsd15")
            width = gr.Slider(label="Width", minimum=512, maximum=1024, value=512, step=64)
            height = gr.Slider(label="Height", minimum=512, maximum=1024, value=512, step=64)
            orientation = gr.Radio(label="Orientation", choices=["Landscape", "Portrait"], value="Landscape")
        
        def send_message(message, chat_history, models, width, height):
            response = di.txt2img(message,"", models,width, height)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": gr.Image(response)})
            return "", chat_history
        
        orientation.change(
            fn=h.set_orientation,
            inputs=[orientation, width, height],
            outputs=[width, height],
        )
        msg.submit(send_message, [msg, chatbot,models,width,height], [msg, chatbot])
    return txt2img_chat