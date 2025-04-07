import gradio as gr
import web as w
import diffuseit as di


def main():
    app = gr.TabbedInterface(
        [w.txt2img_chat_tab(), w.txt2img_tab()],
        ["Chat", "Text to Image"],
        # ["Image to Image", "Inpaint", "Upscale"],
        title="Simple Diffusion Web UI",
        theme=gr.themes.Base(),
        #css="styles.css",
    )
    app.launch(share=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Handle graceful exit
        # Perform any necessary cleanup here
        di.mem_flush()
        print("\nApplication interrupted. Exiting gracefully.")
