import warnings
import gradio as gr
import subprocess
import atexit
import sys
from pathlib import Path
from typing import Optional, List, Dict

from supervisor import LangChainSupervisor


#Filter:
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")


tools_server_process = None

def start_tools_server():
    """Starts the tools server subprocess"""
    global tools_server_process
    print("GUI: Starting Tools-Server...")
    tools_server_process = subprocess.Popen([sys.executable, "src/supervisor/tools_server.py"])
    print(f"GUI: Tools-Server started with PID: {tools_server_process.pid}")
    return tools_server_process.pid

def cleanup():
    """Terminates the tools server on exit"""
    global tools_server_process
    if tools_server_process:
        print("GUI: Terminating Tools-Server...")
        tools_server_process.terminate()
        tools_server_process.wait()
        print("GUI: Tools-Server terminated.")

atexit.register(cleanup)

server_pid = start_tools_server()
supervisor = LangChainSupervisor()

def process_message_streaming(
    history: List[Dict[str, str]],
    current_input_path: Optional[str],
    current_output_path: Optional[str]
):
    """
    Processes user message based on the last entry in history.
    """
    if not history or history[-1]["role"] != "user":
        yield history, gr.update(), gr.update(), gr.update(), gr.update()
        return
        
    message = history[-1]["content"] 
    
    history.append({"role": "assistant", "content": "Thinking..."})
    yield history, gr.update(), gr.update(), gr.update(), gr.update()
    
    response_state = supervisor.process_user_message(message)
    
    last_message = response_state['messages'][-1]
    response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    history[-1]["content"] = response_text
    
    input_update = gr.update()
    output_update = gr.update()
    input_path_text = gr.update()
    output_path_text = gr.update()
    
    if response_state.get('original_image_path'):
        new_input = response_state['original_image_path']
        if new_input != current_input_path:
            print(f"GUI: Input image CHANGED: {current_input_path} -> {new_input}")
            input_update = gr.update(value=new_input)
            input_path_text = gr.update(value=f"{new_input}")
        else:
            print(f"GUI: Input image UNCHANGED: {current_input_path}")
    
    output_path = (response_state.get('rendered_document_path') or 
                   response_state.get('working_image_path'))
    if output_path and output_path != current_output_path:
        print(f"GUI: Output image CHANGED: {current_output_path} -> {output_path}")
        output_update = gr.update(value=output_path)
        output_path_text = gr.update(value=f"{output_path}")
    else:
        print(f"GUI: Output image UNCHANGED: {current_output_path}")
    
    yield history, input_update, output_update, input_path_text, output_path_text

def add_user_message(message: str, history: List[Dict[str, str]]):
    """
    Fügt die User-Nachricht sofort zur History hinzu und leert das Eingabefeld.
    """
    if not message.strip():
        return gr.update(), history
    
    history.append({"role": "user", "content": message})
    
    return gr.update(value=""), history

def load_image_from_file(file, history):
    """
    Handles file upload using Gradio 6.0 "messages" format
    """
    if file is None:
        yield history, gr.update(), gr.update(), gr.update(), gr.update()
        return
    
    file_path = file.name if hasattr(file, 'name') else str(file)
    file_name = Path(file_path).name
    
    history.append({"role": "user", "content": f"Load image: {file_name}"})
    yield history, gr.update(), gr.update(), gr.update(), gr.update()
    
    history.append({"role": "assistant", "content": "Loading image..."})
    yield history, gr.update(), gr.update(), gr.update(), gr.update()
    
    message = f"load image from {file_path}"
    response_state = supervisor.process_user_message(message)
    
    last_message = response_state['messages'][-1]
    response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    history[-1]["content"] = response_text
    
    input_update = gr.update()
    output_update = gr.update()
    input_path_text = gr.update()
    output_path_text = gr.update()
    
    if response_state.get('original_image_path'):
        new_input_path = response_state['original_image_path']
        input_update = gr.update(value=new_input_path)
        output_update = gr.update(value=new_input_path) 
        input_path_text = gr.update(value=f"{new_input_path}")
        output_path_text = gr.update(value=f"{new_input_path}")
        print(f"GUI: New image loaded - Input and Output reset to: {new_input_path}")
    
    yield history, input_update, output_update, input_path_text, output_path_text


def capture_from_webcam_ui(history):
    """
    Handles webcam capture using Gradio 6.0 "messages" format
    """
    history.append({"role": "user", "content": "Capture from webcam"})
    yield history, gr.update(), gr.update(), gr.update(), gr.update()
    
    history.append({"role": "assistant", "content": "Opening webcam... (Press SPACE to capture, ESC to cancel)"})
    yield history, gr.update(), gr.update(), gr.update(), gr.update()
    
    message = "capture image from webcam"
    response_state = supervisor.process_user_message(message)
    
    last_message = response_state['messages'][-1]
    response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    history[-1]["content"] = response_text
    
    input_update = gr.update()
    output_update = gr.update()
    input_path_text = gr.update()
    output_path_text = gr.update()
    
    if response_state.get('original_image_path'):
        new_input_path = response_state['original_image_path']
        input_update = gr.update(value=new_input_path)
        output_update = gr.update(value=new_input_path)
        input_path_text = gr.update(value=f"{new_input_path}")
        output_path_text = gr.update(value=f"{new_input_path}")
        print(f"GUI: Webcam captured - Input and Output reset to: {new_input_path}")
    
    yield history, input_update, output_update, input_path_text, output_path_text


def create_ui():
    """Creates the Gradio interface with updated components for Gradio 6 compliance"""
    
    custom_css = """
        .gradio-container {
            max-width: 1800px !important;
        }
        .large-chat {
            font-size: 16px !important;
        }
        .large-chat .message {
            font-size: 16px !important;
        }
        .large-chat p {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        """

    # NOTE: 'theme' and 'css' MUST remain in Blocks() for Gradio 5.50.
    # Moving them to launch() currently causes a TypeError in this version.
    # We accept the DeprecationWarning here to keep the app crashing-free.
    with gr.Blocks(title="Smart Translator", theme=gr.themes.Soft(), css=custom_css) as demo:
        
        gr.Markdown(
            """
            # Smart Translator
            Translate documents while preserving layout, fonts, and formatting
            """
        )
        
        with gr.Row():
            server_status = gr.Markdown(f" **Tools Server:** Running (PID: {server_pid})")
        
        gr.Markdown("---")
        
        with gr.Row():
            
            with gr.Column(scale=1):
                gr.Markdown("### Chat")
                
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": "Hello! Welcome to Smart Translator. Send a message or upload an image to get started."}],
                    height=600,
                    show_label=False,
                    avatar_images=(None, None),
                    elem_classes="large-chat",
                    type="messages",      
                    allow_tags=False       
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                gr.Markdown("#### Quick Actions")
                
                with gr.Row():
                    file_btn = gr.UploadButton(
                        "Load Image",
                        file_types=["image"],
                        file_count="single"
                    )
                    webcam_btn = gr.Button("Webcam", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Input Document")
                input_image = gr.Image(
                    type="filepath",
                    height=600,
                    show_label=False,
                    interactive=False
                )
                
                input_image_path_display = gr.Textbox(
                    value="No image loaded",
                    label="Image Path",
                    interactive=False,
                    lines=2,
                    max_lines=2,
                    show_copy_button=True
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Processed Document")
                output_image = gr.Image(
                    type="filepath",
                    height=600,
                    show_label=False,
                    interactive=False
                )
                output_image_path_display = gr.Textbox(
                    value="No processed image yet",
                    label="Image Path",
                    interactive=False,
                    lines=2,
                    max_lines=2,
                    show_copy_button=True
                )
        

        gr.Markdown("---")
        gr.Markdown(
            """
            <div style='text-align: center; color: gray; font-size: 0.9em;'>
            Smart Translator | Team Project Advanced Deep Learning
            </div>
            """
        )
        
        
        msg_input.submit(
            add_user_message, 
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            queue=False  
        ).then(
            process_message_streaming,
            inputs=[chatbot, input_image, output_image], 
            outputs=[chatbot, input_image, output_image, input_image_path_display, output_image_path_display]
        )
        
        send_btn.click(
            add_user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            queue=False
        ).then(
            process_message_streaming,
            inputs=[chatbot, input_image, output_image],
            outputs=[chatbot, input_image, output_image, input_image_path_display, output_image_path_display]
        )
        
        file_btn.upload(
            load_image_from_file,
            inputs=[file_btn, chatbot],
            outputs=[chatbot, input_image, output_image, input_image_path_display, output_image_path_display]
        )
        
        webcam_btn.click(
            capture_from_webcam_ui,
            inputs=[chatbot],
            outputs=[chatbot, input_image, output_image, input_image_path_display, output_image_path_display]
        )
    
    return demo



if __name__ == "__main__":
    demo = create_ui()
    
    print("\n" + "="*60)
    print("Starting Smart Translator GUI")
    print("="*60)
    print(f"Tools Server: Running (PID: {server_pid})")
    print(f"Opening browser...")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,  
        show_error=True
    )