import gradio as gr
from cv_pipeline import analyse_image, generate_gradcam
from llm_engine import generate_captions

def process_image(image):
    # Save uploaded image temporarily
    image.save("temp_input.jpg")
    
    # Run CV pipeline
    result = analyse_image("temp_input.jpg")
    
    # Generate Grad-CAM
    gradcam_path = generate_gradcam("temp_input.jpg", "temp_gradcam.jpg")
    
    # Generate captions
    captions = generate_captions(
        scene=result['scene'],
        objects=result['objects'],
        mood=result['mood'],
        colours=result['colours']
    )
    
    # Format output
    scene_info = f"Scene: {result['scene']}"
    objects_info = f"Objects: {', '.join(result['objects'])}"
    mood_info = f"Mood: {result['mood']}"
    colours_info = f"Colours: {', '.join(result['colours'])}"
    
    cv_summary = f"{scene_info}\n{objects_info}\n{mood_info}\n{colours_info}"
    
    funny = captions.get('funny', '')
    aesthetic = captions.get('aesthetic', '')
    professional = captions.get('professional', '')
    
    return gradcam_path, cv_summary, funny, aesthetic, professional

with gr.Blocks(title="Visual Caption AI") as demo:
    gr.Markdown("# Visual Caption AI")
    gr.Markdown("Upload a photo — AI analyses it and generates 3 Instagram captions")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload your photo")
            submit_btn = gr.Button("Generate Captions", variant="primary")
        
        with gr.Column():
            gradcam_output = gr.Image(label="What AI looked at (Grad-CAM)")
            cv_output = gr.Textbox(label="CV Analysis", lines=4)
    
    with gr.Row():
        funny_output = gr.Textbox(label="Funny caption", lines=2)
        aesthetic_output = gr.Textbox(label="Aesthetic caption", lines=2)
        professional_output = gr.Textbox(label="Professional caption", lines=2)
    
    submit_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[gradcam_output, cv_output, funny_output, aesthetic_output, professional_output]
    )

demo.launch()