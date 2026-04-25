import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from sklearn.cluster import KMeans
import numpy as np

# ── Scene Classifier ──────────────────────────────
model = None
transform = None

def init_model():
    global model, transform
    if model is None:
        model = models.mobilenet_v2(pretrained=True)
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

SCENE_KEYWORDS = {
    "food": ["pizza","burger","coffee","restaurant","dining","meal","dish","bread","cake","sushi"],
    "travel": ["beach","mountain","city","street","landmark","temple","forest","ocean","sky","sunset"],
    "lifestyle": ["gym","office","desk","laptop","book","sofa","bedroom","workout","yoga"],
    "fashion": ["dress","suit","jacket","shoes","bag","outfit","style","clothing"]
}

def classify_scene(image_path):
    init_model()
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    _, pred = torch.max(output, 1)
    class_id = pred.item()
    if class_id < 300:
        return "food"
    elif class_id < 500:
        return "travel"
    elif class_id < 800:
        return "lifestyle"
    else:
        return "fashion"

# ── Object Detector ───────────────────────────────
yolo = None

def init_yolo():
    global yolo
    if yolo is None:
        yolo = YOLO('yolov8n.pt')

def detect_objects(image_path):
    init_yolo()
    results = yolo(image_path, verbose=False)
    objects = []
    for r in results:
        for box in r.boxes:
            label = yolo.names[int(box.cls)]
            confidence = float(box.conf)
            if confidence > 0.4:
                objects.append(label)
    return list(set(objects))

# ── Colour + Mood ─────────────────────────────────
def extract_colours(image_path, k=3):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    colours = kmeans.cluster_centers_.astype(int)
    hex_colours = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colours]

    avg = colours.mean(axis=0)
    r, g, b = avg
    if r > 150 and g > 100:
        mood = "warm and golden"
    elif b > 130:
        mood = "cool and calm"
    elif r > 180:
        mood = "bold and vibrant"
    else:
        mood = "neutral and minimal"

    return hex_colours, mood

# ── Run All ───────────────────────────────────────
def analyse_image(image_path):
    scene = classify_scene(image_path)
    objects = detect_objects(image_path)
    colours, mood = extract_colours(image_path)
    return {
        "scene": scene,
        "objects": objects,
        "colours": colours,
        "mood": mood
    }

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

def generate_gradcam(image_path, output_path="gradcam_output.jpg"):
    init_model()
    # Load and resize image to match MobileNet input size
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # normalize pixel values to 0-1
    tensor = transform(img_resized).unsqueeze(0)

    # Target the last convolutional layer of MobileNet
    target_layer = model.features[-1]

    # Generate heatmap using GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor)[0]

    # Overlay heatmap on original image
    visualization = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    # Save output image
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM saved to {output_path}")
    return output_path

if __name__ == "__main__":
    result = analyse_image("test.jpg")
    print(result)
    generate_gradcam("test.jpg")
    