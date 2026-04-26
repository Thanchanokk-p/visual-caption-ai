# 📸 Visual Caption AI

An end-to-end computer vision pipeline that analyses any photo and generates three Instagram captions — funny, aesthetic, and professional — powered by a multi-stage CV system and GPT-4o-mini.

🔗 **Live demo:** [huggingface.co/spaces/sqffriend/visual-caption-ai](https://huggingface.co/spaces/sqffriend/visual-caption-ai)

---

## What it does

Upload any photo and the app will:
1. Classify the scene type (food, travel, lifestyle, fashion)
2. Detect every object present in the image
3. Extract the 3 dominant colours and map them to a mood
4. Generate a Grad-CAM heatmap showing where the AI looked
5. Use all that context to generate 3 Instagram captions via LLM

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Scene classification | MobileNetV2 (torchvision) | Transfer learning — pre-trained on ImageNet |
| Object detection | YOLOv8 nano (ultralytics) | Detects objects with bounding boxes + confidence |
| Colour extraction | K-Means (scikit-learn) | Classical CV — clusters pixels into dominant colours |
| Explainability | Grad-CAM (pytorch-grad-cam) | Heatmap showing model attention |
| Caption generation | GPT-4o-mini (OpenAI API) | Converts CV context into 3 caption styles |
| UI | Gradio | Browser-based interface with image upload |
| Deployment | Hugging Face Spaces | Free cloud hosting, public HTTPS URL |

---

## CV Concepts Covered

- **Transfer learning** — using MobileNetV2 pre-trained weights, no training from scratch
- **Object detection** — bounding boxes, confidence thresholding, NMS via YOLO
- **Classical CV** — K-Means clustering on pixel colour space without deep learning
- **Explainable AI** — Grad-CAM gradient visualisation on convolutional layers
- **LLM integration** — structured prompt engineering, CV as context not decoration

---

## Project Structure
visual-caption-ai/
```
├── .env                # API keys — local only, never committed
├── .gitignore          # Excludes secrets, model weights, generated images
├── app.py              # Gradio UI — connects all components
├── cv_pipeline.py      # Scene classifier, object detector, colour extractor, Grad-CAM
├── llm_engine.py       # OpenAI API integration and caption generation
├── README.md           # This file
└── requirements.txt    # Dependencies for Hugging Face deployment

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/sqffriend/visual-caption-ai
cd visual-caption-ai
```

**2. Create conda environment**
```bash
conda create -n visual-caption python=3.10 -y
conda activate visual-caption
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your OpenAI API key**

Create a `.env` file in the root folder:
OPENAI_API_KEY=sk-your-key-here

**5. Run locally**
```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## Engineering Notes

**Security** — API key stored in `.env`, loaded via `python-dotenv`, excluded from git via `.gitignore`. On Hugging Face, stored as an encrypted secret in Space settings.

**Environment isolation** — Conda environment keeps all dependencies separate from system Python, ensuring reproducibility.

**Deployment** — Hugging Face Spaces auto-builds a Docker container from `requirements.txt` and serves the app at a public HTTPS URL. No DevOps required.

---

## Roadmap

- [ ] Attention heatmap using SALICON eye-tracking model
- [ ] Face and emotion detection to adapt caption tone
- [ ] Feed consistency scorer for 9-image Instagram grid
- [ ] Platform-specific captions for Instagram, LinkedIn, TikTok
- [ ] User auth and history via Supabase
- [ ] Chrome extension for direct Instagram integration

---

## Built with

Python · PyTorch · Ultralytics · scikit-learn · OpenAI API · Gradio · Hugging Face Spaces

---

*Built as a data science portfolio project demonstrating end-to-end computer vision engineering.*
