from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_captions(scene, objects, mood, colours):
    objects_str = ", ".join(objects) if objects else "nothing specific"
    colours_str = ", ".join(colours)

    prompt = f"""
You are a creative Instagram caption writer.

Analyse this photo context:
- Scene type: {scene}
- Objects detected: {objects_str}
- Color mood: {mood}
- Dominant colours: {colours_str}

Write exactly 3 Instagram captions:
1. FUNNY: witty and humorous
2. AESTHETIC: poetic and dreamy
3. PROFESSIONAL: clean and inspiring

Keep each caption under 150 characters.
Format exactly like this:
FUNNY: [caption]
AESTHETIC: [caption]
PROFESSIONAL: [caption]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    raw = response.choices[0].message.content
    captions = {}
    for line in raw.strip().split('\n'):
        if line.startswith('FUNNY:'):
            captions['funny'] = line.replace('FUNNY:', '').strip()
        elif line.startswith('AESTHETIC:'):
            captions['aesthetic'] = line.replace('AESTHETIC:', '').strip()
        elif line.startswith('PROFESSIONAL:'):
            captions['professional'] = line.replace('PROFESSIONAL:', '').strip()

    return captions

if __name__ == "__main__":
    test = generate_captions(
        scene="lifestyle",
        objects=["potted plant", "vase"],
        mood="neutral and minimal",
        colours=["#3b3429", "#c5c99e", "#838754"]
    )
    for style, caption in test.items():
        print(f"{style.upper()}: {caption}")

        