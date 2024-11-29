from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import requests
import random  # For random selection
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load Hugging Face FLAN-T5 pipeline
try:
    text_generator = pipeline("text2text-generation", model="./flan-t5-large", tokenizer="./flan-t5-large")
except Exception as e:
    print(f"Error loading FLAN-T5 pipeline: {e}")
    exit(1)

# Dog API setup for fetching breed images
DOG_API_URL = "https://dog.ceo/api/breed/{breed}/images/random"

# Breed mapping for keywords
BREEDS = {
    "affenpinscher": ["tiny", "playful"],
    "african": ["wild", "rare"],
    "airedale": ["strong", "versatile"],
    "akita": ["loyal", "dignified"],
    "appenzeller": ["hardworking", "energetic"],
    "australian shepherd": ["herding", "energetic"],
    "beagle": ["small", "playful", "curious"],
    "bluetick": ["hunting", "noisy"],
    "borzoi": ["elegant", "fast"],
    "boxer": ["muscular", "energetic", "playful"],
    "bulldog": ["sturdy", "calm", "wrinkled"],
    "bullterrier": ["tough", "fearless"],
    "cattledog": ["australian", "herding"],
    "chihuahua": ["tiny", "bold", "alert"],
    "cockapoo": ["friendly", "curly-haired"],
    "collie": ["herding", "loyal"],
    "coonhound": ["hunting", "loyal"],
    "corgi": ["short-legged", "herding", "affectionate"],
    "dachshund": ["long-bodied", "playful", "small"],
    "dalmatian": ["spotted", "active", "firefighter"],
    "dane": ["giant", "gentle", "elegant"],
    "elkhound": ["norwegian", "hunting"],
    "finnish lapphund": ["northern", "herding"],
    "germanshepherd": ["protective", "loyal", "working"],
    "golden retriever": ["friendly", "family", "gentle"],
    "greyhound": ["fast", "sleek"],
    "husky": ["energetic", "sled", "wolf-like"],
    "labrador": ["loyal", "outdoor", "energetic"],
    "malamute": ["sled", "strong", "wolf-like"],
    "mastiff": ["giant", "guard", "loyal"],
    "papillon": ["tiny", "cheerful"],
    "poodle": ["intelligent", "curly-haired", "elegant"],
    "pug": ["wrinkled", "charming", "compact"],
    "rottweiler": ["strong", "guard", "loyal"],
    "samoyed": ["fluffy", "friendly"],
    "shiba": ["fox-like", "independent", "alert"],
    "spaniel": ["friendly", "sporting"],
    "terrier": ["energetic", "fearless"],
    "vizsla": ["hunting", "affectionate"],
    "whippet": ["fast", "graceful"],
    "yorkshire terrier": ["small", "long-haired", "feisty"]
}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid request format. Expected a JSON object."}), 400

    if "prompt" not in data or not isinstance(data["prompt"], str) or not data["prompt"].strip():
        return jsonify({"error": "Invalid 'prompt' field. It must be a non-empty string."}), 400

    if "language" not in data or data["language"].lower() not in ["english", "dutch"]:
        return jsonify({"error": "Invalid 'language' field. It must be 'english' or 'dutch'."}), 400

    prompt = data["prompt"].strip()
    language = data["language"].lower()

    # Generate a description using the FLAN-T5 pipeline
    try:
        response = text_generator(
            f"Generate a detailed, creative description of a dog based on the following preferences: {prompt}",
            max_length=150,
            num_return_sequences=1,
            temperature=0.7
        )
        ai_description = response[0]["generated_text"]
    except Exception as e:
        print(f"Error generating text with FLAN-T5 pipeline: {e}")
        ai_description = "We couldn't generate a description at this time."

    # Select a breed and fetch an image
    try:
        matched_breeds = []
        for keyword, breeds in BREEDS.items():
            if keyword in prompt.lower():
                matched_breeds.extend(breeds)

        # Randomly choose a breed if matches exist, else fallback to any breed
        breed = random.choice(matched_breeds) if matched_breeds else random.choice(list(BREEDS.keys()))
        breed_api = breed.replace(" ", "/").lower()

        # Fetch image from Dog API
        response = requests.get(DOG_API_URL.format(breed=breed_api))
        response.raise_for_status()
        image_url = response.json().get("message", "https://via.placeholder.com/150")
    except Exception as e:
        print(f"Error fetching breed image: {e}")
        image_url = "https://via.placeholder.com/150"

    return jsonify({
        "generated_text": ai_description,
        "image_url": image_url
    })

if __name__ == '__main__':
    app.run(debug=True)
