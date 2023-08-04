import requests
import base64
from PIL import Image
from io import BytesIO
import os
import supervision as sv

INFERENCE_ENDPOINT = "https://infer.roboflow.com"
API_KEY = "<Key>"
VIDEO = "data/video1.mp4"

prompts = [
    "construction site",
    "school"
]

ACTIVE_PROMPT = "construction site"


def classify_image(image: str) -> dict:
    image_data = Image.fromarray(image)
    buffer = BytesIO()
    image_data.save(buffer, format="JPEG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "api_key": API_KEY,
        "subject": {
            "type": "base64",
            "value": image_data
        },
        "prompt": prompts,
    }

    data = requests.post(INFERENCE_ENDPOINT + "/clip/compare?api_key=" + API_KEY, json=payload)

    response = data.json()

    highest_prediction = 0
    highest_prediction_index = 0

    for i, prediction in enumerate(response["similarity"]):
        if prediction > highest_prediction:
            highest_prediction = prediction
            highest_prediction_index = i

    return prompts[highest_prediction_index]


results = []

for i, frame in enumerate(sv.get_video_frames_generator(source_path=VIDEO, stride=10)):
    print("Frame", i)
    label = classify_image(frame)
    results.append(label)
    if i == 150:
        break

video_length = 10 * len(results)
video_length = video_length / 24
print(f"Does this video contain a {ACTIVE_PROMPT}?", "yes" if ACTIVE_PROMPT in results else "no")

if ACTIVE_PROMPT in results:
    print(f"When does the {ACTIVE_PROMPT} first appear?", round(results.index(ACTIVE_PROMPT) * 10 / 24, 0), "seconds")

print(f"For how long is the {ACTIVE_PROMPT} visible?", round(results.count(ACTIVE_PROMPT) * 10 / 24, 0), "seconds")