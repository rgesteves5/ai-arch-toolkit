"""08 — Multimodal Input (Gemini).

Send an image alongside text using ImagePart. Gemini requires inline
base64 data for images (fileUri only supports gs:// URIs), so we
fetch the image first and encode it.
"""

import base64

import requests

from ai_arch_toolkit import Client, ImagePart, Message, TextPart

IMAGE_URL = "https://picsum.photos/id/237/300/200.jpg"

# Download and base64-encode the image
img_bytes = requests.get(IMAGE_URL, timeout=10).content
image_b64 = base64.b64encode(img_bytes).decode()

client = Client("gemini", model="gemini-2.0-flash")

message = Message(
    role="user",
    content=(
        ImagePart(media_type="image/jpeg", data=image_b64),
        TextPart(text="Describe this image in detail."),
    ),
)

response = client.chat([message])
print("Description:", response.text)
