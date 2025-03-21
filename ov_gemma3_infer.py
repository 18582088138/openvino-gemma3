from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextStreamer
from PIL import Image
from io import BytesIO
from pathlib import Path
import requests

ov_model_path = "ov-gemma-3-4b-it"
device="CPU"

model = OVModelForVisualCausalLM.from_pretrained(ov_model_path, device=device)
processor = AutoProcessor.from_pretrained(ov_model_path)


def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


# image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image_file = Path("COCO_val2014_000000125211.jpg")
text_message = "What is unusual on this image?"
image = load_image(image_file)


inputs = model.preprocess_inputs(text=text_message, image=image, processor=processor)

print(f"Question:\n{text_message}")
# display(image)
print("Answer:")
model.generate(**inputs, do_sample=False, max_new_tokens=128, streamer=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True))
