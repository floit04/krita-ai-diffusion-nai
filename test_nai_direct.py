"""Direct NAI API test - bypasses Qt networking to isolate the issue."""

import json
import sys

try:
    import requests
except ImportError:
    print("pip install requests first!")
    sys.exit(1)

# Read token from settings
import os

settings_path = os.path.expanduser(r"~\AppData\Roaming\krita\ai_diffusion\settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)

token = settings.get("nai_api_token", "").strip()
if not token:
    print("No nai_api_token found in settings.json!")
    sys.exit(1)

print(f"Token: {token[:12]}...{token[-4:]}")

# Exact same request body from the log
request_body = {
    "input": "1girl,",
    "model": "nai-diffusion-4-curated-preview",
    "action": "generate",
    "parameters": {
        "width": 832,
        "height": 1216,
        "sampler": "k_euler",
        "steps": 28,
        "scale": 5.0,
        "cfg_rescale": 0.0,
        "noise_schedule": "native",
        "seed": 12345,
        "n_samples": 1,
        "qualityToggle": True,
        "ucPreset": 0,
        "sm": False,
        "sm_dyn": False,
        "dynamic_thresholding": False,
        "negative_prompt": "lowres, bad anatomy",
        "image_format": "png",
        "params_version": 3,
        "v4_prompt": {
            "caption": {"base_caption": "1girl,", "char_captions": []},
            "use_coords": False,
            "use_order": True,
        },
        "v4_negative_prompt": {
            "caption": {"base_caption": "lowres, bad anatomy", "char_captions": []},
            "legacy_uc": False,
        },
    },
}

print(f"\nSending to https://image.novelai.net/ai/generate-image")
print(f"Model: {request_body['model']}")
print(f"Input: {request_body['input']}")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

try:
    resp = requests.post(
        "https://image.novelai.net/ai/generate-image",
        json=request_body,
        headers=headers,
        timeout=60,
    )
    print(f"\nStatus: {resp.status_code}")
    print(f"Headers: {dict(resp.headers)}")
    if resp.status_code == 200:
        print(f"Success! Response size: {len(resp.content)} bytes")
        print(f"Content-Type: {resp.headers.get('content-type')}")
        # Save the ZIP
        with open("nai_test_output.zip", "wb") as f:
            f.write(resp.content)
        print("Saved to nai_test_output.zip")
    else:
        print(f"Error body: {resp.text[:500]}")
except Exception as e:
    print(f"Request failed: {e}")
