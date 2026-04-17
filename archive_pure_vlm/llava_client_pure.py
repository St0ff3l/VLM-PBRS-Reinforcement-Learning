import base64
import json
import re
from io import BytesIO
from urllib import request
import imageio.v2 as imageio

# P0 Fix: VLM outputs a continuous float (0.0 to 1.0) directly. No if-else.
DEFAULT_PROMPT = (
    "Analyze the image of the reinforcement learning environment. "
    "Evaluate the progress of the agent towards the goal as a float number between 0.0 and 1.0. "
    "0.0 means at the starting point, 1.0 means goal reached. "
    "Only output the float number, strictly no other text."
)

def query_llava_potential_score(
    frame,
    model: str = "llava:7b",
    prompt: str = DEFAULT_PROMPT,
    host: str = "http://localhost:11434",
    timeout: int = 120,
) -> float:
    buffer = BytesIO()
    imageio.imwrite(buffer, frame, format="png")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [image_b64],
        "options": {"temperature": 0.0} # Keep generation stable
    }
    
    try:
        req = request.Request(
            url=f"{host.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        reply_text = data.get("response", "").strip().upper()
        
        # MCQ Parsing: Extract A, B, or C to prevent overlapping words in chatty responses.
        if re.search(r'\bA\b|^A', reply_text):
            return 0.3
        elif re.search(r'\bC\b|^C', reply_text):
            return 1.0
        elif re.search(r'\bB\b|^B', reply_text):
            return 0.0
        
        # Fallback: Rigorous extraction of float if semantics missed
        match = re.search(r'(0\.\d+|1\.0|0|1)', reply_text)
        if match:
            return float(match.group(1))
            
        print(f"[VLM Parse Warning] Unrecognized Output: {reply_text}")
        return 0.0
    except Exception as e:
        print(f"[VLM Request Error] {e}")
        return 0.0
