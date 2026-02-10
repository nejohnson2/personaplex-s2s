import os
import subprocess
import sys

# Allow MPS to fall back to CPU for unsupported ops (e.g. aten::index_copy).
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Required: Hugging Face token must be set so the model weights can be downloaded.
if "HF_TOKEN" not in os.environ:
    raise RuntimeError("Please set HF_TOKEN in your environment before running.")

input_wav = "input.wav"
output_wav = "output.wav"

voice_prompt = "VARM2.pt"  # try: NATF2.pt, NATM1.pt, VARF3.pt, etc.
text_prompt = (
    "You enjoy having a good conversation. "
    "Be playful and a little dramatic, but stay helpful. "
    "If I pause, ask me a fun follow-up question."
    "Spend most of your time laughing and telling jokes."
)

output_json = "output.json"

cmd = [
    sys.executable, "-m", "moshi.offline",
    "--device", "mps",
    "--input-wav", input_wav,
    "--output-wav", output_wav,
    "--output-text", output_json,
    "--voice-prompt", voice_prompt,
    "--text-prompt", text_prompt,
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"Done. Wrote {output_wav} and {output_json}")
