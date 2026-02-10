# Voice Processor

Speech-to-speech inference using [PersonaPlex](https://github.com/NVIDIA/personaplex) by NVIDIA, built on the [Moshi](https://github.com/kyutai-labs/moshi) architecture by Kyutai. Feed in an audio clip with a voice and persona prompt, and get a spoken response back.

## Prerequisites

- Python 3.12
- macOS with Apple Silicon (uses MPS for inference)
- [Homebrew](https://brew.sh/)
- A [Hugging Face](https://huggingface.co/) account and access token

## Installation

1. **Install the Opus audio codec:**

   ```bash
   brew install opus
   ```

2. **Clone this repo and the PersonaPlex submodule:**

   ```bash
   git clone git@github.com:NVIDIA/personaplex.git
   ```

3. **Create a virtual environment and install:**

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install personaplex/moshi/
   ```

4. **Set your Hugging Face token:**

   Add this to your `~/.zshrc` (or export it in your current shell):

   ```bash
   export HF_TOKEN=your_token_here
   ```

   Then reload your shell:

   ```bash
   source ~/.zshrc
   ```

## Usage

1. Place your input audio file as `input.wav` in the project root.

2. Run the script:

   ```bash
   python hello-world.py
   ```

3. The model will process the audio and write:
   - `output.wav` — the spoken response
   - `output.json` — the text transcript of the response

## Customization

In `hello-world.py` you can change:

- **`voice_prompt`** — the voice style (e.g. `NATF2.pt`, `NATM1.pt`, `VARF3.pt`, `VARM2.pt`)
- **`text_prompt`** — the persona/behavior instructions for the model

## Notes

- The first run downloads model weights from Hugging Face, which may take a while.
- Input audio should be clear speech with minimal background noise.
- Short clips (a few seconds to ~30s) work best.
- MPS fallback to CPU is enabled automatically for unsupported PyTorch ops.
