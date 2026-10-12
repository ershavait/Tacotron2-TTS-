import os
import sys
import argparse
import torch
import soundfile as sf
import matplotlib.pyplot as plt

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tacotron2.model import Tacotron2, Tacotron2Config
from src.common.tokenizer import Tokenizer
from src.common.audio import denormalize
import librosa

# Standard test sentences: short, medium, medium-long, and classic long LJSpeech sentence
FIXED_TEST_SENTENCES = [
    ("short", "Hello, this is a test."),
    ("medium", "Generative neural text to speech models learn speech synthesis."),
    ("long", "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition.")
]

def evaluate_checkpoint(checkpoint_dir, output_dir="eval_milestones"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    
    config = Tacotron2Config(num_chars=tokenizer.vocab_size)
    model = Tacotron2(config).to(device)

    # Find model weights in checkpoint directory
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            # Check if any .safetensors or .bin in folder
            files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors") or f.endswith(".bin")]
            if not files:
                raise FileNotFoundError(f"No weights found in {checkpoint_dir}")
            weights_path = os.path.join(checkpoint_dir, files[0])

    print(f"Loading weights from: {weights_path}")
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location=device)

    # Strip unexpected accelerator prefixes if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "").replace("module.", "")
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict)
    model.eval()

    chk_name = os.path.basename(os.path.normpath(checkpoint_dir))
    save_dir = os.path.join(output_dir, chk_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n--- Evaluating Checkpoint: {chk_name} ---")

    for label, text in FIXED_TEST_SENTENCES:
        tokens = tokenizer.encode(text)
        text_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.inference_mode():
            mel_postnet_out, attn_weights = model.inference(text_tensor, max_decode_steps=1200)

        # Plot Free Inference Attention Alignment
        attn = attn_weights[0].T.cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
        ax.set_title(f"Inference Alignment ({label}): {text[:40]}...")
        ax.set_ylabel("Character Index")
        ax.set_xlabel("Decoder Mel Timesteps")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"alignment_{label}.png")
        plt.savefig(plot_path)
        plt.close()

        # Reconstruct Audio via Griffin-Lim baseline for quick audibility check
        mel_denorm = denormalize(mel_postnet_out[0].T.cpu()).numpy()
        # Griffin-lim invert
        S = librosa.db_to_amplitude(mel_denorm)
        linear_spec = librosa.feature.inverse.mel_to_stft(S, sr=22050, n_fft=1024)
        wav = librosa.griffinlim(linear_spec, n_iter=32, hop_length=256, win_length=1024)

        wav_path = os.path.join(save_dir, f"audio_{label}.wav")
        sf.write(wav_path, wav, 22050)
        print(f"[{label.upper()}] Saved Alignment: {plot_path} | Audio: {wav_path}")

    print(f"\nCheck {save_dir} to inspect the diagonal and listen to the audio files!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Tacotron 2 Checkpoint on Fixed Test Sentences")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint folder (e.g. work_dir/tacotron2/checkpoint_0)")
    parser.add_argument("--output_dir", type=str, default="eval_milestones", help="Where to save alignment plots and audio wavs")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint_dir, args.output_dir)
