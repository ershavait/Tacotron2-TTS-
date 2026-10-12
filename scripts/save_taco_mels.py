import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tacotron2.model import Tacotron2, Tacotron2Config
from src.tacotron2.dataset import TTSDataset, TTSCollator

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save Tacotron2 Mel-spectrograms for HiFi-GAN fine-tuning")

    parser.add_argument("--path_to_manifest", nargs="+", type=str, required=True,
                        help="List of paths to metadata CSV files (e.g. train and test manifests)")
    parser.add_argument("--path_to_save", type=str, required=True,
                        help="Directory to save generated .npy Mel-spectrograms")
    parser.add_argument("--taco_weights", type=str, required=True,
                        help="Path to trained Tacotron2 checkpoint file")

    ### Inference Config ###
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)

    ### DATASET CONFIG ###
    parser.add_argument("--sampling_rate", type=int, default=22050)
    parser.add_argument("--segment_size", type=int, default=8192)
    parser.add_argument("--num_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--min_db", type=float, default=-100.0)
    parser.add_argument("--max_scaled_abs", type=float, default=4.0)

    return parser.parse_args()

def gen_mels(args):
    print(f"Loading Tacotron2 weights from: {args.taco_weights}")
    model = Tacotron2(config=Tacotron2Config())
    state_dict = torch.load(args.taco_weights, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(args.device)

    os.makedirs(args.path_to_save, exist_ok=True)

    for manifest_path in args.path_to_manifest:
        print(f"Processing manifest: {manifest_path}")

        dataset = TTSDataset(manifest_path, 
                             sample_rate=args.sampling_rate, 
                             n_fft=args.n_fft, 
                             window_size=args.window_size, 
                             hop_size=args.hop_size, 
                             fmin=args.fmin, 
                             fmax=args.fmax, 
                             num_mels=args.num_mels, 
                             min_db=args.min_db, 
                             max_scaled_abs=args.max_scaled_abs)

        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=TTSCollator(), num_workers=args.workers)

        for batch in tqdm(loader, desc="Generating Mels"):
            text, input_lengths, mel, stops, encoder_mask, decoder_mask, paths = batch

            save_paths = []
            for file_path in paths:
                base_id = os.path.splitext(os.path.basename(file_path))[0]
                save_paths.append(os.path.join(args.path_to_save, f"{base_id}.npy"))
            
            with torch.no_grad():
                mels_out, mels_postnet_out, stop_preds, alignment = model(
                    text.to(args.device), 
                    input_lengths.to("cpu"), 
                    mel.to(args.device), 
                    encoder_mask.to(args.device), 
                    decoder_mask.to(args.device)
                )

            output_lens = (~decoder_mask).sum(axis=-1)
            
            for pred_mel, valid_len, save_dest in zip(mels_postnet_out, output_lens, save_paths):
                # Convert to shape (num_mels, time)
                mel_np = pred_mel.T.cpu().numpy()
                mel_np = mel_np[:, :valid_len]
                np.save(save_dest, mel_np)

    print(f"All Mel-spectrograms successfully cached to: {args.path_to_save}")

if __name__ == "__main__":
    args = parse_args()
    gen_mels(args)
