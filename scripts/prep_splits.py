import os
import argparse
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_audio_duration(path):
    try:
        return sf.info(path).duration
    except Exception:
        return librosa.get_duration(path=path)

def random_split_and_prep_ljspeech_metadata(path_to_data, path_to_save="data/", seed=42, test_split_pct=0.01, sort=True):
    """
    Process LJSpeech dataset:
    1. Loads metadata.csv (formatted as ID|RawText|NormalizedText)
    2. Builds full file paths to wav files
    3. Calculates audio duration for padding optimization
    4. Splits into train_metadata.csv and test_metadata.csv
    """
    path_to_metadata = os.path.join(path_to_data, "metadata.csv")
    if not os.path.exists(path_to_metadata):
        raise FileNotFoundError(f"Could not find metadata.csv at {path_to_metadata}")
    
    metadata = pd.read_csv(path_to_metadata, sep="|", header=None, quoting=3)
    if metadata.shape[1] >= 3:
        metadata = metadata[[0, 2]]
    else:
        metadata = metadata[[0, 1]]
    metadata.columns = ["file_path", "normalized_transcript"]

    # Remove any NaN or empty normalized transcripts
    metadata = metadata.dropna(subset=["normalized_transcript"]).reset_index(drop=True)
    
    # Construct full audio file path
    full_path_func = lambda x: os.path.join(path_to_data, "wavs", f"{x}.wav")
    metadata["file_path"] = metadata["file_path"].apply(full_path_func)

    # Verify that audio files exist
    exists = metadata["file_path"].apply(os.path.isfile)
    missing = (~exists).sum()
    if missing > 0:
        raise FileNotFoundError(f"{missing} wav files are missing in {os.path.join(path_to_data, 'wavs')}. Is extraction still running?")

    print(f"Verified all {len(metadata)} audio files. Calculating durations...")
    tqdm.pandas(desc="Calculating audio durations")
    metadata["duration"] = metadata["file_path"].progress_apply(get_audio_duration)

    # Split train and test sets
    train_df, test_df = train_test_split(metadata, test_size=test_split_pct, random_state=seed)

    if sort:
        # Sort from longest to shortest to minimize padding in batches
        train_df = train_df.sort_values(by=["duration"], ascending=False)
        test_df = test_df.sort_values(by=["duration"], ascending=False)

    os.makedirs(path_to_save, exist_ok=True)
    train_path = os.path.join(path_to_save, "train_metadata.csv")
    test_path = os.path.join(path_to_save, "test_metadata.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Done! Saved {len(train_df)} train samples to {train_path}")
    print(f"Saved {len(test_df)} test samples to {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/val splits from LJSpeech dataset")
    parser.add_argument("--path_to_ljspeech", "--path_to_ljspech", dest="path_to_ljspeech", type=str, required=True,
                        help="Path to unzipped LJSpeech directory containing metadata.csv and wavs/")
    parser.add_argument("--path_to_save", type=str, default="data",
                        help="Output directory to save train_metadata.csv and test_metadata.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test_split_pct", type=float, default=0.01, help="Fraction for validation split (default 0.01)")

    args = parser.parse_args()
    random_split_and_prep_ljspeech_metadata(
        path_to_data=args.path_to_ljspeech, 
        path_to_save=args.path_to_save,
        seed=args.seed,
        test_split_pct=args.test_split_pct
    )
