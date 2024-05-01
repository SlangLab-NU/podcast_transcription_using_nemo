import os
import argparse
import json
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.transcribe_utils import transcribe_partial_audio
# Source code from github:
# https://github.com/NVIDIA/NeMo/blob/43ccc1d6bd82ec788d970f90c3ed7192882651b3/nemo/collections/asr/parts/utils/transcribe_utils.py#L455


def transcribe_audio(manifest_file, asr_model_path, output_dir):
    # Load ASR model
    # asr_model_subword = os.path.join(os.getcwd(), asr_model_path)
    asr_model_subword = nemo_asr.models.ASRModel.restore_from(
        restore_path=asr_model_path)

    with open(manifest_file, "r") as f:
        manifest_data = json.load(f)

    # # Transcribe each audio file in the manifest
    # for item in manifest_data:
    #     audio_file_path = item["audio_filepath"]

    #     input_file_name = os.path.basename(audio_file_path)

    #     output_file_name = os.path.splitext(input_file_name)[0] + ".json"
    #     output_file_path = os.path.join(output_dir, output_file_name)

    #     # Transcribe audio using NeMo ASR
    #     tr = transcribe_partial_audio(
    #         asr_model=asr_model_subword, path2manifest=manifest_file, return_hypotheses=True)

    #     # Save transcription to a JSON file
    #     with open(output_file_path, "w") as outfile:
    #         json.dump(tr, outfile)

    # Transcribe audio using NeMo ASR
    tr = transcribe_partial_audio(
        asr_model=asr_model_subword, path2manifest=manifest_file, return_hypotheses=True)

    print("Transcription: ", tr[0].text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio using NeMo ASR")
    parser.add_argument("manifest_file", help="Path to the JSON manifest file")
    parser.add_argument("asr_model_path", help="Path to the ASR model")
    parser.add_argument(
        "output_dir", help="Directory to save the output JSON files")
    args = parser.parse_args()

    print("Manifest file:", args.manifest_file)
    # Check if file exists:
    if not os.path.exists(args.manifest_file):
        raise FileNotFoundError("Manifest file not found")

    print("ASR model path:", args.asr_model_path)
    # Check if file exists:
    if not os.path.exists(args.asr_model_path):
        raise FileNotFoundError("ASR model not found")

    print("Output directory:", args.output_dir)
    # Check if directory exists:
    if not os.path.exists(args.output_dir):
        raise FileNotFoundError("Output directory not found")

    transcribe_audio(args.manifest_file, args.asr_model_path, args.output_dir)
