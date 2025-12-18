whisperx \
  "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/generated/extract_parquet_data/audio/00001.wav"
  --model large-v3 \
  --compute_type int8 \
  --vad_method pyannote \
  --output_format all \
  --output_dir ./whisperx_output \
  --diarize \
  --print_progress
