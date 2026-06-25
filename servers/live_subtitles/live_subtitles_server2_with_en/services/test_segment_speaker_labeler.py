from audio_utils import resolve_audio_paths, resolve_audio_paths_as_tensor_list
from segment_speaker_labeler import (
    SegmentSpeakerLabeler,
    DEFAULT_THRESHOLD_SAME,
    DEFAULT_THRESHOLD_POSSIBLE,
    DEFAULT_THRESHOLD_NEW_SPEAKER,
)
from embedding_model_factory import (
    EmbeddingModelType,
    create_embedding_model,
    list_available_models,
)
from audio_tagger import AudioTagger

audio_paths = [
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\last_50_segments\segment_002\sound.wav",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\last_50_segments\segment_004\sound.wav"
]

MODEL_TYPE = EmbeddingModelType.MODELSCOPE_ERES2NETV2

embedding_model = create_embedding_model(MODEL_TYPE)
audio_tagger = AudioTagger()

labeler = SegmentSpeakerLabeler(
    embedding_model=embedding_model,
    audio_tagger=audio_tagger,
    debug=True,
)

# Process a long meeting recording
result = labeler.label_long_audio(
    audio=audio_paths,
    min_duration=2.0,           # Only keep segments >= 2 seconds
)

# Results:
# - Long audio broken into clean segments
# - Multi-speaker segments separated
# - Each segment labeled with speaker identity
for seg in result["segments"]:
    print(f"{seg['start_time']:.1f}s-{seg['end_time']:.1f}s: "
          f"{seg['primary_speaker']} (confidence: {seg['primary_confidence']:.3f})")

print(f"Detected speakers: {result['summary']['speaker_labels']}")
print(f"Confidence breakdown: {result['summary']['confidence_summary']}")
