def detect_pauses_spectral(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256
) -> List[dict]:
    """
    Use spectral features to detect pauses vs. background noise.
    """
    # Compute spectral features
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    
    # Spectral flatness (high for noise, low for speech)
    spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]
    
    # Spectral centroid (higher for speech)
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    
    # RMS energy
    rms = librosa.feature.rms(S=S)[0]
    
    # Combined pause score
    pause_score = (
        spectral_flatness * 0.4 +  # High flatness = noise/pause
        (1 - rms / rms.max()) * 0.4 +  # Low energy = pause
        (1 - spectral_centroid / spectral_centroid.max()) * 0.2  # Low centroid = pause
    )
    
    # Threshold to find pauses
    pause_threshold = 0.7
    frame_duration = hop_length / sr
    
    pauses = []
    in_pause = False
    pause_start = 0
    
    for i, score in enumerate(pause_score):
        if score > pause_threshold and not in_pause:
            in_pause = True
            pause_start = i
        elif score <= pause_threshold and in_pause:
            in_pause = False
            duration = (i - pause_start) * frame_duration
            if duration > 0.2:  # 200ms minimum
                pauses.append({
                    "start": pause_start * frame_duration,
                    "end": i * frame_duration,
                    "duration": duration,
                    "avg_pause_score": float(np.mean(pause_score[pause_start:i]))
                })
    
    return pauses