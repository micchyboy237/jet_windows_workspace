"""
generic_speaker_labeling.py
Fully generic speaker labeling system - starts with generic labels in messages
Progressively improves through unsupervised and zero-shot methods
Enhanced with rich logging, terminal file links, and organized outputs
"""
import numpy as np
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import re
import os
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import librosa
import openai


def setup_logger(output_dir):
    """Setup rich logging with file and console output"""
    log_file = Path(output_dir) / "speaker_labeling.log"
    
    # Create logger
    logger = logging.getLogger('SpeakerLabeler')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler - detailed debug logging
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(file_formatter)
    
    # Console handler - info and above for clean output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(console_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class LabelQuality(Enum):
    """Progressive label quality levels"""
    GENERIC = 1
    DESCRIPTIVE = 2
    RELATIONAL = 3
    PROPER_NAME = 4


@dataclass
class SpeakerIdentity:
    """Dynamic speaker identity that evolves through stages"""
    speaker_id: str
    current_label: str
    quality: LabelQuality = LabelQuality.GENERIC
    confidence: float = 0.0
    acoustic_signature: Optional[np.ndarray] = None
    linguistic_profile: Dict[str, float] = field(default_factory=dict)
    topic_distribution: Dict[int, float] = field(default_factory=dict)
    name_candidates: Dict[str, float] = field(default_factory=dict)
    avg_turn_length: float = 0.0
    question_ratio: float = 0.0
    interaction_patterns: Dict[str, int] = field(default_factory=dict)
    label_history: List[Dict] = field(default_factory=list)
    
    def upgrade_label(self, new_label: str, quality: LabelQuality,
                     confidence: float, evidence: Dict = None):
        """Record label progression with evidence"""
        self.label_history.append({
            'timestamp': datetime.now().isoformat(),
            'from_label': self.current_label,
            'to_label': new_label,
            'from_quality': self.quality.name,
            'to_quality': quality.name,
            'confidence': confidence,
            'evidence': evidence or {}
        })
        self.current_label = new_label
        self.quality = quality
        self.confidence = confidence
    
    def to_dict(self) -> Dict:
        """Export identity to dictionary"""
        return {
            'speaker_id': self.speaker_id,
            'current_label': self.current_label,
            'quality': self.quality.name,
            'confidence': self.confidence,
            'label_progression': self.label_history,
            'name_candidates': self.name_candidates,
            'behavior_patterns': {
                'avg_turn_length': self.avg_turn_length,
                'question_ratio': self.question_ratio,
                'linguistic_profile': self.linguistic_profile,
                'topic_distribution': {str(k): v for k, v in self.topic_distribution.items()}
            }
        }


class GenericAudioProcessor:
    """
    Process audio features without gender/emotion/speaker assumptions.
    All characteristics are discovered from the raw signal.
    """
    def __init__(self):
        self.logger = logging.getLogger('SpeakerLabeler')
    
    def extract_features(self, audio_path: str) -> Dict[str, float]:
        """Extract comprehensive audio features"""
        features = {
            'duration': 0.0,
            'rms_energy': 0.0,
            'zero_crossing_rate': 0.0,
            'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0,
            'spectral_rolloff': 0.0,
            'silence_ratio': 0.0
        }
        
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=10.0)
            if len(y) == 0:
                return features
            
            # Basic features
            features['duration'] = len(y) / sr
            features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
            features['zero_crossing_rate'] = float(
                np.sum(np.abs(np.diff(np.signbit(y)))) / len(y)
            )
            features['silence_ratio'] = float(np.sum(np.abs(y) < 0.02) / len(y))
            
            # Spectral features
            features['spectral_centroid'] = float(
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            )
            features['spectral_bandwidth'] = float(
                np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            )
            features['spectral_rolloff'] = float(
                np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            )
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            self.logger.debug(f"    Extracted {len(features)} features from {Path(audio_path).name}")
            return features
            
        except Exception as e:
            self.logger.warning(f"    Could not process {Path(audio_path).name}: {e}")
            return features
    
    def _load_wav_basic(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Basic WAV loading without librosa"""
        import wave
        with wave.open(audio_path, 'rb') as wav:
            sr = wav.getframerate()
            n_frames = min(wav.getnframes(), sr * 10)
            signal = np.frombuffer(wav.readframes(n_frames), dtype=np.int16)
            signal = signal / 32768.0
        return signal, sr
    
    def compute_embedding(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features to embedding vector for clustering"""
        feature_keys = sorted([k for k in features.keys()
                              if isinstance(features[k], (int, float))])
        return np.array([features[k] for k in feature_keys])
    
    def cluster_speakers(self, embeddings: List[np.ndarray]) -> List[int]:
        """
        Automatically discover number of speakers through clustering.
        No assumption about how many speakers exist.
        """
        if len(embeddings) < 2:
            return [0] * len(embeddings)
        
        embeddings_array = np.array(embeddings)
        max_clusters = min(len(embeddings), 10)
        best_labels = None
        best_score = -1
        
        for n_clusters in range(1, max_clusters + 1):
            if n_clusters == 1:
                labels = np.zeros(len(embeddings), dtype=int)
                score = 0.5
            else:
                try:
                    clustering = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity='nearest_neighbors',
                        n_neighbors=min(5, len(embeddings) - 1),
                        random_state=42,
                        assign_labels='kmeans'
                    )
                    labels = clustering.fit_predict(embeddings_array)
                    if len(set(labels)) > 1:
                        score = silhouette_score(embeddings_array, labels)
                    else:
                        score = 0.5
                except Exception:
                    continue
            
            complexity_penalty = n_clusters / max_clusters * 0.2
            adjusted_score = score - complexity_penalty
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_labels = labels
        
        return best_labels.tolist() if best_labels is not None else [0] * len(embeddings)
    
    def _simple_clustering(self, embeddings: np.ndarray) -> List[int]:
        """Simple distance-based clustering fallback"""
        n = len(embeddings)
        if n <= 2:
            return [0] * n
        
        values = embeddings[:, 0] if embeddings.shape[1] > 0 else np.arange(n)
        median = np.median(values)
        return [0 if v < median else 1 for v in values]


class UnsupervisedPatternDiscoverer:
    """
    Discover topics, patterns, and relationships from text without predefined categories.
    Uses NMF topic modeling and statistical pattern analysis.
    """
    def __init__(self, n_topics: int = 3):
        self.n_topics = n_topics
        self.vectorizer = None
        self.topic_model = None
        self.topic_keywords = {}
        self.feature_names = []
        self.logger = logging.getLogger('SpeakerLabeler')
    
    def discover_topics(self, all_texts: List[str]) -> Dict[int, List[str]]:
        """Discover topics automatically from text corpus"""
        if len(all_texts) < 3:
            return {}
        
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        
        try:
            X = self.vectorizer.fit_transform(all_texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            actual_n_topics = min(self.n_topics, X.shape[0] - 1, X.shape[1] - 1)
            if actual_n_topics < 2:
                return {}
            
            self.topic_model = NMF(
                n_components=actual_n_topics,
                random_state=42,
                max_iter=200
            )
            self.topic_model.fit(X)
            
            for topic_idx in range(actual_n_topics):
                top_indices = np.argsort(
                    self.topic_model.components_[topic_idx]
                )[-8:]
                self.topic_keywords[topic_idx] = [
                    self.feature_names[i] for i in top_indices
                ]
            
            self.logger.debug(f"    Discovered {actual_n_topics} topics from {len(all_texts)} texts")
            return self.topic_keywords
            
        except Exception as e:
            self.logger.debug(f"    Topic discovery skipped: {e}")
            return {}
    
    def get_speaker_topic_distribution(self, speaker_texts: List[str]) -> Dict[int, float]:
        """Get topic distribution for a specific speaker"""
        if self.topic_model is None or not speaker_texts:
            return {}
        
        try:
            X = self.vectorizer.transform(speaker_texts)
            topic_dist = self.topic_model.transform(X)
            return {
                i: float(np.mean(topic_dist[:, i]))
                for i in range(topic_dist.shape[1])
            }
        except Exception:
            return {}
    
    def discover_speech_patterns(self, texts: List[str]) -> Dict[str, float]:
        """Discover statistical patterns in how someone speaks"""
        if not texts:
            return {}
        
        patterns = {}
        
        # Word count patterns
        word_counts = [len(t.split()) for t in texts]
        patterns['avg_words_per_turn'] = float(np.mean(word_counts)) if word_counts else 0
        patterns['std_words_per_turn'] = float(np.std(word_counts)) if word_counts else 0
        
        # Question patterns
        patterns['question_ratio'] = float(
            sum(1 for t in texts if '?' in t) / len(texts)
        )
        
        # Politeness markers
        polite_words = ['please', 'thank', 'could', 'would', 'may', 'appreciate']
        politeness_count = sum(
            1 for t in texts
            for w in polite_words
            if w in t.lower()
        )
        patterns['politeness_ratio'] = float(politeness_count / len(texts))
        
        # Urgency markers
        urgency_markers = ['!', 'urgent', 'asap', 'immediately', 'right now']
        urgency_count = sum(
            1 for t in texts
            for m in urgency_markers
            if m in t.lower()
        )
        patterns['urgency_ratio'] = float(urgency_count / len(texts))
        
        # Personal pronoun usage
        patterns['first_person_ratio'] = float(
            sum(1 for t in texts if re.search(r'\b(i|me|my|mine)\b', t.lower())) / len(texts)
        )
        patterns['second_person_ratio'] = float(
            sum(1 for t in texts if re.search(r'\b(you|your|yours)\b', t.lower())) / len(texts)
        )
        
        return patterns


class ZeroShotIdentityDetector:
    """
    Use LLM for zero-shot speaker identity detection.
    No training data, no predefined roles - pure semantic understanding.
    """
    def __init__(self):
        self.logger = logging.getLogger('SpeakerLabeler')
    
    def analyze_conversation(self, messages: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze entire conversation to understand speaker identities.
        Returns discovered roles, relationships, and name candidates.
        """
        transcript = self._build_transcript(messages)
        
        prompt = f"""
Analyze this conversation transcript. Identify who each speaker is based on what they say and how they interact.
For each speaker, determine:
1. What they appear to be doing in this conversation (their functional role)
2. How they relate to other speakers (power dynamics, relationship type)
3. Any names, titles, or identifiers that likely belong to them
4. The likely context/setting of this conversation
Be specific and observant. Don't use generic categories like "speaker_0" - describe what you actually observe.

Transcript:
{transcript}

Return a JSON object where each key is the EXACT speaker ID from the transcript (e.g., "speaker_0", "speaker_1").
Each value should have:
- "observed_role": What this person does in the conversation
- "relationship_to_others": How they relate to other speakers  
- "name_candidates": Array of possible names/titles for this speaker
- "conversation_context": The likely setting (discovered, not assumed)
- "confidence": Your confidence (0.0-1.0)

Output ONLY valid JSON, no other text. Example format:
{{"speaker_0": {{"observed_role": "...", ...}}, "speaker_1": {{...}}}}
"""
        
        try:
            client = openai.OpenAI(
                base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:8080/v1"),
                api_key="sk-1234",
            )
            
            stream: openai.Stream[openai.types.chat.ChatCompletionChunk] = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="Qwen/Qwen3.5-2B",
                max_tokens=2000,
                temperature=0.3,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    },
                },
                stream=True,
            )
            
            content = ""
            for part in stream:
                if part.choices and part.choices[0].delta:
                    delta = part.choices[0].delta
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        print(delta.reasoning_content, flush=True, end="")
                    elif hasattr(delta, "content") and delta.content:
                        content += delta.content
                        print(delta.content, flush=True, end="")
            
            # Clean up response
            if content.startswith('```'):
                content = re.sub(r'```\w*\n?', '', content)
                content = content.replace('```', '')
            
            result = json.loads(content)
            self.logger.debug(f"    LLM analysis completed for {len(result)} speakers")
            return result
            
        except Exception as e:
            self.logger.warning(f"    LLM analysis failed: {e}, using local analysis")
            return self._local_analysis(messages)
    
    def _build_transcript(self, messages: List[Dict]) -> str:
        """Build formatted transcript with generic speaker labels"""
        lines = []
        for i, msg in enumerate(messages):
            speaker = msg.get('speaker', f'speaker_{i}')
            text = msg.get('text', '')
            lines.append(f"[{speaker}]: {text}")
        return '\n'.join(lines)
    
    def _local_analysis(self, messages: List[Dict]) -> Dict[str, Dict]:
        """
        Local analysis without LLM API.
        Uses statistical patterns to discover speaker identities.
        """
        speaker_texts = defaultdict(list)
        speaker_turns = defaultdict(list)
        
        for i, msg in enumerate(messages):
            speaker = msg.get('speaker', f'speaker_{i}')
            speaker_texts[speaker].append(msg.get('text', ''))
            speaker_turns[speaker].append(i)
        
        # Analyze turn-taking patterns
        turn_order = [msg.get('speaker', f'speaker_{i}')
                     for i, msg in enumerate(messages)]
        transitions = Counter()
        for i in range(len(turn_order) - 1):
            if turn_order[i] != turn_order[i + 1]:
                transitions[(turn_order[i], turn_order[i + 1])] += 1
        
        results = {}
        for speaker, texts in speaker_texts.items():
            combined = ' '.join(texts)
            
            # Calculate metrics
            question_ratio = sum(1 for t in texts if '?' in t) / len(texts)
            avg_length = np.mean([len(t.split()) for t in texts])
            
            # Extract name candidates
            name_matches = re.findall(
                r'(?:my name(?:\'s| is)|I(?:\'m| am)|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                combined, re.IGNORECASE
            )
            
            # Find names others use to address this speaker
            others_addressing = []
            for other_speaker, other_texts in speaker_texts.items():
                if other_speaker != speaker:
                    for text in other_texts:
                        name_match = re.match(r'^([A-Z][a-z]+)[,!]', text.strip())
                        if name_match and name_match.group(1) not in ['I', 'You', 'We', 'It']:
                            others_addressing.append(name_match.group(1))
            
            # Determine role based on patterns
            speaks_first = speaker_turns[speaker][0] == 0 if speaker_turns[speaker] else False
            speaks_last = speaker_turns[speaker][-1] == len(messages) - 1 if speaker_turns[speaker] else False
            
            if speaks_first and question_ratio > 0.3:
                observed_role = "initiates_conversation_with_questions"
            elif speaks_first and question_ratio < 0.2:
                observed_role = "opens_conversation_with_statements"
            elif question_ratio > 0.4:
                observed_role = "primarily_asks_questions"
            elif question_ratio < 0.1 and avg_length > 15:
                observed_role = "provides_detailed_information"
            elif 'help' in combined.lower() or 'support' in combined.lower():
                observed_role = "seeks_or_provides_assistance"
            else:
                observed_role = "active_conversation_participant"
            
            # Determine relationship dynamics
            outgoing = sum(1 for (s1, s2) in transitions if s1 == speaker)
            incoming = sum(1 for (s1, s2) in transitions if s2 == speaker)
            
            if incoming > outgoing:
                relationship = "frequently_responded_to"
            elif outgoing > incoming:
                relationship = "frequently_initiates_exchanges"
            else:
                relationship = "balanced_interaction"
            
            results[speaker] = {
                'observed_role': observed_role,
                'relationship_to_others': relationship,
                'name_candidates': list(set(name_matches + others_addressing)),
                'conversation_context': self._discover_context(all_texts=speaker_texts),
                'confidence': 0.6 + (0.1 * min(len(texts), 4))
            }
        
        return results
    
    def _discover_context(self, all_texts: Dict[str, List[str]]) -> str:
        """Discover conversation context from word distributions"""
        all_words = []
        for texts in all_texts.values():
            for text in texts:
                all_words.extend(text.lower().split())
        
        word_freq = Counter(all_words)
        
        context_signals = {
            'customer_service': ['help', 'issue', 'problem', 'order', 'account'],
            'medical': ['symptom', 'pain', 'doctor', 'medicine', 'feel'],
            'technical': ['error', 'code', 'system', 'data', 'server'],
            'personal': ['feel', 'think', 'love', 'family', 'friend'],
            'educational': ['learn', 'study', 'class', 'teacher', 'exam']
        }
        
        scores = {}
        for context, signals in context_signals.items():
            scores[context] = sum(word_freq.get(s, 0) for s in signals)
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "general_conversation"


class GenericSpeakerLabeler:
    """
    Main labeling engine.
    Starts with generic labels (speaker_0, speaker_1) already in messages.
    Progressively improves using unsupervised methods and zero-shot LLM.
    Enhanced with rich logging and organized output capabilities.
    """
    def __init__(self, log_dir=None):
        """Initialize with optional logging directory"""
        self.audio_processor = GenericAudioProcessor()
        self.pattern_discoverer = UnsupervisedPatternDiscoverer()
        self.identity_detector = ZeroShotIdentityDetector()
        self.speaker_identities: Dict[str, SpeakerIdentity] = {}
        
        # Setup logging
        if log_dir:
            self.logger = setup_logger(log_dir)
        else:
            self.logger = logging.getLogger('SpeakerLabeler')
            if not self.logger.handlers:
                # Basic console logging if no directory specified
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # Store audio features per speaker for visualization
        self._stage1_audio_features = {}
    
    def process_conversation(self,
                            messages: List[Dict],
                            audio_files: Optional[List[str]] = None
                            ) -> Dict[str, SpeakerIdentity]:
        """
        Process a conversation with messages already containing generic labels.
        
        Args:
            messages: List of messages, each with 'speaker' key (e.g., 'speaker_0')
            audio_files: Optional list of audio file paths
        
        Returns:
            Dictionary mapping speaker IDs to their SpeakerIdentity objects
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING GENERIC SPEAKER LABELING")
        self.logger.info("=" * 60)
        
        # Count unique speakers
        unique_speakers = set()
        for msg in messages:
            speaker = msg.get('speaker', 'unknown')
            unique_speakers.add(speaker)
        
        self.logger.info(f"\n📋 Input: {len(messages)} messages from {len(unique_speakers)} speakers")
        self.logger.info(f"   Initial labels: {sorted(unique_speakers)}")
        
        # Initialize identities
        self.speaker_identities = {}
        for speaker in sorted(unique_speakers):
            self.speaker_identities[speaker] = SpeakerIdentity(
                speaker_id=speaker,
                current_label=speaker,
                quality=LabelQuality.GENERIC,
                confidence=0.5
            )
            self.logger.debug(f"   Initialized identity for {speaker}")
        
        # Stage 1: Unsupervised Pattern Discovery
        self.logger.info("\n" + "=" * 40)
        self.logger.info("STAGE 1: Unsupervised Pattern Discovery")
        self.logger.info("=" * 40)
        self._stage1_descriptive_labels(messages, audio_files)
        
        # Stage 2: Zero-Shot Identity Detection
        self.logger.info("\n" + "=" * 40)
        self.logger.info("STAGE 2: Zero-Shot Identity Detection")
        self.logger.info("=" * 40)
        self._stage2_relational_labels(messages)
        
        # Stage 3: Name Extraction
        self.logger.info("\n" + "=" * 40)
        self.logger.info("STAGE 3: Name Extraction")
        self.logger.info("=" * 40)
        self._stage3_proper_names(messages)
        
        # Final Results
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL RESULTS")
        self.logger.info("=" * 60)
        
        for speaker, identity in self.speaker_identities.items():
            self.logger.info(f"\n{speaker}:")
            self.logger.info(f"  Final Label: '{identity.current_label}'")
            self.logger.info(f"  Quality: {identity.quality.name}")
            self.logger.info(f"  Confidence: {identity.confidence:.2f}")
            
            if identity.label_history:
                self.logger.info(f"  Progression:")
                for step in identity.label_history:
                    self.logger.info(f"    {step['from_quality']} → {step['to_quality']}: "
                                   f"'{step['from_label']}' → '{step['to_label']}'")
        
        self.logger.debug(f"Processing complete. {len(self.speaker_identities)} speakers labeled")
        return self.speaker_identities
    
    def _stage1_descriptive_labels(self, messages: List[Dict],
                                   audio_files: Optional[List[str]] = None):
        """Stage 1: Discover descriptive patterns from text and audio"""
        speaker_texts = defaultdict(list)
        for msg in messages:
            speaker = msg.get('speaker', 'unknown')
            speaker_texts[speaker].append(msg.get('text', ''))
        
        # Discover topics from all text
        all_texts = [msg.get('text', '') for msg in messages if msg.get('text', '').strip()]
        topics = self.pattern_discoverer.discover_topics(all_texts)
        
        if topics:
            self.logger.info(f"  Discovered {len(topics)} topics:")
            for topic_id, keywords in topics.items():
                self.logger.info(f"    Topic {topic_id}: {', '.join(keywords[:5])}")
        
        # Process audio features if available
        if audio_files:
            self.logger.info(f"\n  Processing {len(audio_files)} audio files...")
            for i, audio_file in enumerate(audio_files):
                if os.path.exists(audio_file):
                    features = self.audio_processor.extract_features(audio_file)
                    
                    # Map audio file to speaker
                    for msg in messages:
                        if msg.get('audio_file') == audio_file:
                            speaker = msg.get('speaker')
                            if speaker:
                                self._stage1_audio_features[speaker] = features
                                break
                    
                    self.logger.debug(f"    Audio features extracted: {Path(audio_file).name}")
        
        # Analyze each speaker's patterns
        for speaker, texts in speaker_texts.items():
            identity = self.speaker_identities.get(speaker)
            if not identity:
                continue
            
            # Discover speech patterns
            patterns = self.pattern_discoverer.discover_speech_patterns(texts)
            topic_dist = self.pattern_discoverer.get_speaker_topic_distribution(texts)
            
            # Build descriptive label from patterns
            descriptor_parts = []
            
            avg_words = patterns.get('avg_words_per_turn', 0)
            if avg_words > 20:
                descriptor_parts.append("detailed")
            elif avg_words > 8:
                descriptor_parts.append("moderate")
            else:
                descriptor_parts.append("concise")
            
            question_ratio = patterns.get('question_ratio', 0)
            if question_ratio > 0.4:
                descriptor_parts.append("inquiring")
            elif question_ratio < 0.1:
                descriptor_parts.append("declarative")
            
            politeness = patterns.get('politeness_ratio', 0)
            if politeness > 0.3:
                descriptor_parts.append("polite")
            
            urgency = patterns.get('urgency_ratio', 0)
            if urgency > 0.1:
                descriptor_parts.append("urgent")
            
            # Add topic focus if available
            if topic_dist:
                best_topic = max(topic_dist, key=topic_dist.get)
                if best_topic in topics and topics[best_topic]:
                    top_word = topics[best_topic][0]
                    descriptor_parts.append(f"focuses_on_{top_word}")
            
            if descriptor_parts:
                new_label = '_'.join(descriptor_parts[:3])
            else:
                new_label = "conversational_participant"
            
            # Update identity
            identity.linguistic_profile = patterns
            identity.topic_distribution = topic_dist
            identity.avg_turn_length = avg_words
            identity.question_ratio = question_ratio
            
            identity.upgrade_label(
                new_label=new_label,
                quality=LabelQuality.DESCRIPTIVE,
                confidence=0.7,
                evidence={
                    'patterns': {k: round(v, 3) for k, v in patterns.items()},
                    'topics': {str(k): [topics.get(k, [])[0]] for k in topic_dist}
                }
            )
            
            self.logger.info(f"\n  {speaker}:")
            self.logger.info(f"    Patterns: avg_words={avg_words:.1f}, questions={question_ratio:.2f}")
            self.logger.info(f"    Label: '{speaker}' → '{new_label}'")
            self.logger.debug(f"    Full patterns: {json.dumps(patterns, indent=2)}")
    
    def _stage2_relational_labels(self, messages: List[Dict]):
        """Stage 2: Zero-shot identity and relationship detection"""
        self.logger.info("  Analyzing conversation with zero-shot detection...")
        analysis = self.identity_detector.analyze_conversation(messages)
        
        if not analysis:
            self.logger.info("  No analysis available, keeping descriptive labels")
            return
        
        for speaker, identity in self.speaker_identities.items():
            if speaker not in analysis:
                continue
            
            speaker_analysis = analysis[speaker]
            observed_role = speaker_analysis.get('observed_role', '')
            relationship = speaker_analysis.get('relationship_to_others', '')
            name_candidates = speaker_analysis.get('name_candidates', [])
            context = speaker_analysis.get('conversation_context', '')
            confidence = speaker_analysis.get('confidence', 0.7)
            
            # Build new label based on analysis
            if observed_role:
                new_label = observed_role.replace(' ', '_')
            elif relationship:
                new_label = relationship.replace(' ', '_')
            else:
                new_label = identity.current_label
            
            # Store name candidates
            if name_candidates:
                identity.name_candidates = {
                    name: 0.7 for name in name_candidates
                }
            
            # Upgrade label
            identity.upgrade_label(
                new_label=new_label,
                quality=LabelQuality.RELATIONAL,
                confidence=confidence,
                evidence={
                    'observed_role': observed_role,
                    'relationship': relationship,
                    'context': context,
                    'name_candidates': name_candidates
                }
            )
            
            self.logger.info(f"\n  {speaker}:")
            self.logger.info(f"    Observed role: {observed_role}")
            self.logger.info(f"    Relationship: {relationship}")
            if name_candidates:
                self.logger.info(f"    Possible names: {name_candidates}")
            self.logger.info(f"    Label: → '{new_label}'")
            self.logger.debug(f"    Full analysis: {json.dumps(speaker_analysis, indent=2)}")
    
    def _stage3_proper_names(self, messages: List[Dict]):
        """Stage 3: Extract and validate proper names"""
        for speaker, identity in self.speaker_identities.items():
            if not identity.name_candidates:
                continue
            
            speaker_msgs = [m for m in messages if m.get('speaker') == speaker]
            speaker_texts = [m.get('text', '') for m in speaker_msgs]
            
            best_name = None
            best_score = 0.0
            
            # Validate each name candidate
            for name in identity.name_candidates:
                score = self._validate_name(name, messages, speaker, speaker_texts)
                self.logger.debug(f"    Name '{name}' validation score: {score:.2f}")
                if score > best_score:
                    best_score = score
                    best_name = name
            
            # Apply best name if confidence is high enough
            if best_name and best_score > 0.5:
                identity.upgrade_label(
                    new_label=best_name,
                    quality=LabelQuality.PROPER_NAME,
                    confidence=best_score,
                    evidence={
                        'validated_name': best_name,
                        'validation_score': best_score,
                        'other_candidates': list(identity.name_candidates.keys())
                    }
                )
                
                self.logger.info(f"\n  {speaker}:")
                self.logger.info(f"    Best name: '{best_name}' (score: {best_score:.2f})")
                self.logger.info(f"    Label: → '{best_name}'")
            else:
                self.logger.debug(f"  {speaker}: No name met validation threshold")
    
    def _validate_name(self, name: str, messages: List[Dict],
                      target_speaker: str, speaker_texts: List[str]) -> float:
        """Validate if a name likely belongs to this speaker"""
        score = 0.0
        
        # Check if speaker introduces themselves with this name
        for text in speaker_texts:
            if re.search(rf"(?:my name(?:'s| is)|I(?:'m| am))\s+{name}", text, re.IGNORECASE):
                score += 0.5
                break
        
        # Check if others address this speaker by name
        for msg in messages:
            if msg.get('speaker') != target_speaker:
                text = msg.get('text', '')
                if re.match(rf'^{name}[,!]', text.strip(), re.IGNORECASE):
                    score += 0.3
        
        return min(score, 1.0)


if __name__ == "__main__":
    from _main_speaker_labeling_system import main
    main()
