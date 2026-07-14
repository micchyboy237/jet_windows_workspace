"""Outlier buffer orchestration for speaker labeling.

Extracted from SegmentSpeakerLabeler to keep the main class focused on
core labeling logic. Handles outlier-aware labeling flows.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from rich.console import Console

try:
    from services.speaker_labeler_utils.outlier_pool import OutlierMatch
    # from services.segment_speaker_labeler import SegmentSpeakerLabeler
except ImportError:
    from speaker_labeler_utils.outlier_pool import OutlierMatch
    # from segment_speaker_labeler import SegmentSpeakerLabeler

console = Console()


class OutlierOrchestrator:
    """Orchestrates the outlier buffer labeling flow."""

    # def __init__(self, labeler: "SegmentSpeakerLabeler", debug: bool = False):
    def __init__(self, labeler, debug: bool = False):
        self._labeler = labeler
        self.debug = debug

    # ------------------------------------------------------------------
    # Should-create-new-speaker decision
    # ------------------------------------------------------------------
    def should_create_new_speaker(
        self,
        best_score: float,
        top_matches: List[Dict],
        context: Optional[Dict],
        embedding: np.ndarray,
    ) -> bool:
        """Determine if we should create a new speaker."""
        labeler = self._labeler
        if len(labeler._speakers) == 0:
            return True
        if not top_matches:
            return True

        best_match = top_matches[0]
        match_type = best_match["match_type"]
        confidence = best_match["confidence"]

        if match_type in ("strong_match", "early_match"):
            return False
        if match_type == "possible_match":
            return False
        if confidence < labeler.threshold_new_speaker:
            if context and "previous_speaker" in context:
                prev_speaker = context["previous_speaker"]
                if prev_speaker and prev_speaker in labeler._speakers:
                    for match in top_matches:
                        if (
                            match["label"] == prev_speaker
                            and match["confidence"] >= labeler.threshold_possible
                        ):
                            return False
            return True
        if best_score < labeler.threshold_new_speaker:
            return True
        return False

    # ------------------------------------------------------------------
    # Label with outlier buffer
    # ------------------------------------------------------------------
    def label_with_outlier_buffer(
        self,
        embedding: np.ndarray,
        top_matches: List[Dict],
        actual_best_score: float,
        timestamp: float,
        context: Optional[Dict],
        segment_id: str,
        audio_duration: float,
    ) -> Tuple[List[Dict], bool]:
        """Label using the outlier buffer for speaker validation."""
        labeler = self._labeler
        outlier_matches = labeler.outlier_pool.find_matches(embedding)
        should_create = self.should_create_new_speaker(
            actual_best_score, top_matches, context, embedding
        )
        just_created_speaker = False

        if should_create or not top_matches:
            if outlier_matches:
                best_match = outlier_matches[0]
                results, just_created_speaker = self.handle_outlier_promotion(
                    outlier_matches=outlier_matches,
                    embedding=embedding,
                    timestamp=timestamp,
                    segment_id=segment_id,
                    audio_duration=audio_duration,
                )
            else:
                results = self.handle_new_outlier(
                    embedding=embedding,
                    timestamp=timestamp,
                    segment_id=segment_id,
                    audio_duration=audio_duration,
                )
        else:
            primary_match = top_matches[0]
            if outlier_matches and primary_match["confidence"] < labeler.threshold_same:
                best_outlier = outlier_matches[0]
                if best_outlier.confidence > primary_match["confidence"]:
                    results, just_created_speaker = self.handle_outlier_promotion(
                        outlier_matches=outlier_matches,
                        embedding=embedding,
                        timestamp=timestamp,
                        segment_id=segment_id,
                        audio_duration=audio_duration,
                    )
                    return results, just_created_speaker
                for om in outlier_matches[:2]:
                    if om.confidence >= labeler.outlier_pool.promotion_threshold:
                        self.merge_outlier_into_speaker(
                            outlier_label=om.outlier_label,
                            speaker_label=primary_match["label"],
                            similarity=om.confidence,
                            timestamp=timestamp,
                        )
            results = labeler._build_standard_results(
                top_matches=top_matches,
                embedding=embedding,
                timestamp=timestamp,
                context=context,
                segment_id=segment_id,
                audio_duration=audio_duration,
            )
        return results, just_created_speaker

    # ------------------------------------------------------------------
    # Outlier promotion
    # ------------------------------------------------------------------
    def handle_outlier_promotion(
        self,
        outlier_matches: List["OutlierMatch"],
        embedding: np.ndarray,
        timestamp: float,
        segment_id: str,
        audio_duration: float,
    ) -> Tuple[List[Dict], bool]:
        """Promote matching outlier(s) to a full speaker with validation.
        
        Improved logic:
        1. Validate that the current embedding and outlier embedding are 
        sufficiently similar (above threshold_same).
        2. If below threshold_same but above threshold_possible, create the 
        speaker but mark it as 'young' with low centroid quality.
        3. Log all promotion validation details for traceability.
        """
        labeler = self._labeler
        best_match = outlier_matches[0]
        matched_outlier = best_match.outlier_entry
        
        # --- NEW: Cross-validate the two embeddings ---
        from scipy.spatial.distance import cdist
        
        outlier_emb = matched_outlier.embedding.reshape(1, -1) if matched_outlier.embedding.ndim == 1 else matched_outlier.embedding
        current_emb = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding
        
        cross_similarity = float(1.0 - cdist(outlier_emb, current_emb, metric="cosine")[0, 0])
        
        # Determine quality of the new speaker based on cross-similarity
        if cross_similarity >= labeler.threshold_same:
            quality_tier = "high"
            match_type = "outlier_promotion_strong"
        elif cross_similarity >= labeler.threshold_possible:
            quality_tier = "medium"
            match_type = "outlier_promotion_possible"
        elif cross_similarity >= labeler.outlier_pool.promotion_threshold:
            quality_tier = "low"
            match_type = "outlier_promotion_weak"
        else:
            # Fallback: cross-similarity below promotion threshold —
            # still promote but log a warning; the alternative is to keep
            # the segment as a new outlier
            quality_tier = "minimal"
            match_type = "outlier_promotion_minimal"
            if self.debug:
                console.print(
                    f"[yellow]⚠️  Low cross-similarity ({cross_similarity:.3f}) "
                    f"between outlier {best_match.outlier_label} and current segment. "
                    f"Promoting anyway but centroid quality will be low.[/]"
                )
        
        # --- Create the new speaker ---
        new_label = labeler.create_new_speaker(
            embedding=matched_outlier.embedding,
            timestamp=matched_outlier.timestamp,
            segment_id=matched_outlier.segment_id,
            audio_duration=matched_outlier.audio_duration,
        )
        
        # Promote and remove outlier from pool
        labeler.outlier_pool.promote_single(
            label=best_match.outlier_label,
            timestamp=timestamp,
            target_speaker=new_label,
            confidence=best_match.confidence,
        )
        
        # Add current embedding with appropriate match_type
        labeler.update_reference(
            label=new_label,
            embedding=embedding,
            timestamp=timestamp,
            segment_id=segment_id,
            audio_duration=audio_duration,
            match_type=match_type,  # Now reflects actual quality
        )
        
        # --- NEW: Log detailed promotion analytics ---
        promotion_details = {
            "new_speaker": new_label,
            "outlier_label": best_match.outlier_label,
            "outlier_confidence": round(best_match.confidence, 4),
            "cross_similarity": round(cross_similarity, 4),
            "quality_tier": quality_tier,
            "outlier_segment_id": matched_outlier.segment_id,
            "current_segment_id": segment_id,
        }
        
        if self.debug:
            console.print(
                f"[bold green]🎉 Outlier Promotion: "
                f"{best_match.outlier_label} → {new_label}\n"
                f"   Cross-sim: {cross_similarity:.3f} "
                f"(tier: {quality_tier})\n"
                f"   Pool confidence: {best_match.confidence:.3f}\n"
                f"   Speakers: {labeler.speaker_count}[/]"
            )
        
        results = [{
            "label": new_label,
            "confidence": cross_similarity,  # Use cross-similarity, not pool confidence
            "match_type": match_type,
            "is_primary": True,
            "is_new_speaker": True,
            "segment_count": labeler._speakers[new_label].segment_count,
            "last_seen": timestamp,
            "segment_id": segment_id,
            "promoted_from_outlier": True,
            "promotion_details": promotion_details,  # Include for downstream analysis
        }]
        
        return results, True

    # ------------------------------------------------------------------
    # New outlier creation
    # ------------------------------------------------------------------
    def handle_new_outlier(
        self,
        embedding: np.ndarray,
        timestamp: float,
        segment_id: str,
        audio_duration: float,
    ) -> List[Dict]:
        """Create a new outlier entry for an unmatched segment."""
        labeler = self._labeler
        outlier_label = labeler.outlier_pool.add(
            embedding=embedding,
            timestamp=timestamp,
            segment_id=segment_id,
            audio_duration=audio_duration,
        )
        if self.debug:
            console.print(
                f"[yellow]📦 No matches → outlier: {outlier_label} "
                f"(pool size: {labeler.outlier_pool.count})[/]"
            )
        return [{
            "label": outlier_label,
            "confidence": 1.0,
            "match_type": "outlier_pending",
            "is_primary": True,
            "is_new_speaker": False,
            "is_outlier": True,
            "segment_count": 1,
            "last_seen": timestamp,
            "segment_id": segment_id,
        }]

    # ------------------------------------------------------------------
    # Merge outlier into speaker
    # ------------------------------------------------------------------
    def merge_outlier_into_speaker(
        self,
        outlier_label: str,
        speaker_label: str,
        similarity: float,
        timestamp: float,
    ) -> bool:
        """Merge an outlier into an existing speaker."""
        labeler = self._labeler
        outlier = labeler.outlier_pool.get(outlier_label)
        if outlier is None or speaker_label not in labeler._speakers:
            return False

        labeler.update_reference(
            label=speaker_label,
            embedding=outlier.embedding,
            timestamp=timestamp,
            segment_id=outlier.segment_id,
            audio_duration=outlier.audio_duration,
            match_type="possible_match" if similarity < labeler.threshold_same else "strong_match",
        )
        labeler.outlier_pool.remove(outlier_label)
        labeler.outlier_pool.record_promotion(
            type_="merge",
            outlier_labels=[outlier_label],
            target_speaker=speaker_label,
            confidence=similarity,
            timestamp=timestamp,
        )
        if self.debug:
            console.print(
                f"[blue]🔗 Merged outlier: {outlier_label} → {speaker_label} "
                f"(sim={similarity:.3f})[/]"
            )
        return True
