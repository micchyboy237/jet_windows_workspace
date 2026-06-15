"""
segment_speaker_labeler_health_mixin.py
========================================
Drop-in mixin that adds chart-ready analytics methods to SegmentSpeakerLabeler
by delegating to CentroidHealthChecker.

Usage
-----
In segment_speaker_labeler.py, change the class declaration to:

    from segment_speaker_labeler_health_mixin import SpeakerLabelerHealthMixin

    class SegmentSpeakerLabeler(SpeakerLabelerHealthMixin):
        ...

Then the following new methods become available on every labeler instance:

    labeler.get_centroid_health_report()   → CentroidHealthReport (full typed object)
    labeler.get_centroid_health_dict()     → Dict  (JSON-serialisable, for APIs / charts)
    labeler.get_similarity_matrix_dict()   → Dict  (heatmap-ready)
    labeler.get_speaker_insights()         → Dict  (summary cards, alerts, badges)
    labeler.get_chart_data()               → Dict  (all chart payloads in one call)

Design principles
-----------------
- Zero coupling: the mixin only reads ``self._speakers`` (same attribute used by
  SegmentSpeakerLabeler internally).  It never writes to it.
- Adapter pattern: ``_build_embeddings_per_label()`` converts SpeakerReference
  embeddings (which may be shaped (1,D) or (D,)) into the flat List[Embedding]
  that CentroidHealthChecker expects.
- HealthThresholds are derived from the labeler's own thresholds so numbers are
  consistent with the running system. Can be overridden per-call.
- All public methods return plain dicts or typed dataclasses — callers choose.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from services.helpers.speaker_centroid_health import (
        CentroidHealth,
        CentroidHealthChecker,
        CentroidHealthReport,
        Embedding,
        HealthFlag,
        HealthThresholds,
        LabelID,
    )
except ImportError:
    from helpers.speaker_centroid_health import (
        CentroidHealth,
        CentroidHealthChecker,
        CentroidHealthReport,
        Embedding,
        HealthFlag,
        HealthThresholds,
        LabelID,
    )


# ---------------------------------------------------------------------------
# Internal type alias (mirrors what SpeakerReference holds)
# ---------------------------------------------------------------------------

class SpeakerLabelerHealthMixin:
    """
    Mixin for SegmentSpeakerLabeler.

    Assumes the host class exposes:
        self._speakers : Dict[str, SpeakerReference]
            SpeakerReference has .embeddings: List[np.ndarray]
            and .centroid: Optional[np.ndarray]

        self.consolidation_threshold : float
        self.threshold_same          : float
        self.mature_segment_count    : int
        self.young_segment_count     : int
    """

    # ------------------------------------------------------------------
    # Core adapter
    # ------------------------------------------------------------------

    def _build_embeddings_per_label(self) -> Dict[LabelID, List[Embedding]]:
        """
        Convert SpeakerReference.embeddings → Dict[label, List[flat Embedding]].

        SpeakerReference stores embeddings as List[np.ndarray] where each array
        may be shaped (1, D) or (D,).  CentroidHealthChecker expects (D,) vectors.
        """
        result: Dict[LabelID, List[Embedding]] = {}
        for label, ref in self._speakers.items():  # type: ignore[attr-defined]
            if not ref.embeddings:
                continue
            flat: List[Embedding] = []
            for emb in ref.embeddings:
                arr = np.asarray(emb, dtype=np.float32)
                flat.append(arr.flatten())          # (1,D) → (D,) safely
            result[label] = flat
        return result

    def _make_health_thresholds(
        self,
        override: Optional[HealthThresholds] = None,
    ) -> HealthThresholds:
        """
        Build HealthThresholds from the labeler's own parameters so all numbers
        are consistent.  Caller may pass an override to use custom values.
        """
        if override is not None:
            return override
        return HealthThresholds(
            min_embedding_count=self.mature_segment_count,          # type: ignore[attr-defined]
            min_mean_cosine_sim=self.threshold_same - 0.20,         # type: ignore[attr-defined]
            max_intra_spread=1.0 - (self.threshold_same - 0.20),    # type: ignore[attr-defined]
            min_silhouette_score=0.10,
            max_inter_centroid_similarity=self.consolidation_threshold - 0.05,  # type: ignore[attr-defined]
            merge_similarity_threshold=self.consolidation_threshold,            # type: ignore[attr-defined]
        )

    def _build_checker(
        self,
        thresholds: Optional[HealthThresholds] = None,
    ) -> CentroidHealthChecker:
        """Instantiate a CentroidHealthChecker from the current speaker state."""
        return CentroidHealthChecker(
            embeddings_per_label=self._build_embeddings_per_label(),
            thresholds=self._make_health_thresholds(thresholds),
        )

    # ------------------------------------------------------------------
    # 1. Typed report object  (for internal / Python consumers)
    # ------------------------------------------------------------------

    def get_centroid_health_report(
        self,
        thresholds: Optional[HealthThresholds] = None,
    ) -> Optional[CentroidHealthReport]:
        """
        Run a full centroid health check and return a typed CentroidHealthReport.

        Returns None when no speakers have embeddings yet.

        Parameters
        ----------
        thresholds : HealthThresholds, optional
            Override the default thresholds derived from the labeler's config.

        Returns
        -------
        CentroidHealthReport or None
        """
        emb_map = self._build_embeddings_per_label()
        if not emb_map:
            return None
        checker = CentroidHealthChecker(
            embeddings_per_label=emb_map,
            thresholds=self._make_health_thresholds(thresholds),
        )
        return checker.check_all()

    # ------------------------------------------------------------------
    # 2. JSON-serialisable health dict  (for REST APIs, dashboards)
    # ------------------------------------------------------------------

    def get_centroid_health_dict(
        self,
        thresholds: Optional[HealthThresholds] = None,
    ) -> Dict[str, Any]:
        """
        Full centroid health as a JSON-serialisable dict.

        Schema
        ------
        {
          "speaker_count": int,
          "healthy_count": int,
          "unhealthy_count": int,
          "merge_candidate_count": int,
          "merge_candidates": [
              {"label_a": str, "label_b": str, "similarity": float}, ...
          ],
          "speakers": {
            "<label>": {
              "label": str,
              "embedding_count": int,
              "flags": [str, ...],           # flag names
              "is_healthy": bool,
              "cohesion": float,             # mean cosine sim to centroid
              "spread": float,               # mean cosine distance within cluster
              "silhouette": float,
              "nearest_label": str | None,
              "nearest_similarity": float | None,
              "segment_count": int,          # from SpeakerReference
              "centroid_quality": float,     # from SpeakerReference
              "first_seen": float,
              "last_seen": float,
            }, ...
          },
          "thresholds": { ... }
        }
        """
        report = self.get_centroid_health_report(thresholds)
        if report is None:
            return {"speaker_count": 0, "speakers": {}, "error": "no_speakers"}

        speakers_out: Dict[str, Any] = {}
        for label, h in report.results.items():
            ref = self._speakers.get(label)  # type: ignore[attr-defined]
            speakers_out[label] = {
                "label": label,
                "embedding_count": h.embedding_count,
                "flags": [f.name for f in h.flags],
                "is_healthy": h.is_healthy,
                "cohesion": round(h.mean_cosine_sim_to_centroid, 4),
                "spread": round(h.intra_cluster_spread, 4),
                "silhouette": round(h.silhouette_score, 4),
                "nearest_label": h.nearest_centroid_label,
                "nearest_similarity": (
                    round(h.nearest_centroid_similarity, 4)
                    if h.nearest_centroid_similarity is not None else None
                ),
                # Enrich from SpeakerReference (available in the labeler)
                "segment_count": ref.segment_count if ref else h.embedding_count,
                "centroid_quality": ref.centroid_quality if ref else 0.0,
                "first_seen": (ref.first_seen or 0.0) if ref else 0.0,
                "last_seen": ref.last_seen if ref else 0.0,
            }

        t = report.thresholds
        return {
            "speaker_count": len(report.results),
            "healthy_count": len(report.healthy_labels),
            "unhealthy_count": len(report.unhealthy_labels),
            "merge_candidate_count": len(report.merge_candidates),
            "merge_candidates": [
                {"label_a": a, "label_b": b, "similarity": round(sim, 4)}
                for a, b, sim in report.merge_candidates
            ],
            "speakers": speakers_out,
            "thresholds": {
                "min_embedding_count": t.min_embedding_count,
                "min_mean_cosine_sim": t.min_mean_cosine_sim,
                "max_intra_spread": t.max_intra_spread,
                "min_silhouette_score": t.min_silhouette_score,
                "max_inter_centroid_similarity": t.max_inter_centroid_similarity,
                "merge_similarity_threshold": t.merge_similarity_threshold,
            },
        }

    # ------------------------------------------------------------------
    # 3. Similarity matrix  (heatmap chart)
    # ------------------------------------------------------------------

    def get_similarity_matrix_dict(self) -> Dict[str, Any]:
        """
        Pairwise cosine similarity matrix enriched with health flags.

        Replaces / supersedes the existing get_speaker_similarity_matrix().

        Schema
        ------
        {
          "labels": [str, ...],
          "matrix": [[float, ...], ...],       # shape: (N, N), values in [-1, 1]
          "segment_counts": [int, ...],
          "centroid_qualities": [float, ...],
          "flags_per_label": {label: [str, ...]},
          "is_healthy_per_label": {label: bool},
        }
        """
        report = self.get_centroid_health_report()
        if report is None or len(report.results) < 2:
            # Fall back to raw similarity without health overlay
            labels: List[str] = []
            centroids_list = []
            seg_counts: List[int] = []
            for label, ref in self._speakers.items():  # type: ignore[attr-defined]
                if ref.has_valid_centroid:
                    labels.append(label)
                    centroids_list.append(ref.centroid.flatten())
                    seg_counts.append(ref.segment_count)
            if len(labels) < 2:
                return {"labels": labels, "matrix": [], "segment_counts": seg_counts}
            mat = np.vstack(centroids_list)
            from scipy.spatial.distance import cdist
            sims = (1.0 - cdist(mat, mat, metric="cosine")).tolist()
            return {
                "labels": labels,
                "matrix": [[round(v, 4) for v in row] for row in sims],
                "segment_counts": seg_counts,
                "centroid_qualities": [],
                "flags_per_label": {},
                "is_healthy_per_label": {},
            }

        ordered = list(report.results.keys())
        checker = self._build_checker()
        centroids_arr = np.vstack([
            checker._centroids[lbl]
            for lbl in ordered
            if lbl in checker._centroids
        ])
        # Reorder to match only labels that made it into checker
        ordered = [lbl for lbl in ordered if lbl in checker._centroids]

        from scipy.spatial.distance import cdist as _cdist
        sims_np = 1.0 - _cdist(centroids_arr, centroids_arr, metric="cosine")

        flags_map: Dict[str, List[str]] = {
            lbl: [f.name for f in report.results[lbl].flags]
            for lbl in ordered
        }
        healthy_map: Dict[str, bool] = {
            lbl: report.results[lbl].is_healthy for lbl in ordered
        }
        refs = self._speakers  # type: ignore[attr-defined]
        seg_counts_ord = [refs[lbl].segment_count for lbl in ordered]
        qualities = [refs[lbl].centroid_quality for lbl in ordered]

        return {
            "labels": ordered,
            "matrix": [[round(float(v), 4) for v in row] for row in sims_np.tolist()],
            "segment_counts": seg_counts_ord,
            "centroid_qualities": qualities,
            "flags_per_label": flags_map,
            "is_healthy_per_label": healthy_map,
        }

    # ------------------------------------------------------------------
    # 4. Insights / summary cards  (UI alert panels, badges)
    # ------------------------------------------------------------------

    def get_speaker_insights(self) -> Dict[str, Any]:
        """
        High-level insights for dashboard summary cards and alert banners.

        Schema
        ------
        {
          "total_speakers": int,
          "healthy_speakers": int,
          "unhealthy_speakers": int,
          "system_health": "good" | "warning" | "critical",
          "alerts": [
            {"level": "info"|"warning"|"error", "message": str, "labels": [str]},
            ...
          ],
          "badges": {
            "<label>": {
              "label": str,
              "badge": "✅ Healthy" | "⚠️ Diffuse" | "🚫 Contaminated" | ...,
              "color": "green" | "yellow" | "red" | "grey",
            }, ...
          },
          "top_merge_candidates": [
            {"label_a": str, "label_b": str, "similarity": float}, ...
          ],
          "flag_summary": {"HEALTHY": int, "IMMATURE": int, ...},
        }
        """
        report = self.get_centroid_health_report()

        if report is None:
            return {
                "total_speakers": 0,
                "healthy_speakers": 0,
                "unhealthy_speakers": 0,
                "system_health": "good",
                "alerts": [],
                "badges": {},
                "top_merge_candidates": [],
                "flag_summary": {},
            }

        # --- Badges ---
        _flag_badge: Dict[HealthFlag, tuple[str, str]] = {
            HealthFlag.HEALTHY:     ("✅ Healthy",      "green"),
            HealthFlag.IMMATURE:    ("🌱 Immature",     "grey"),
            HealthFlag.DIFFUSE:     ("🌊 Diffuse",      "yellow"),
            HealthFlag.TOO_CLOSE:   ("⚠️ Too Close",   "yellow"),
            HealthFlag.REDUNDANT:   ("♻️ Redundant",   "red"),
            HealthFlag.CONTAMINATED:("🚫 Contaminated","red"),
        }
        _flag_priority = [
            HealthFlag.CONTAMINATED, HealthFlag.REDUNDANT,
            HealthFlag.TOO_CLOSE, HealthFlag.DIFFUSE,
            HealthFlag.IMMATURE, HealthFlag.HEALTHY,
        ]

        def _worst_flag(flags: List[HealthFlag]) -> HealthFlag:
            for f in _flag_priority:
                if f in flags:
                    return f
            return HealthFlag.HEALTHY

        badges: Dict[str, Dict[str, str]] = {}
        for label, h in report.results.items():
            worst = _worst_flag(h.flags)
            badge_text, color = _flag_badge[worst]
            badges[label] = {"label": label, "badge": badge_text, "color": color}

        # --- Flag summary counts ---
        flag_summary: Dict[str, int] = {}
        for h in report.results.values():
            for f in h.flags:
                flag_summary[f.name] = flag_summary.get(f.name, 0) + 1

        # --- Alerts ---
        alerts: List[Dict[str, Any]] = []

        contaminated = [
            lbl for lbl, h in report.results.items()
            if HealthFlag.CONTAMINATED in h.flags
        ]
        if contaminated:
            alerts.append({
                "level": "error",
                "message": f"{len(contaminated)} speaker(s) have contaminated centroids",
                "labels": contaminated,
            })

        redundant = [
            lbl for lbl, h in report.results.items()
            if HealthFlag.REDUNDANT in h.flags
        ]
        if redundant:
            alerts.append({
                "level": "error",
                "message": f"{len(redundant)} speaker(s) are likely duplicates — consider merging",
                "labels": redundant,
            })

        too_close = [
            lbl for lbl, h in report.results.items()
            if HealthFlag.TOO_CLOSE in h.flags
        ]
        if too_close:
            alerts.append({
                "level": "warning",
                "message": f"{len(too_close)} speaker(s) are dangerously similar to another",
                "labels": too_close,
            })

        diffuse = [
            lbl for lbl, h in report.results.items()
            if HealthFlag.DIFFUSE in h.flags
        ]
        if diffuse:
            alerts.append({
                "level": "warning",
                "message": f"{len(diffuse)} speaker(s) have high intra-cluster spread",
                "labels": diffuse,
            })

        immature = [
            lbl for lbl, h in report.results.items()
            if HealthFlag.IMMATURE in h.flags
        ]
        if immature:
            alerts.append({
                "level": "info",
                "message": f"{len(immature)} speaker(s) need more data before they're reliable",
                "labels": immature,
            })

        # --- System health level ---
        if contaminated or redundant:
            system_health = "critical"
        elif too_close or diffuse:
            system_health = "warning"
        else:
            system_health = "good"

        return {
            "total_speakers": len(report.results),
            "healthy_speakers": len(report.healthy_labels),
            "unhealthy_speakers": len(report.unhealthy_labels),
            "system_health": system_health,
            "alerts": alerts,
            "badges": badges,
            "top_merge_candidates": [
                {"label_a": a, "label_b": b, "similarity": round(sim, 4)}
                for a, b, sim in report.merge_candidates[:5]
            ],
            "flag_summary": flag_summary,
        }

    # ------------------------------------------------------------------
    # 5. Per-speaker cohesion time-series  (line chart)
    # ------------------------------------------------------------------

    def get_cohesion_series(self) -> Dict[str, Any]:
        """
        Per-speaker cohesion as a function of embedding index.

        Useful for a "centroid drift over time" line chart — each data point
        is the cosine similarity of embedding[i] to the final centroid.

        Schema
        ------
        {
          "speakers": {
            "<label>": {
              "label": str,
              "series": [float, ...],      # cosine sim of each embedding to centroid
              "mean_cohesion": float,
              "min_cohesion": float,
              "trend": "stable" | "improving" | "degrading",
            }, ...
          }
        }
        """
        emb_map = self._build_embeddings_per_label()
        checker = CentroidHealthChecker(
            embeddings_per_label=emb_map,
            thresholds=self._make_health_thresholds(),
        )

        result: Dict[str, Any] = {"speakers": {}}
        for label, embs in checker._embeddings.items():
            centroid = checker._centroids[label]
            sims = (embs @ centroid).tolist()            # cosine sim, each embedding
            mean_coh = float(np.mean(sims))
            min_coh  = float(np.min(sims))

            # Simple trend: compare first-half vs second-half mean
            mid = max(len(sims) // 2, 1)
            first_half  = float(np.mean(sims[:mid]))
            second_half = float(np.mean(sims[mid:])) if len(sims) > 1 else first_half
            delta = second_half - first_half
            if abs(delta) < 0.02:
                trend = "stable"
            elif delta > 0:
                trend = "improving"
            else:
                trend = "degrading"

            result["speakers"][label] = {
                "label": label,
                "series": [round(s, 4) for s in sims],
                "mean_cohesion": round(mean_coh, 4),
                "min_cohesion": round(min_coh, 4),
                "trend": trend,
            }

        return result

    # ------------------------------------------------------------------
    # 6. All chart payloads in one call  (single round-trip for frontend)
    # ------------------------------------------------------------------

    def get_chart_data(
        self,
        thresholds: Optional[HealthThresholds] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate all chart-ready payloads in a single call.

        Returns
        -------
        {
          "health":      get_centroid_health_dict(),
          "similarity":  get_similarity_matrix_dict(),
          "insights":    get_speaker_insights(),
          "cohesion":    get_cohesion_series(),
          "summary_text": str,   # plain-text summary from CentroidHealthReport
        }

        This is the recommended endpoint for dashboard loads — avoids four
        separate calls and reuses the same CentroidHealthChecker state.
        """
        report = self.get_centroid_health_report(thresholds)

        return {
            "health":       self.get_centroid_health_dict(thresholds),
            "similarity":   self.get_similarity_matrix_dict(),
            "insights":     self.get_speaker_insights(),
            "cohesion":     self.get_cohesion_series(),
            "summary_text": report.summary() if report else "No speakers yet.",
        }
