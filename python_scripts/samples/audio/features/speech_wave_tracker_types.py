from enum import StrEnum
from typing import TypedDict


class WaveState(StrEnum):
    """
    Finite-state machine states for wave detection lifecycle.
    """

    BELOW = "below"
    RISING = "rising"
    SUSTAINED = "sustained"
    FALLING = "falling"
    VALIDATED = "validated"
    REJECTED = "rejected"


class ValidationResult(StrEnum):
    """
    Final validation result of the detected wave.
    """

    VALID = "valid"
    INVALID = "invalid"


class ThresholdDirection(StrEnum):
    """
    Describes probability movement relative to threshold.
    """

    ABOVE = "above"
    BELOW = "below"


class TransitionReason(StrEnum):
    """
    Explains why a transition occurred.
    """

    PROBABILITY_ABOVE_THRESHOLD = "probability_above_threshold"
    PROBABILITY_BELOW_THRESHOLD = "probability_below_threshold"

    SUSTAINED_MULTI_PASS = "sustained_multi_pass"

    SHAPE_CHECK_PASSED = "shape_check_passed"
    SHAPE_CHECK_FAILED = "shape_check_failed"

    DURATION_CHECK_PASSED = "duration_check_passed"
    DURATION_CHECK_FAILED = "duration_check_failed"

    VALIDATED = "validated"
    REJECTED = "rejected"


class WaveFlags(TypedDict):
    """
    Lifecycle flags that become permanently true
    once their corresponding phase has occurred.
    """

    has_risen: bool
    has_multi_passed: bool
    has_fallen: bool
    is_valid: bool


class WaveValidationDetails(TypedDict, total=False):
    """
    Validation metrics collected during validation phase.
    """

    shape_valid: bool
    duration_valid: bool

    min_duration_met: bool
    max_duration_met: bool

    peak_probability: float
    average_probability: float

    rejection_reason: str


class WaveStateSnapshot(TypedDict):
    """
    Full snapshot of the wave state machine.
    """

    state: WaveState

    flags: WaveFlags

    validation_result: ValidationResult | None

    probability: float
    threshold: float

    transition_reason: TransitionReason | None

    validation: WaveValidationDetails
