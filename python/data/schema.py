'''Quality schema definitions for the Retino project.

This module defines enumerations and data structures that are shared across
the pipeline components.  At the moment only :class:`QualityStatus` is used
by the inference pipeline and the test suite.

The enum values are deliberately verbose strings because they are stored
in JSON reports and displayed in the UI.  The enum members are named in
ALL_CAPS to match the usage throughout the code base.
''' 

from enum import Enum


class QualityStatus(Enum):
    """Image quality assessment outcomes.

    The pipeline classifies an image into one of three coarse quality
    categories.  ``UNGRADEABLE`` is used for images that are so poor that a
    clinical decision cannot be made.  ``BORDERLINE`` indicates that the
    image is marginal and may be enhanced before further analysis.
    ``GOOD`` represents an image that meets the quality thresholds.
    """

    GOOD = "Good"
    BORDERLINE = "Borderline"
    POOR = "Poor"
    UNGRADEABLE = "Ungradeable"
