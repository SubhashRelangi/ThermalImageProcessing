from .thermal_argumentation import (
    thermal_flip,
    thermal_rotate,
    add_gaussian_noise,
    add_salt_pepper_noise
)
from .thermal_argumentation_scores import (
    augmentation_difference_score
)

from .synthetic_thermal import synthetic_thermal
from .synthetic_thermal_scores import thermal_similarity_score

from .background_subtraction import (
    spatial_background_subtraction,
    video_background_subtraction
)