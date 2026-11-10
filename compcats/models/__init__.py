"""
This module initializes and imports various neural network models.
"""

##########################
# Evaluation
##########################

# Heavy optional deps (timm, clip, open_clip, dreamsim, ...) are only needed for the
# pretrained zoo. If they are missing, defer the error to the first attribute access so
# that lightweight models (custommodel, qwen3vl) can be used without them.
try:
    from .evaluate.imagenet1k import evaluate as imagenet1k_evaluataion
except ImportError:
    pass


##########################
# Pretrained models
##########################

try:
    from .pretrained import *
except ImportError as _e:
    _import_error = _e
    def __getattr__(name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise AttributeError(
            f"compcats.models.{name} needs optional dependencies that are not installed "
            f"({_import_error})"
        ) from _import_error
