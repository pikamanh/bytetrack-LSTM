# Lazy imports — avoid importing torch at package-load time
# (needed so prepare_mot_tracklets.py can run without torch)

def __getattr__(name):
    if name == "MotionEncoder":
        from .motion_encoder import MotionEncoder
        return MotionEncoder
    if name == "FiLMLayer":
        from .film_layer import FiLMLayer
        return FiLMLayer
    if name in ("FiLMReIDModel", "build_film_reid_model"):
        from .film_resnest_reid import FiLMReIDModel, build_film_reid_model
        return locals()[name]
    raise AttributeError(f"module 'yolox.FiLM' has no attribute {name!r}")
