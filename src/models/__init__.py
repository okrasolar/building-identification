from .models import Classifier, Segmenter
from .train_funcs import train_model


__all__ = ["Classifier", "Segmenter", "train_model", "STR2MODEL"]


STR2MODEL = {"classifier": Classifier, "segmenter": Segmenter}
