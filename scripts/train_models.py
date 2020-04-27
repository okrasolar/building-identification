import sys
from argparse import ArgumentParser

sys.path.append("..")

from src.models import STR2MODEL, train_model


if __name__ == "__main__":
    parser = ArgumentParser()

    # figure out which model to use
    parser.add_argument(
        "--model_name",
        type=str,
        default="classifier",
        help="One of {classifier, segmenter}.",
    )
    parser.add_argument("--max_epochs", type=int, default=1000)

    temp_args = parser.parse_known_args()[0]

    model_args = (
        STR2MODEL[temp_args.model_name].add_model_specific_args(parser).parse_args()
    )
    model = STR2MODEL[temp_args.model_name](model_args)

    train_model(model, model_args)
