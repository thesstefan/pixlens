import argparse

from pixlens.eval import grounding_dino, sam

parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models"
)


def main() -> None:
    parser.parse_args()

    sam.load_sam_predictor()
    grounding_dino.load_grounding_dino()

    print("PixLens - Hello!")


if __name__ == "__main__":
    main()
