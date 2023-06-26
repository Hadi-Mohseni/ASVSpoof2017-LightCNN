from argparse import ArgumentParser


def print_hello():
    print("hello")


parent_parser = ArgumentParser(
    "LightCNN_ASVSpoof2017",
    description="An implementation of anounced LightCNN trained on ASVSpoof2017",
    # usage="[test or predict] [--dataset_path --annot_path or --path]",
)

subparsers = parent_parser.add_subparsers()
# -------------------- Test -------------------- #

test_parser = subparsers.add_parser("test", help="test a whole dataset")
test_parser.add_argument(
    "dataset",
    type=str,
    help="path to the dir containing test samples",
)
test_parser.add_argument(
    "annotation",
    type=str,
    help="comlete path to the test annotation file",
)

# -------------------- Predict -------------------- #
predict_parser = subparsers.add_parser("predict", help="test a sample")
predict_parser.add_argument(
    "path",
    type=str,
    help="complete path to the sample",
)


args = parent_parser.parse_args()
print(vars(args))
