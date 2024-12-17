import argparse


def get_args(dataset: str, default_lr: float, default_momentum: float = 0):
    parser = argparse.ArgumentParser()

    # Learning settings
    parser.add_argument('--learning-rate', type=float, default=default_lr)
    parser.add_argument('--momentum', type=float, default=default_momentum)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--local-steps', type=int, default=5)

    # Accuracy testing
    parser.add_argument('--accuracy-logging-interval', type=int, default=5)
    parser.add_argument('--accuracy-logging-interval-is-in-sec', action=argparse.BooleanOptionalAction)

    # Traces
    parser.add_argument('--availability-traces', type=str, default=None)
    parser.add_argument('--traces', type=str, default="none", choices=["none", "fedscale", "diablo"])
    parser.add_argument('--min-bandwidth', type=int, default=0)  # The minimum bandwidth a node must have to participate, in bytes/s.
    parser.add_argument('--seed', type=int, default=42)

    # Other settings
    parser.add_argument('--log-level', type=str, default="INFO")
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--dataset-base-path', type=str, default=None)
    parser.add_argument('--duration', type=int, default=3600)  # Set to 0 to run forever
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--partitioner', type=str, default="iid")
    parser.add_argument('--peers', type=int, default=10)
    parser.add_argument('--train-device-name', type=str, default="cpu")
    parser.add_argument('--accuracy-device-name', type=str, default="cpu")
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction)
    parser.add_argument('--latencies-file', type=str, default="data/latencies.txt")
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--activity-log-interval', type=int, default=None)
    parser.add_argument('--chunks-in-sample', type=int, default=10)
    parser.add_argument('--success-fraction', type=float, default=1)

    return parser.parse_args()
