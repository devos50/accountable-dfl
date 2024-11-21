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
    parser.add_argument('--dl-test-mode', type=str, default="local")
    parser.add_argument('--das-test-subprocess-jobs', type=int, default=1)
    parser.add_argument('--das-test-num-models-per-subprocess', type=int, default=10)
    parser.add_argument('--accuracy-logging-interval', type=int, default=5)
    parser.add_argument('--accuracy-logging-interval-is-in-sec', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dl-accuracy-method', type=str, default="individual")  # individual or aggregate

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
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--partitioner', type=str, default="iid")
    parser.add_argument('--peers', type=int, default=10)
    parser.add_argument('--active-participants', type=str, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=None)
    parser.add_argument('--train-device-name', type=str, default="cpu")
    parser.add_argument('--accuracy-device-name', type=str, default="cpu")
    parser.add_argument('--bypass-model-transfers', action=argparse.BooleanOptionalAction)
    parser.add_argument('--bypass-training', action=argparse.BooleanOptionalAction)
    parser.add_argument('--store-best-models', action=argparse.BooleanOptionalAction)
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction)
    parser.add_argument('--log-events', action=argparse.BooleanOptionalAction)
    parser.add_argument('--topology', type=str, default="exp-one-peer")
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--latencies-file', type=str, default="data/latencies.txt")
    parser.add_argument('--gl-round-timeout', type=int, default=60)
    parser.add_argument('--dl-round-timeout', type=int, default=120)
    parser.add_argument('--fix-aggregator', action=argparse.BooleanOptionalAction)
    parser.add_argument('--success-fraction', type=float, default=1.0)
    parser.add_argument('--liveness-success-fraction', type=float, default=0.4)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--num-aggregators', type=int, default=1)
    parser.add_argument('--aggregation-timeout', type=float, default=300)
    parser.add_argument('--activity-log-interval', type=int, default=None)
    parser.add_argument('--flush-statistics-interval', type=int, default=600)
    parser.add_argument('--write-view-histories', action=argparse.BooleanOptionalAction)
    parser.add_argument('--chunks-in-sample', type=int, default=10)
    parser.add_argument('--gossip-interval', type=float, default=60)

    return parser.parse_args()
