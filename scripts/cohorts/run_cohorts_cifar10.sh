#!/bin/bash

for cohort_value in "$@"; do
  echo "Evaluating cohort $cohort_value"
  python3 -u simulations/dfl/cifar10.py \
  --peers 200 \
  --num-aggregators 1 \
  --activity-log-interval 60 \
  --accuracy-logging-interval 1800 \
  --accuracy-logging-interval-is-in-sec \
  --duration 540000 \
  --bypass-model-transfers \
  --seed 24082 \
  --capability-trace data/client_device_capacity \
  --instant-network \
  --partitioner dirichlet \
  --alpha 0.1 \
  --fix-aggregator \
  --checkpoint-interval 18000 \
  --checkpoint-interval-is-in-sec \
  --cohort-file "cohorts.txt" \
  --train-device-name "cuda:0" \
  --accuracy-device-name "cuda:0" \
  --cohort $cohort_value > "output_${cohort_value}.log" 2>&1 &
done

wait
