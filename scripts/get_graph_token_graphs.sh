#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <task>" >&2
    echo "Supported tasks: cycle-check | shortest-path" >&2
    exit 1
fi

task="$1"
case "$task" in
    cycle-check)
        algorithms=(er ba sbm sfn complete star path)
        num_graphs=500
        output_subdir="graphs"
        ;;
    shortest-path)
        algorithms=(path sfn)
        num_graphs=2000
        output_subdir="shortest_path_graphs"
        ;;
    *)
        echo "Unknown task: $task" >&2
        echo "Supported tasks: cycle-check | shortest-path" >&2
        exit 1
        ;;
esac

cd graph-token
set -x
if [[ ! -d graphenv ]]; then
    python3 -m venv graphenv
fi
source graphenv/bin/activate
pip install -r requirements.txt

output_dir="../graph_token_outputs/${output_subdir}"
mkdir -p "$output_dir"

for algorithm in "${algorithms[@]}"; do
    for split in train test; do
        python graph_generator.py \
            --algorithm="$algorithm" \
            --number_of_graphs="$num_graphs" \
            --split="$split" \
            --output_path="$output_dir"
    done
done

deactivate
cd ..
