#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <task> <method>" >&2
    echo "Tasks: cycle_check | shortest_path | zinc" >&2
    echo "Methods: graph-token | autograph" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 1
fi

task="$1"
method="$2"

case "$task" in
    cycle_check|shortest_path|zinc) ;;
    *)
        echo "Unknown task: $task" >&2
        usage
        exit 1
        ;;
esac

case "$method" in
    graph-token|autograph) ;;
    *)
        echo "Unknown method: $method" >&2
        usage
        exit 1
        ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

check_data() {
    local base="$1"
    shift
    pushd "$base" >/dev/null
    for rel in "$@"; do
        if [[ ! -e "$rel" ]]; then
            echo "Missing required data: $base/$rel" >&2
            exit 1
        fi
    done
    popd >/dev/null
}

python_cmd=()

case "$task:$method" in
    cycle_check:graph-token)
        check_data "graph_token_outputs" "tasks/cycle_check"
        python_cmd=(python sequence_transformer/cycle_check/train_graph_token.py --data-root "$ROOT_DIR/graph_token_outputs/tasks/cycle_check")
        ;;
    cycle_check:autograph)
        check_data "autograph_data" "cycle_check/train.txt" "cycle_check/test.txt"
        python_cmd=(python sequence_transformer/cycle_check/train_autograph.py --data-root "$ROOT_DIR/autograph_data/cycle_check")
        ;;
    shortest_path:graph-token)
        check_data "graph_token_outputs" "shortest_path_seqs/train.txt" "shortest_path_seqs/test.txt"
        python_cmd=(python sequence_transformer/shortest_paths/train.py "$ROOT_DIR/graph_token_outputs/shortest_path_seqs" --tokenization graph-token)
        ;;
    shortest_path:autograph)
        check_data "autograph_data" "shortest_path/train.txt" "shortest_path/test.txt"
        python_cmd=(python sequence_transformer/shortest_paths/train.py "$ROOT_DIR/autograph_data/shortest_path" --tokenization autograph)
        ;;
    zinc:graph-token)
        check_data "graph_token_outputs" "zinc_tokens.jsonl"
        python_cmd=(python sequence_transformer/zinc/train.py --tokens "$ROOT_DIR/graph_token_outputs/zinc_tokens.jsonl")
        ;;
    zinc:autograph)
        check_data "autograph_data" "zinc_tokens.jsonl"
        python_cmd=(python sequence_transformer/zinc/train.py --tokens "$ROOT_DIR/autograph_data/zinc_tokens.jsonl")
        ;;
    *)
        echo "Unsupported task/method combination: $task + $method" >&2
        exit 1
        ;;
esac

echo "Running: ${python_cmd[*]}"
"${python_cmd[@]}"
