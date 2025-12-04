#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <method>" >&2
    echo "method: graph-token | autograph" >&2
    exit 1
fi

method="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root"

case "$method" in
    graph-token)
        data_root="$repo_root/GraphGPS/datasets/ZINC"
        output_path="$repo_root/graph_token_outputs/zinc_tokens.jsonl"
        mkdir -p "$(dirname "$output_path")"
        cmd=(python "$repo_root/zinc_tokenization.py" --root "$data_root" --tokenizer graph_task --output "$output_path")
        ;;
    autograph)
        data_root="$repo_root/GraphGPS/datasets/ZINC"
        output_path="$repo_root/autograph_data/zinc_tokens.jsonl"
        mkdir -p "$(dirname "$output_path")"
        cmd=(python "$repo_root/zinc_tokenization.py" --root "$data_root" --tokenizer autograph --output "$output_path")
        ;;
    *)
        echo "Unknown method: $method" >&2
        echo "method: graph-token | autograph" >&2
        exit 1
        ;;
esac

echo "Running: ${cmd[*]}"
"${cmd[@]}"
