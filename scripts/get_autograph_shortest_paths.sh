conda deactivate; conda activate autograph
for division in train test; do
    mkdir -p autograph_sequences/$division
    for alg in path sfn; do
        python autograph_graphml_tokenizer.py "graph_token_outputs/shortest_path_graphs/$alg/$division" "autograph_sequences/$division/$alg.txt"
    done
done
rm -rf autograph_sequences