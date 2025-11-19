conda deactivate; conda activate autograph
for division in train test; do
    mkdir -p autograph_sequences/$division
    for alg in ba complete er path sbm sfn star; do
        python batch_graphml_tokenizer.py "graph_token_outputs/graphs/$alg/$division" "autograph_sequences/$division/$alg.txt"
    done

    python combine_sequences_labels.py --batch --sequences-dir "autograph_sequences/$division" --labels-dir \
       "graph_token_outputs/cycle_check_labels/$division" --output-file "data/cycle_check/autograph/$division.txt"
done
rm -rf autograph_sequences