# Required: graph_token_outputs/shortest_path_labels, graph_token_outputs/shortest_path_graphs

cd graph-token
set -e
set -x
python3 -m venv graphenv
source graphenv/bin/activate
pip install -r requirements.txt

task="cycle_check"
task_dir="tasks"
graphs_dir="../graph_token_outputs/shortest_path_graphs"
for algorithm in "path" "sfn"
do
    for split in {train,test}
    do
        python graph_task_generator.py \
            --task="$task" \
            --algorithm="$algorithm" \
            --task_dir="$task_dir" \
            --graphs_dir="$graphs_dir" \
            --split=$split \
            --random_seed=1234
    done
done
cd ..
deactivate
conda activate graphgps
python graph_token_shortest_paths.py graph-token/tasks/cycle_check graph_token_outputs/shortest_path_labels graph_token_outputs/shortest_path_seqs
rm -rf graph-token/tasks
