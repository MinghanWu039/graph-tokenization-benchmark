# Required: graph_token_outputs/shortest_path_labels, graph_token_outputs/shortest_path_graphs

cd graph-token
set -e
set -x
python3 -m venv graphenv
source graphenv/bin/activate
pip install -r requirements.txt

task="cycle_check"
task_dir="tasks"
graphs_dir="../graph_token_outputs/graphs"
for split in {train,test}
do
    python graph_task_generator.py \
        --task="$task" \
        --algorithm=all \
        --task_dir="$task_dir" \
        --graphs_dir="$graphs_dir" \
        --split=$split \
        --random_seed=1234
done
cd ..
deactivate
conda activate graphgps
python cycle_check_label_extractor.py graph-token/tasks/cycle_check graph_token_outputs/cycle_check_labels --split train
python cycle_check_label_extractor.py graph-token/tasks/cycle_check graph_token_outputs/cycle_check_labels --split test
rm -rf graph-token/tasks