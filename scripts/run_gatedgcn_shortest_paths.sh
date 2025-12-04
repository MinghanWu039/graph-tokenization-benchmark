# Assume graph_token_outputs/shortest_path_graphs/ and graph_token_outputs/shortest_path_labels/ exist.
# Ensure GraphGPS/configs/GatedGCN/shortest_paths-GatedGCN.yaml points to their absolute paths via data_src entries.

cd GraphGPS
conda activate graphgps
python main.py --cfg configs/GatedGCN/shortest_paths-GatedGCN.yaml wandb.use=False
cd ..
