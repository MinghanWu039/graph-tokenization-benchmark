# Assume graph_token_outputs/shortest_path_graphs/ and graph_token_outputs/shortest_path_labels/ exist.
# Ensure GraphGPS/configs/GPS/shortest_paths-detection-GPS.yaml points to their absolute paths via data_src entries.

cd GraphGPS
conda activate graphgps
python main.py --cfg configs/GPS/shortest_paths-detection-GPS.yaml wandb.use=False
cd ..
