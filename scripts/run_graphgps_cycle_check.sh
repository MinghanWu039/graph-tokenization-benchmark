# Assume graph_token_outputs/graphs/ and graph_token_outputs/cycle_check_labels/ exist.
# Assume the absolute paths of these directories are set in GraphGPS/configs/GPS/simple-cycle-detection-GPS.yaml > data_src.

conda deactivate; conda activate graphgps
cd GraphGPS
python main.py --cfg configs/GPS/simple-cycle-detection-GPS.yaml wandb.use False
cd ..