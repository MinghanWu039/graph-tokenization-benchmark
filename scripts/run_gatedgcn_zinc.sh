conda deactivate; conda activate graphgps
cd GraphGPS
python main.py --cfg configs/GatedGCN/zinc-GatedGCN.yaml wandb.use False
cd ..