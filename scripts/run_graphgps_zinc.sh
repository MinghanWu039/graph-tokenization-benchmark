conda deactivate; conda activate graphgps
cd GraphGPS
python main.py --cfg configs/GPS/zinc-GPS-LapPE+RWSE.yaml wandb.use False
cd ..