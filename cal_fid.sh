torchrun --standalone --nproc_per_node=4  meanflow_metric.py  --cond=False --metrics='fid50k_full' \
    --network="/path/to/your/network-snapshot.pkl" \
    --data='../data/cifar10-32x32.zip'