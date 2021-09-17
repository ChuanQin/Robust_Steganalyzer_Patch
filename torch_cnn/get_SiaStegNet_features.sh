#!/bin/bash
#SBATCH -p gpu3
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem 40G
#SBATCH --gres gpu:1
#SBATCH -o get_SiaStegNet_features_%j.out

date
for payload in 0.1 0.2 0.3 0.4 0.5; do
    srun singularity exec -B /data-x/g15/ --nv /app/singularity/deepo/20190624.sif \
    python3 -u get_torch_cnn_features.py \
    --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
    --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_$payload\/ \
    --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB-Pytorch/SiaStegNet/SUNIWARD/payload_$payload\/ \
    --ckpt_dir /public/qinchuan/deep-learning/SiaStegNet/SiaStegNet_BOSS_BOWS_SUNIWARD_payload_$payload\.pth \
    --cover_feature_path /public/qinchuan/data/feature/BOSS_BOWS2/spatial/256/SiaStegNet/S-UNIWARD/payload_$payload\/cover.mat \
    --stego_feature_path /public/qinchuan/data/feature/BOSS_BOWS2/spatial/256/SiaStegNet/S-UNIWARD/payload_$payload\/S-UNIWARD_$payload\.mat \
    --adv_feature_path /public/qinchuan/data/feature/BOSS_BOWS2/spatial/256/SiaStegNet/S-UNIWARD/payload_$payload\/ADV-EMB-Pytorch_SiaStegNet_SUNIWARD_payload_$payload\.mat \
    --batch_size 50 \
    --num_workers 16
done
# python3 -m pdb get_torch_cnn_features.py \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.4/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB-Pytorch/SiaStegNet/SUNIWARD/payload_0.4/ \
# --ckpt_dir /public/qinchuan/deep-learning/SiaStegNet/SiaStegNet_BOSS_BOWS_SUNIWARD_payload_0.4.pth \
# --cover_feature_path /public/qinchuan/data/feature/BOSS_BOWS2/spatial/256/SiaStegNet/S-UNIWARD/payload_0.4/cover.mat \
# --stego_feature_path /public/qinchuan/data/feature/BOSS_BOWS2/spatial/256/SiaStegNet/S-UNIWARD/payload_0.4/S-UNIWARD_0.4.mat \
# --adv_feature_path /public/qinchuan/data/feature/BOSS_BOWS2/spatial/256/SiaStegNet/S-UNIWARD/payload_0.4/ADV-EMB-Pytorch_SiaStegNet_SUNIWARD_payload_0.4.mat \
# --batch_size 50 \
# --num_workers 8