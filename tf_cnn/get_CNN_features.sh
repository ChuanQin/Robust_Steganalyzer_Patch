#!/bin/bash
#SBATCH -p gpu3
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem 20G
#SBATCH --gres gpu:1
#SBATCH -o extract_deep_features_%j.out

date
srun --pty python3 -m pdb get_cnn_fea.py \
--model_type SRNet \
--img_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
--prob_dir /data-x/g12/qinchuan/data/mod_probs/BOSS_BOWS/imresize_256/cover/SUNIWARD/payload_0.4/ \
--feature_path /data-x/g12/qinchuan/data/features/patch/BOSS_BOWS/imresize_256/SRNet/SUNIWARD/cover.mat \
--load_path /public/qinchuan/deep-learning/SRNet/UNIWARD/payload_0.4/Model_670000.ckpt \
--batch_size 50 \
--num_workers 8
date


# # conda activate tensorflow_1.14
# date
# srun python3 -u get_cnn_fea.py \
# --model_type SRNet \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.3/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/SRNet/UNIWARD/payload_0.3/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.3/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.3/S-UNIWARD_0.3.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.3/ADV-EMB_SRNet_UNIWARD_payload_0.3.mat \
# --load_path /public/qinchuan/deep-learning/SRNet/UNIWARD/payload_0.3/Model_120000.ckpt \
# --batch_size 50 \
# --num_workers 8
# date

# srun python3 -u get_cnn_fea.py \
# --model_type SRNet \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.2/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/SRNet/UNIWARD/payload_0.2/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.2/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.2/S-UNIWARD_0.2.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.2/ADV-EMB_SRNet_UNIWARD_payload_0.2.mat \
# --load_path /public/qinchuan/deep-learning/SRNet/UNIWARD/payload_0.2/Model_160000.ckpt \
# --batch_size 50 \
# --num_workers 8
# date

# srun python3 -u get_cnn_fea.py \
# --model_type SRNet \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.1/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/SRNet/UNIWARD/payload_0.1/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.1/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.1/S-UNIWARD_0.1.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.1/ADV-EMB_SRNet_UNIWARD_payload_0.1.mat \
# --load_path /public/qinchuan/deep-learning/SRNet/UNIWARD/payload_0.1/Model_140000.ckpt \
# --batch_size 50 \
# --num_workers 8
# date

# srun python3 -u get_cnn_fea.py \
# --model_type SRNet \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.5/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/SRNet/UNIWARD/payload_0.5/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.5/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.5/S-UNIWARD_0.5.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.5/ADV-EMB_SRNet_UNIWARD_payload_0.5.mat \
# --load_path /public/qinchuan/deep-learning/SRNet/UNIWARD/payload_0.5/Model_210000.ckpt \
# --batch_size 50 \
# --num_workers 8
# date
########################################### YeNet spatial ###########################################
# date
# srun python3 -u get_cnn_fea.py \
# --model_type YeNet \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.5/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/YeNet/UNIWARD/payload_0.5/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.5/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.5/S-UNIWARD_0.5.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.5/ADV-EMB_SRNet_UNIWARD_payload_0.5.mat \
# --load_path /public/qinchuan/deep-learning/YeNet/UNIWARD/payload_0.5/Model_780000.ckpt \
# --batch_size 50 \
# --num_workers 8
# date
# srun python3 -u get_cnn_fea.py \
# --model_type YeNet \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.4/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/YeNet/UNIWARD/payload_0.4/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.4/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.4/S-UNIWARD_0.4.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.4/ADV-EMB_SRNet_UNIWARD_payload_0.4.mat \
# --load_path /public/qinchuan/deep-learning/YeNet/UNIWARD/payload_0.4/Model_680000.ckpt \
# --batch_size 50 \
# --num_workers 8
# date
# srun python3 -u get_cnn_fea.py \
# --model_type YeNet \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.3/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/YeNet/UNIWARD/payload_0.3/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.3/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.3/S-UNIWARD_0.3.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.3/ADV-EMB_SRNet_UNIWARD_payload_0.3.mat \
# --load_path /public/qinchuan/deep-learning/YeNet/UNIWARD/payload_0.3/Model_840000.ckpt \
# --batch_size 50 \
# --num_workers 8
# date
# srun python3 -u get_cnn_fea.py \
# --model_type YeNet \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.2/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/YeNet/UNIWARD/payload_0.2/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.2/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.2/S-UNIWARD_0.2.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.2/ADV-EMB_SRNet_UNIWARD_payload_0.2.mat \
# --load_path /public/qinchuan/deep-learning/YeNet/UNIWARD/payload_0.2/Model_830000.ckpt \
# --batch_size 50 \
# --num_workers 8
# date
# srun python3 -u get_cnn_fea.py \
# --model_type YeNet \
# --cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
# --stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.1/ \
# --adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/YeNet/UNIWARD/payload_0.1/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.1/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.1/S-UNIWARD_0.1.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/YeNet/S-UNIWARD/payload_0.1/ADV-EMB_SRNet_UNIWARD_payload_0.1.mat \
# --load_path /public/qinchuan/deep-learning/YeNet/UNIWARD/payload_0.1/Model_160000.ckpt \
# --batch_size 50 \
# --num_workers 8
# date

########################################### SRNet JPEG ###########################################
date
for payload in 0.1 0.2 0.3 0.4 0.5; do
    srun python3 -u get_cnn_fea.py \
    --model_type SRNet \
    --cover_dir /data-x/g15/qinchuan/JPEG/imresize-256/QF75/cover/ \
    --stego_dir /data-x/g15/qinchuan/JPEG/imresize-256/QF75/J-UNIWARD/payload_$payload\/ \
    --adv_dir /data-x/g15/qinchuan/JPEG/imresize-256/QF75/ADV-EMB/SRNet/J-UNIWARD/payload__$payload\/ \
    --cover_feature_path ~/data/feature/BOSS_BOWS2/JPEG/QF75/256/SRNet/J-UNIWARD/payload_$payload\/cover.mat \
    --stego_feature_path ~/data/feature/BOSS_BOWS2/JPEG/QF75/256/SRNet/J-UNIWARD/payload_$payload\/J-UNIWARD_$payload\.mat \
    --adv_feature_path ~/data/feature/BOSS_BOWS2/JPEG/QF75/256/SRNet/J-UNIWARD/payload_$payload\/ADV-EMB_SRNet_UNIWARD_payload_$payload\.mat \
    --load_path /public/qinchuan/deep-learning/SRNet/QF75/J-UNIWARD/payload_$payload \
    --batch_size 50 \
    --num_workers 8
    date
done
# srun --pty python3 -m pdb get_cnn_fea.py \
# --model_type SRNet \
# --cover_dir /data-x/g15/qinchuan/JPEG/imresize-256/QF75/cover/ \
# --stego_dir /data-x/g15/qinchuan/JPEG/imresize-256/QF75/J-UNIWARD/payload_0.4/ \
# --adv_dir /data-x/g15/qinchuan/JPEG/imresize-256/QF75/ADV-EMB/SRNet/J-UNIWARD/payload_0.4/ \
# --cover_feature_path ~/data/feature/BOSS_BOWS2/JPEG/QF75/256/SRNet/J-UNIWARD/payload_0.4/cover.mat \
# --stego_feature_path ~/data/feature/BOSS_BOWS2/JPEG/QF75/256/SRNet/J-UNIWARD/payload_0.4/J-UNIWARD_0.4.mat \
# --adv_feature_path ~/data/feature/BOSS_BOWS2/JPEG/QF75/256/SRNet/J-UNIWARD/payload_0.4/ADV-EMB_SRNet_UNIWARD_payload_0.4.mat \
# --load_path /public/qinchuan/deep-learning/SRNet/QF75/J-UNIWARD/payload_0.4 \
# --batch_size 50 \
# --num_workers 8