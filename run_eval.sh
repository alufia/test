#!/bin/bash
#SBATCH -p defq                         
#SBATCH --job-name=qwen2_eval
#SBATCH --nodes=1                      
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8         
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --output=log/%x_%j.log        
#SBATCH --error=log/%x_%j.err          

# #SBATCH -w dgx-[01-40]

# 작업 디렉토리
WORK_DIR=/gpfs/user/wsko/UltraEval-Audio

srun -l \
     --container-image=/gpfs/user/wsko/docker/torch24+latest.sqsh \
     --container-workdir="${WORK_DIR}" \
     --container-mounts=/gpfs/user/wsko/UltraEval-Audio:/gpfs/user/wsko/UltraEval-Audio \
     --no-container-mount-home \
     --output=log/%x_srun_%j.log \
     --error=log/%x_srun_%j.err \
     bash -c "
        pip install -r requirements.txt
        pip install -r requirments-offline-model.txt
        python audio_evals/main.py --dataset llama-questions-s2t --model qwen2-audio-chat
        python audio_evals/main.py --dataset speech-web-questions --model qwen2-audio-chat
     "
