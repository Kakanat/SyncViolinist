apt update && apt install -y ffmpeg
python final_evaluation.py TVCG --visualize  --o ./results/TVCG_test --gpu 0 --oracle
python final_evaluation.py TGM2B_fullbody_rhoff --visualize  --o ./results/TGM2B_test --gpu 0 --oracle 
python final_evaluation.py A2BD --visualize  --o ./results/A2BD_test --gpu 0 --oracle
python final_evaluation.py hiroki_tune5 --visualize  --o ./results/hiroki_tune5_test --gpu 0
