# Evaluation on LSDIR_DIV2K_valid datasets:
CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir ../ \
    --save_dir ../results \
    --model_id 7

# When Test datasets are included 
# CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir ../ \
#     --save_dir ../results \
#     --include_test \
#     --model_id 0
