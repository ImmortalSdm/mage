PRETRAIN_CHKPT=$1
OUTPUT_DIR=$2
IMAGENET_DIR=$3

python -u -m torch.distributed.launch --nproc_per_node=8 \   # --node_rank=0 --nnodes=4
# --master_addr="${MASTER_SERVER_ADDRESS}" --master_port=12344 \
main_finetune.py \
--batch_size 32 \
--model vit_base_patch16 \
--global_pool \
--finetune ${PRETRAIN_CHKPT} \
--epochs 1000 \
--blr 2.5e-4 --layer_decay 0.65 --interpolation bicubic \
--weight_decay 0.05 --drop_path 0.1 --reprob 0 --mixup 0.8 --cutmix 1.0 \
--output_dir ${OUTPUT_DIR} \
--data_path ${IMAGENET_DIR} \
--dist_eval # --dist_url tcp://${MASTER_SERVER_ADDRESS}:6311
2>&1 | tee out.log