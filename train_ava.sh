# Train YOWOv2 on AVA dataset
python train_custom.py \
        --cuda \
        -d ava_v2.2 \
        -v yowo_v2_tiny \
        --root /nvme0/dev/ \
        --num_workers 4 \
        --eval_epoch 1 \
        --eval \
        --eval_first \
        --max_epoch 9 \
        --lr_epoch 3 4 5 6 \
        -lr 0.0001 \
        -ldr 0.5 \
        -bs 16 \
        -accu 16 \
        -K 16 \
        --save_dir