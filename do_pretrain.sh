source activate brainmvp
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--dataset "BraTS2021" "BraTS2023_SSA" "BraTS2023_MEN"  "UCSF_upright_sir24" "BrainAtlas_registration_sir24" \
--base_dir "./Dataset/Pre-train/BrainMVP-16k" \
--dst_h 240 \
--dst_w 240 \
--dst_d 155 \
--max_epochs=1500 \
--start_epoch 0 \
--save_interval 100 \
--accumulation_steps 2 \
--lr_schedule 'cosine_anneal' \
--batch_size=3 \
--lr 3e-4 \
--mask_rate 0.8 \
--num_workers 6 \
--template_index flair t1 t1c t2 mra pd dwi adc \
--mask_block_size 8 \
--debug \
--logdir "uniformer_pretrain"