export CUDA_VISIBLE_DEVICES=0,1,2,3
nproc_per_node=4
save_path="exp_think3_1x_bs"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m src.trainer.fsdp_sft_trainer \
    diffusion.time_reweighting=cart \
    data.train_files=data/think3/train.parquet \
    data.val_files=data/think3/train.parquet \
    data.max_length=2048 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.truncation=right \
    optim.lr=2e-6 \
    data.micro_batch_size_per_gpu=4 \
    data.perbatch_cutoff_type=random_with_input_pad \
    data.perbatch_cutoff=True \
    model.partial_pretrain=Dream/dream_ins \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=test_exp \
    trainer.project_name=diff-verl \
    trainer.experiment_name=test_exp \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    # trainer.train_batch_size=768
