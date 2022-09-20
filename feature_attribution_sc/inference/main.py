# pl packages
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# python packages
import os
os.chdir('../')
import sys
root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)
import importlib

# single cell packages
import scvi
import scanpy as sc
import scgen

# own packages
from feature_attribution_sc.data.get_data import get_scgen_data, get_hlca_data
from feature_attribution_sc.models.get_model import get_scgen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='scanvi')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--feature_importance', type=str, default='None')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--mode', type=str, default='test')
    return parser.parse_args()


if __name__ == '__main__':
    # GET GPU AND ARGS
    if torch.cuda.is_available():
        print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    args = parse_args()

    # GET DATA
    if args.model_name == 'scanvi':  # scANVI - HLCA
        dataloader, adata = get_hlca_data(batch_size=args.batch_size)  # this is 1 dataloader
    elif args.model_name == 'scgen':  # scGen - Norman
        dataloader, adata = get_scgen_data(batch_size=args.batch_size)  # these are several dataloaders
        datasets_ls = ['norman19_model1_shuffled', 'norman19_model1_random', 'norman19_model0_shuffled',
                       'norman19_model4_shuffled', 'norman19_model4_random', 'norman19_model2_random',
                       'norman19_model2_shuffled', 'norman19_model0_random.pt', 'norman19_model3_shuffled',
                       'norman19_model0_random', 'norman19_model3_random']  # might be useful

    # FIX SEED FOR REPRODUCIBILITY
    seed_everything(90)

    # CHECKPOINT HANDLING  - do we need checkpoints?
    CHECKPOINT_PATH = '/lustre/scratch/users/yuge.ji/fasc/' # TO DO - more beautiful directory structure
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        dirpath=CHECKPOINT_PATH,
        filename="hack-features-{epoch:02d}-{val_loss:.2f}",
    )

    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=CHECKPOINT_PATH,
        filename="best_checkpoint",
    )

    # TO DO: INCLUDE PRETRAINED SCGEN MODELS
    if os.path.exists(os.path.join(CHECKPOINT_PATH, 'best_checkpoint.ckpt')):
        resume_from_checkpoint = os.path.join(CHECKPOINT_PATH, 'best_checkpoint.ckpt')
    else:
        resume_from_checkpoint = None


    # TRAIN / TEST
    # TO DO: INCLUDE MASKING OF GENES BEFOREHAND AND GIVE ESTIM THAT STUFF
    if args.model_name == 'scanvi':
        print('Using scaNVI trained model, not implemented yet')
    elif args.model_name == 'scgen':
        print('Using scGEN trained model')
        estim = SCGENCustom(adata, args.feature_importance, args.threshold)
        if args.mode == 'train':
            estim.train(
                enable_checkpointing=True,
                callbacks=[checkpoint_callback,
                           best_checkpoint_callback,
                           pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=args.patience),
                           TQDMProgressBar(refresh_rate=100)],
                # resume_from_checkpoint=resume_from_checkpoint,
                check_val_every_n_epoch=1,
                limit_train_batches=int(len(id_datamodule.idx_train) / args.batch_size),  # 1000,
                limit_val_batches=int(len(id_datamodule.idx_val) / args.batch_size),  # 1000,
                logger=TensorBoardLogger(CHECKPOINT_PATH),  # TO DO: CHECK THE LOGGING IN SCVI
                gpus=1,
                num_sanity_val_steps=0,
                log_every_n_steps=100,
                # auto_lr_find='learning_rate',
                auto_lr_find=False,
                progress_bar_refresh_rate=100
            )
        elif args.mode == 'test':
            estim.test(
                enable_checkpointing=True,
                callbacks=[checkpoint_callback,
                           best_checkpoint_callback,
                           pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=args.patience),
                           TQDMProgressBar(refresh_rate=100)],
                # resume_from_checkpoint=resume_from_checkpoint,
                check_val_every_n_epoch=1,
                limit_train_batches=int(len(id_datamodule.idx_train) / args.batch_size),  # 1000,
                limit_val_batches=int(len(id_datamodule.idx_val) / args.batch_size),  # 1000,
                logger=TensorBoardLogger(CHECKPOINT_PATH),
                gpus=1,
                num_sanity_val_steps=0,
                log_every_n_steps=100,
                # auto_lr_find='learning_rate',
                auto_lr_find=False,
                progress_bar_refresh_rate=100
            )


