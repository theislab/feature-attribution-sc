# pl packages
import argparse
import torch
from pytorch_lightning.utilities.seed import seed_everything

# python packages
import os
import pandas as pd

os.chdir('../')
import sys

root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)

# single cell packages
import scanpy as sc

# own packages
from feature_attribution_sc.models.scgen_models import SCGENCustom


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--feature_importance', type=str,
                        default='/home/icb/till.richter/git/feature-attribution-sc/outputs/baselines/task1_random.csv')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='/home/icb/yuge.ji/projects/feature-attribution-sc/')
    parser.add_argument('--CHECKPOINT_PATH', type=str,
                        default='/home/icb/till.richter/git/feature-attribution-sc/trained_models/scgen/')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    # GET GPU AND ARGS
    if torch.cuda.is_available():
        print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    args = parse_args()

    # GET DATA
    adata = sc.read(f'{args.data_path}/datasets/scgen_norman19.h5ad')

    # FIX SEED FOR REPRODUCIBILITY
    seed_everything(90)

    # GET MODEL
    SCGENCustom.setup_anndata(adata)
    model = SCGENCustom(adata, feature_importance=pd.read_csv(args.feature_importance), threshold=args.threshold)
    # model.feature_importance = pd.read_csv(args.feature_importance)  # load feature importance
    # model.threshold = args.threshold  # set threshold

    # CHECKPOINT CALLBACK
    CHECKPOINT_PATH = args.CHECKPOINT_PATH + args.feature_importance.split('/')[-1].split('.')[
        0] + '_' + str(args.threshold) + '_checkpoints.pt'  # set checkpoint path
    print(f'CHECKPOINT_PATH: {CHECKPOINT_PATH}')

    model.save(CHECKPOINT_PATH, overwrite=True)  # save model to checkpoint

    # TRAIN
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.patience
    )
