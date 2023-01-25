# pl packages
import argparse

# python packages
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

os.chdir("../")
import sys

root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)


# single cell packages

# own packages
from feature_attribution_sc.data.get_data import get_hlca_data, get_scgen_data
from feature_attribution_sc.inference.test_scgen import test_scgen
from feature_attribution_sc.models.scgen_models import SCGENCustom


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="scgen")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--feature_importance", type=str, default="random")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--mask_rate", type=float, default=0.0)
    return parser.parse_args()


def write_results(result, model_name, feature_importance, threshold, mode):
    """
    Append the experiment to a csv output file from which the graph is plotted

    :param result: performance of this experiment, becomes y-axis of plot
    :param model_name: type of experiment, e.g., scgen, scvi
    :param feature_importance: feature importance method, e.g., random
    :param threshold: threshold of this, becomes x-axis of plot
    :param mode: default test, this can become train if the model is retrained
    :return:
    """
    # get paths for experiment
    csv_file = model_name + "_" + feature_importance + "_" + mode + ".csv"
    RESULT_PATH = os.path.join("/home/icb/till.richter/git/feature-attribution-sc/outputs/performance", csv_file)

    # write result as pandas df
    new_result = {"threshold": [threshold], "performance": [result]}
    new_result_df = pd.DataFrame(data=new_result)

    if os.path.exists(RESULT_PATH):  # there are already some results
        old_result_df = pd.read_csv(RESULT_PATH)
        result_df = pd.concat([old_result_df, new_result_df])
        result_df.to_csv(RESULT_PATH)
    else:
        new_result_df.to_csv(RESULT_PATH)
    print("Saved results at", RESULT_PATH)


if __name__ == "__main__":
    # GET GPU AND ARGS
    if torch.cuda.is_available():
        print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    args = parse_args()

    # GET DATA
    if args.model_name == "scanvi":  # scANVI - HLCA
        dataloader, adata = get_hlca_data(batch_size=args.batch_size)  # this is 1 dataloader
    elif args.model_name == "scgen":  # scGen - Norman
        dataloader, adata = get_scgen_data(batch_size=args.batch_size)  # these are several dataloaders
        datasets_ls = [
            "norman19_model1_shuffled",
            "norman19_model1_random",
            "norman19_model0_shuffled",
            "norman19_model4_shuffled",
            "norman19_model4_random",
            "norman19_model2_random",
            "norman19_model2_shuffled",
            "norman19_model0_random.pt",
            "norman19_model3_shuffled",
            "norman19_model0_random",
            "norman19_model3_random",
        ]  # might be useful

    # FIX SEED FOR REPRODUCIBILITY
    seed_everything(90)

    # CHECKPOINT HANDLING  - do we need checkpoints?
    CHECKPOINT_PATH = "/lustre/scratch/users/yuge.ji/fasc/"  # TO DO - more beautiful directory structure
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
    if os.path.exists(os.path.join(CHECKPOINT_PATH, "best_checkpoint.ckpt")):
        resume_from_checkpoint = os.path.join(CHECKPOINT_PATH, "best_checkpoint.ckpt")
    else:
        resume_from_checkpoint = None

    # TRAIN / TEST
    # TO DO: INCLUDE MASKING OF GENES BEFOREHAND AND GIVE ESTIM THAT STUFF
    if args.model_name == "scanvi":
        print("Using scaNVI trained model, not implemented yet")
    elif args.model_name == "scgen":
        print("Using scGEN trained model")
        SCGENCustom.setup_anndata(adata)
        # estim = SCGENCustom(adata, args.feature_importance, args.threshold)
        MODEL_DIR = "/home/icb/yuge.ji/projects/feature-attribution-sc"

    if args.mode == "train":
        estim.train(  # noqa: F821
            enable_checkpointing=True,
            callbacks=[
                checkpoint_callback,
                best_checkpoint_callback,
                pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
                TQDMProgressBar(refresh_rate=100),
            ],
            # resume_from_checkpoint=resume_from_checkpoint,
            check_val_every_n_epoch=1,
            limit_train_batches=int(len(id_datamodule.idx_train) / args.batch_size),  # 1000,  # noqa: F821
            limit_val_batches=int(len(id_datamodule.idx_val) / args.batch_size),  # 1000,  # noqa: F821
            logger=TensorBoardLogger(CHECKPOINT_PATH),  # TO DO: CHECK THE LOGGING IN SCVI
            gpus=1,
            num_sanity_val_steps=0,
            log_every_n_steps=100,
            # auto_lr_find='learning_rate',
            auto_lr_find=False,
            progress_bar_refresh_rate=100,
        )
    elif args.mode == "test":
        if args.model_name == "scanvi":  # TBD
            print("TO BE DONE")
        elif args.model_name == "scgen":
            estim = {}
            for file in os.listdir(f"{MODEL_DIR}/models"):
                if "scgen" in file:
                    SCGENCustom.__name__ = "SCGEN"
                    estim["_".join(file.split("_")[1:])] = SCGENCustom.load(f"{MODEL_DIR}/models/{file}", adata=adata)
                    SCGENCustom.__name__ = "SCGENCustom"
                    estim["_".join(file.split("_")[1:])].module.feature_attribution = args.feature_importance
                    estim["_".join(file.split("_")[1:])].module.threshold = args.threshold
                    r2_ground_truth, r2_no_treatment = test_scgen(
                        adata=adata, model=estim["_".join(file.split("_")[1:])], example_pert="KLF1"
                    )
                    print(
                        "MODEL {}\n\nR2 GROUND TRUTH:\n{}\nR2 NO TREATMENT:\n{}\n".format(
                            file.split("_")[1:], r2_ground_truth, r2_no_treatment
                        )
                    )
                    write_results(
                        result=r2_ground_truth,
                        model_name=file,
                        feature_importance=args.feature_importance,
                        threshold=args.threshold,
                        mode=args.mode,
                    )
