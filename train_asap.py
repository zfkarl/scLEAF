#!/usr/bin/env python

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping,ModelCheckpoint


from scclip.dataset import CellTextDataModule
from scclip.clip import CLIPModel, Classifier
from scclip.vit import ViTConfig
from scclip.callback import Monitor
from scclip.config import get_model_config
from scclip.logger import create_logger

import os
import argparse
import torch
import numpy as np


from pathlib import Path

HOME = Path.home()
print("Start", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clip")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint", type=str, default=None)
    # Dataset
    parser.add_argument("--rna_data_path", type=str, default='/data2/zeyu/zeyu1/dataset/sc-data/data/citeseq_control_rna.npz')
    parser.add_argument("--rna_text_path", type=str, default='/data2/zeyu/zeyu1/dataset/sc-data/data/text_embeddings_cite.npy')
    parser.add_argument("--rna_label_path", type=str, default='/data2/zeyu/zeyu1/dataset/sc-data/data/citeseq_control_cellTypes.txt')
    parser.add_argument("--atac_data_path", type=str, default='/data2/zeyu/zeyu1/dataset/sc-data/data/asapseq_control_atac.npz')
    parser.add_argument("--atac_text_path", type=str, default='/data2/zeyu/zeyu1/dataset/sc-data/data/text_embeddings_asap.npy')
    parser.add_argument("--atac_label_path", type=str, default='/data2/zeyu/zeyu1/dataset/sc-data/data/asapseq_control_cellTypes_v2.txt')
    parser.add_argument("--gene_emb_path", type=str, default='/data2/zeyu/zeyu1/dataset/sc-data/data/cite-asap-gene-emb.npy')
    # DataModule
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="cite-asap")
    parser.add_argument("--experiment", action=argparse.BooleanOptionalAction, default=True)
    # Module
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--warmup_steps", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--use_imputed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument( "--requires_grad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--version", type=str, default="asap")
    parser.add_argument("--pretrain_epochs", type=int, default=500)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--logit_scale", type=float, default=1)  # 2.6592)
    parser.add_argument("--num_patches", type=int, default=128)
    parser.add_argument( "--early_stop", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.checkpoint is None:
        
        atac_data = CellTextDataModule(args.batch_size,args.atac_data_path, args.atac_text_path, args.atac_label_path, args.gene_emb_path)

        

        model_config = get_model_config("small")
        cell_config = ViTConfig(
            **{
                "modality": "cell",
                "num_patches": args.num_patches,
                "feature_size": atac_data.dataset.input_size,
                "text_emb_size": atac_data.dataset.text_emb_size,
                "gene_emb_size": atac_data.dataset.gene_emb_size,
                "attention_probs_dropout_prob": args.dropout,
                "hidden_dropout_prob": args.dropout,
                **model_config,
            }
        )

        model = CLIPModel(
            args,
            cell_config=cell_config,
        )


        # out_dir
        if args.experiment:
            args.default_root_dir = f"results/{args.data_dir}/{args.logit_scale}_{args.requires_grad}_{args.pretrain_epochs}_{args.lr}_{args.version}"
        else:
            args.default_root_dir = (
                f"results/{args.data_dir}/{args.logit_scale}_{args.pretrain_epochs}"
            )
        # os.makedirs(args.default_root_dir, exist_ok=True)
        print("default_root_dir:", args.default_root_dir, flush=True)

        # trainer
        logger = TensorBoardLogger(
            save_dir=args.default_root_dir, default_hp_metric=False, version=""
        )

 
        checkpoint_callback = ModelCheckpoint(
            monitor='loss/val',  
            save_last = args.default_root_dir + "/lightning_logs/checkpoints",
            dirpath= args.default_root_dir + "/lightning_logs/checkpoints",  # 模型保存路径
            filename=f'best-pretrain',  
            save_top_k=1,  
            mode='min'  
        )

        trainer = Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            devices=1,
            gradient_clip_val=5,
            num_sanity_val_steps=0,
            logger=logger,
            max_epochs=args.pretrain_epochs,
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps = 1
        )

        # pretrain
        print('Pretrain on ATAC Data!')
        trainer.fit(model, atac_data)
        print('Finish Pretrain on ATAC Data!')
        
        
    else:
        
        model = CLIPModel.load_from_checkpoint(args.checkpoint) 
        print("normalize", args.normalize, flush=True)
        model.config.normalize = args.normalize
        args.default_root_dir = args.checkpoint.split("lightning_logs/")[0]
        
        atac_data = CellTextDataModule(args.batch_size,args.atac_data_path, args.atac_text_path, args.atac_label_path, args.gene_emb_path)

        
        model_config = get_model_config("small")
        cell_config = ViTConfig(
            **{
                "modality": "cell",
                "num_patches": args.num_patches,
                "feature_size": atac_data.dataset.input_size,
                "text_emb_size": atac_data.dataset.text_emb_size,
                "gene_emb_size": atac_data.dataset.gene_emb_size,
                "attention_probs_dropout_prob": args.dropout,
                "hidden_dropout_prob": args.dropout,
                **model_config,
            }
        )

        logger = TensorBoardLogger(
                save_dir=args.default_root_dir, default_hp_metric=False, version=""
            )

       
        atac_classes = int(len(np.unique(atac_data.dataset.labels)))
        #print('atac_classes: ', atac_classes)
        atac_cls_model = Classifier(atac_classes, args, cell_config=cell_config )

        ## can load weights by: 
        atac_cls_model.cell_model.load_state_dict(model.cell_model.state_dict())
        atac_cls_model.cell_projection.load_state_dict(model.cell_projection.state_dict())
        atac_cls_model.text_projection.load_state_dict(model.text_projection.state_dict())  
        
        atac_cls_callback = [ModelCheckpoint(
            monitor='validation loss',  
            dirpath= args.default_root_dir + "/lightning_logs/checkpoints",  # 模型保存路径
            filename=f'best-finetune-atac',  
            save_top_k=1,  
            mode='min'  
        )]
        if args.early_stop:
            atac_cls_callback.append(EarlyStopping(monitor="validation loss", patience=10, mode ='min'))

        atac_trainer = Trainer(
            callbacks=atac_cls_callback,
            accelerator="gpu",
            devices=1,
            gradient_clip_val=5,
            num_sanity_val_steps=0,
            logger=logger,
            max_epochs=args.finetune_epochs,
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps = 1
        )

        print('Finetune on ATAC Data!')
        atac_trainer.fit(atac_cls_model, atac_data)
        print('Finish Finetune on ATAC Data!')

        print('Test on ATAC Data!')
        atac_cls_model = Classifier.load_from_checkpoint(args.default_root_dir + "/lightning_logs/checkpoints/best-finetune-atac.ckpt") 
        atac_trainer.test(atac_cls_model, atac_data)
        print('Finish Test on ATAC Data!')