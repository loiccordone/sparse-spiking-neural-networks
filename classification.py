from os.path import join
import sys
import argparse

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import MinkowskiEngine as ME

from gesture_dataset import SparseDvsGestureDataset
from models.snn import DenseSNN
from models.sparse_snn import SparseSNN
from classification_module import ClassificationLitModule

def main():
    parser = argparse.ArgumentParser(description='Classify event dataset')
    parser.add_argument('-device', default=0, type=int, help='device')
    parser.add_argument('-precision', default=16, type=int, help='whether to use AMP {16, 32, 64}')

    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-sample_size', default=1500000, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=150, type=int, help='simulating time-steps')
    parser.add_argument('-image_shape', default=(128,128), type=tuple, help='spatial resolution of events')

    parser.add_argument('-dataset', default='dvsg', type=str, help='dataset used {dvsg}')
    parser.add_argument('-path', default='DvsGesture', type=str, help='path to dataset')

    parser.add_argument('-model', default='sparse-snn', type=str, help='model used {snn, sparse-snn}')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-lr', default=1e-2, type=float, help='learning rate used')
    parser.add_argument('-epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-test', action='store_true', help='whether to test the model')

    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')

    args = parser.parse_args()
    print(args)

    if args.dataset == "dvsg":
        dataset = SparseDvsGestureDataset
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

    train_dataset = dataset(args, mode="train")
    test_dataset = dataset(args, mode="test")

    train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=0, collate_fn=ME.utils.batch_sparse_collate, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.b, num_workers=0, collate_fn=ME.utils.batch_sparse_collate)

    if args.model == 'snn':
        model = DenseSNN()
        sparse = False
    elif args.model == 'sparse-snn':
        model = SparseSNN()
        sparse = True
    else:
        sys.exit(f"{args.model} is not a supported model.")
        
    module = ClassificationLitModule(model, epochs=args.epochs, lr=args.lr, sparse=sparse)

    # LOAD PRETRAINED MODEL
    if args.pretrained is not None:
        ckpt_path = join("pretrained", join(args.model, args.pretrained))
        module = module.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)

    callbacks=[]
    if args.save_ckpt:
        ckpt_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=f"ckpt-{args.dataset}/",
            filename=f"{args.dataset}" + "-{epoch:02d}-{val_acc:.4f}",
            save_top_k=3,
            mode='max',
        )
        callbacks.append(ckpt_callback)

    trainer = pl.Trainer(
        gpus=[args.device], gradient_clip_val=1., max_epochs=args.epochs,
        limit_train_batches=1., limit_val_batches=1.,
        check_val_every_n_epoch=1,
        deterministic=False,
        precision=args.precision,
        callbacks=callbacks,
    )

    if args.train:
        trainer.fit(module, train_dataloader, test_dataloader)
    if args.test:
        trainer.test(module, test_dataloader)

if __name__ == '__main__':
    main()