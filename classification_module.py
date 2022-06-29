import torch
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics

import MinkowskiEngine as ME

from optim import RAdam

class ClassificationLitModule(pl.LightningModule):
    def __init__(self, model, epochs=10, lr=5e-3, num_classes=11, sparse=True):
        super().__init__()
        
        self.lr, self.epochs = lr, epochs
        self.num_classes = num_classes
        self.sparse = sparse
        self.all_nnz, self.all_nnumel = 0, 0

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.train_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.val_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.test_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx, mode):
        coords, feats, target = batch

        if self.sparse:
            data = []
            for t in range((coords[-1,1]+1).item()): # nb_steps
                indices_t = coords[:,1]==t
                coords_t = coords[indices_t,:]
                data.append(ME.SparseTensor(feats[indices_t,:].to(self.device),
                                            coords_t[:,(0,2,3)].to(self.device),
                                            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED
                                        )
                        )
        if not self.sparse:
            data = ME.SparseTensor(feats.to(self.device), coords.to(self.device)).dense()[0]

        outputs = self(data)
        loss = nn.functional.cross_entropy(outputs, target)

        # Measure sparsity if testing
        if mode=="test":
            self.process_nz(self.model.get_nz_numel())

        # Metrics computation
        sm_outputs = outputs.softmax(dim=-1)

        acc, acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        confmat = getattr(self, f'{mode}_confmat')

        acc(sm_outputs, target)
        acc_by_class(sm_outputs, target)
        confmat(sm_outputs, target)

        if mode != "test":
            self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=(mode == "train"))

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="test")

    def on_mode_epoch_end(self, mode):
        print()

        mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()
        for i,acc_i in enumerate(acc_by_class):
            self.log(f'{mode}_acc_{i}', acc_i)
        self.log(f'{mode}_acc', acc)

        print(f"{mode} accuracy: {100*acc:.2f}%")
        mode_acc.reset()
        mode_acc_by_class.reset()

        print(f"{mode} confusion matrix:")
        self_confmat = getattr(self, f"{mode}_confmat")
        confmat = self_confmat.compute()
        self.log(f'{mode}_confmat', confmat)
        print(confmat)
        self_confmat.reset()

        if mode=="test":
            print(f"Total sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / self.all_nnumel:.2f}%)")
            self.all_nnz, self.all_nnumel = 0, 0

    def on_train_epoch_end(self):
        self.on_mode_epoch_end(mode="train")
        
    def on_validation_epoch_end(self):
        self.on_mode_epoch_end(mode="val")

    def on_test_epoch_end(self):
        self.on_mode_epoch_end(mode="test")

    def process_nz(self, nz_numel):
        nz, numel = nz_numel
        total_nnz, total_nnumel = 0, 0

        for module, nnz in nz.items():
            if "act" in module:
                nnumel = numel[module]
                if nnumel != 0:
                    total_nnz += nnz
                    total_nnumel += nnumel
        if total_nnumel != 0:
            self.all_nnz += total_nnz
            self.all_nnumel += total_nnumel

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}