import torch
from torchvision.ops import box_iou
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
import pytorch_lightning as pl
import config

def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()

class FRCNN(pl.LightningModule):
    def __init__(self, learning_rate=config.LR, num_classes=config.NUM_CLASSES+1, pretrained=True, pretrained_backbone=True,
                    trainable_backbone_layers=0):
        super().__init__()

        self.learning_rate = learning_rate

        model = fasterrcnn_resnet50_fpn(
                                            pretrained=pretrained,
                                            pretrained_backbone=pretrained_backbone,
                                            trainable_backbone_layers=trainable_backbone_layers,
                                        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        head = FastRCNNPredictor(in_features, num_classes)
        model.roi_heads.box_predictor = head
        self.model = model

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):

        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}


    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        self.log('val_iou',avg_iou)
        return {"avg_val_iou": avg_iou, "log": logs}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.1, cooldown=1,
                                                                             verbose=True, min_lr=1e-7),
                     "monitor": 'val_iou'}
        return [opt], [scheduler]

model = FRCNN()

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_iou',
    min_delta=0.001,
    patience=config.EARLY_STOP_PATIENCE,
    verbose=True,
    mode='max'
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_iou',
    mode='max',
    dirpath=f'{config.ROOT}/models/',
    verbose=True,
    save_weights_only=True
)

trainer = pl.Trainer(
            gpus=1,
            auto_select_gpus=True,
            accelerator = 'gpu' if config.USE_GPU else 'cpu',
            accumulate_grad_batches=config.GRAD_ACCUM_BATCH,
            benchmark=True,
            precision=config.PRECISION,
            stochastic_weight_avg=True,  # Better generalization
            min_epochs=5,
            max_epochs=config.EPOCHS,
            progress_bar_refresh_rate=5,
            num_sanity_val_steps=5,
            callbacks=[early_stop_callback, checkpoint_callback],
        )