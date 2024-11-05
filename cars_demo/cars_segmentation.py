





if __name__ == '__main__':
    import os
    import cv2

    import torch
    import numpy as np
    import albumentations as A
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as BaseDataset
    from torch.optim import lr_scheduler
    import segmentation_models_pytorch as smp
    import pytorch_lightning as pl

    DATA_DIR = './data/CamVid/'

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')


    class Dataset(BaseDataset):
        """CamVid Dataset. Read images, apply augmentation transformations.

        Args:
            images_dir (str): path to images folder
            masks_dir (str): path to segmentation masks folder
            class_values (list): values of classes to extract from segmentation mask
            augmentation (albumentations.Compose): data transfromation pipeline
                (e.g. flip, scale, etc.)

        """

        CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
                   'tree', 'signsymbol', 'fence', 'car',
                   'pedestrian', 'bicyclist', 'unlabelled']

        def __init__(
                self,
                images_dir,
                masks_dir,
                classes=None,
                augmentation=None,
        ):
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

            # convert str names to class values on masks
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

            self.augmentation = augmentation

        def __getitem__(self, i):
            image = cv2.imread(self.images_fps[i])
            # BGR-->RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)

            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)

        def __len__(self):
            return len(self.ids)


    # helper function for data visualization
    def visualize(**images):
        """PLot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            if name == 'image':
                plt.imshow(image.transpose(1, 2, 0))
            else:
                plt.imshow(image)
        plt.show()


    dataset = Dataset(x_train_dir, y_train_dir, classes=['car'])
    # get some sample
    image, mask = dataset[0]
    visualize(image=image, cars_mask=mask.squeeze(), )


    # training set images augmentation
    def get_training_augmentation():
        train_transform = [
            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
            A.RandomCrop(height=320, width=320, always_apply=True),

            A.GaussNoise(p=0.2),
            A.Perspective(p=0.5),

            A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1),
                    A.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return A.Compose(train_transform)


    def get_validation_augmentation():
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            A.PadIfNeeded(384, 480),
        ]
        return A.Compose(test_transform)


    # Visualize resulted augmented images and masks
    augmented_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        classes=['car'],
    )

    # same image with different random transforms
    for i in range(3):
        image, mask = augmented_dataset[3]
        visualize(image=image, mask=mask.squeeze())

    CLASSES = ['car']

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        classes=CLASSES,
    )

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Some training hyperparameters
    EPOCHS = 20
    T_MAX = EPOCHS * len(train_loader)
    OUT_CLASSES = 1


    class CamVidModel(pl.LightningModule):

        def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
            super().__init__()
            self.model = smp.create_model(
                arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
            )
            # preprocessing parameteres for image
            params = smp.encoders.get_preprocessing_params(encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

            # for image segmentation dice loss could be the best first choice
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

            # initialize step metics
            self.training_step_outputs = []
            self.validation_step_outputs = []
            self.test_step_outputs = []

        def forward(self, image):
            # normalize image here
            image = (image - self.mean) / self.std
            mask = self.model(image)
            return mask

        def shared_step(self, batch, stage):
            image, mask = batch

            # Shape of the image should be (batch_size, num_channels, height, width)
            # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
            assert image.ndim == 4

            # Check that image dimensions are divisible by 32,
            # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
            # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
            # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
            # and we will get an error trying to concat these features
            h, w = image.shape[2:]
            assert h % 32 == 0 and w % 32 == 0

            assert mask.ndim == 4

            # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
            assert mask.max() <= 1.0 and mask.min() >= 0

            logits_mask = self.forward(image)

            # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
            loss = self.loss_fn(logits_mask, mask)

            # Lets compute metrics for some threshold
            # first convert mask values to probabilities, then
            # apply thresholding
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            # We will compute IoU metric by two ways
            #   1. dataset-wise
            #   2. image-wise
            # but for now we just compute true positive, false positive, false negative and
            # true negative 'pixels' for each image and class
            # these values will be aggregated in the end of an epoch
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
            return {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

        def shared_epoch_end(self, outputs, stage):
            # aggregate step metics
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])

            # per image IoU means that we first calculate IoU score for each image
            # and then compute mean over these scores
            per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

            # dataset IoU means that we aggregate intersection and union over whole dataset
            # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
            # in this particular case will not be much, however for dataset
            # with "empty" images (images without target class) a large gap could be observed.
            # Empty images influence a lot on per_image_iou and much less on dataset_iou.
            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            metrics = {
                f"{stage}_per_image_iou": per_image_iou,
                f"{stage}_dataset_iou": dataset_iou,
            }

            self.log_dict(metrics, prog_bar=True)

        def training_step(self, batch, batch_idx):
            train_loss_info = self.shared_step(batch, "train")
            # append the metics of each step to the
            self.training_step_outputs.append(train_loss_info)
            return train_loss_info

        def on_train_epoch_end(self):
            self.shared_epoch_end(self.training_step_outputs, "train")
            # empty set output list
            self.training_step_outputs.clear()
            return

        def validation_step(self, batch, batch_idx):
            valid_loss_info = self.shared_step(batch, "valid")
            self.validation_step_outputs.append(valid_loss_info)
            return valid_loss_info

        def on_validation_epoch_end(self):
            self.shared_epoch_end(self.validation_step_outputs, "valid")
            self.validation_step_outputs.clear()
            return

        def test_step(self, batch, batch_idx):
            test_loss_info = self.shared_step(batch, "test")
            self.test_step_outputs.append(test_loss_info)
            return test_loss_info

        def on_test_epoch_end(self):
            self.shared_epoch_end(self.test_step_outputs, "test")
            # empty set output list
            self.test_step_outputs.clear()
            return

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
            return


    #model = CamVidModel("FPN", "resnext50_32x4d", in_channels=3, out_classes=OUT_CLASSES)
    model = CamVidModel("FPN", "mit_b0", in_channels=3, out_classes=OUT_CLASSES)

    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(test_metrics)

    images, masks = next(iter(test_loader))
    with torch.no_grad():
        model.eval()
        logits = model(images)
    pr_masks = logits.sigmoid()
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        # Number of samples visualized
        if idx <= 4:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image.numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.numpy().squeeze())
            plt.title("Ground truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.numpy().squeeze())
            plt.title("Prediction")
            plt.axis("off")
            plt.show()
        else:
            break