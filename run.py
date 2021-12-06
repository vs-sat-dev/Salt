import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image

from dataset import SaltDataset
from model import Unet


transform_train = A.Compose([
    A.Rotate(limit=0.45, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
])

transform_test = A.Compose([
    ToTensorV2()
])


# Save batch_size images to preds directory
def save_images(images, masks, preds):
    for i in range(images.shape[0]):
        image = images[i]
        image = image.cpu()
        save_image(image, f'preds/image{i + 1}.png')

        mask = masks[i]
        mask = mask.cpu()
        save_image(mask, f'preds/mask{i + 1}.png')

        pred = preds[i]
        pred = pred.cpu()
        save_image(pred, f'preds/pred{i + 1}.png')


# Check accuracy on the testing dataset
def check_accuracy(model, loader, save_preds=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        num_pixels = 0
        correct_pixels = 0
        dice_score = 0
        model.eval()

        for img, msk in loader:
            img = img.to(device)
            msk = msk.unsqueeze(1).to(device)

            preds = model(img)

            if save_preds:
                save_images(img, msk, preds)

            preds = (preds > 0.5).float()
            num_pixels += torch.numel(preds)
            correct_pixels += (preds == msk).float().sum()
            dice_score += (2 * (preds * msk).sum()) / ((preds + msk).sum() + 1e-8)
        model.train()

    return (correct_pixels / num_pixels), dice_score


def run(images_train, masks_train, images_test, masks_test, images_val, masks_val):

    epochs = 100
    batch_size = 32
    learning_rate = 0.0001
    weight_decay = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = False
    if device == 'cuda':
        pin_memory = True

    model = Unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_dataset = SaltDataset(images_train, masks_train, transform=transform_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin_memory)

    test_dataset = SaltDataset(images_test, masks_test, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=pin_memory)

    val_dataset = SaltDataset(images_val, masks_val, transform=transform_test)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=pin_memory)

    for epoch in range(epochs):
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            preds = model(images)
            loss = criterion(preds, masks)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Check accuracy on the testing dataset
        correct_pixels, dice_score = check_accuracy(model, test_loader)
        print(f'Epoch: {epoch+1} correct_pixels: {correct_pixels} dice_score: {dice_score}')

    correct_pixels, dice_score = check_accuracy(model, val_loader, save_preds=True)
    print(f'Validation correct_pixels: {correct_pixels} dice_score: {dice_score}')

