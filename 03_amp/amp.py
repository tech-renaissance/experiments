import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import matplotlib.pyplot as plt
import time
import os
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler   # UPDATED: AMP


# Configure the data path and the logger
DATA_DIR = '/root/Downloads/imagenet'
LOG_FILE = 'log.txt'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(message)s')

# Hyperparameters
MODEL_NAME = 'resnet50'
NUM_CLASSES = 1000
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
GAMMA = 0.1
NUM_WORKERS = 8
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 5

# Enable cuDNN benchmark for better performance
torch.backends.cudnn.benchmark = True


def print_and_record(*args):
    """Print and record the contents at the same time."""
    print(*args)
    text = ' '.join(map(str, args))
    logging.info(text)


def train_model(model, device, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    """
    Function to train the model.
    """
    since = time.time()

    scaler = GradScaler()  # UPDATED: AMP

    # Store training history for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_top1': [], 'val_top1': [],
        'train_top5': [], 'val_top5': []
    }

    best_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        print_and_record(f'Epoch {epoch+1}/{num_epochs}')
        print_and_record('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                current_lr = optimizer.param_groups[0]['lr']
                print_and_record('Learning rate:', current_lr)  # Print the current learning rate
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects_top1 = 0
            running_corrects_top5 = 0

            # Using tqdm for a progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize():<5} Epoch {epoch+1}", unit="batch")

            # Iterate over data.
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                ### UPDATED: AMP (BEGIN) ###
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                ### UPDATED: AMP (END) ###

                # DELETED (The following 6 lines)
                # with torch.set_grad_enabled(phase == 'train'):
                #     outputs = model(inputs)
                #     loss = criterion(outputs, labels)
                #     if phase == 'train':
                #         loss.backward()
                #         optimizer.step()

                # Statistics
                batch_loss = loss.item()
                running_loss += batch_loss * inputs.size(0)

                # Calculate top-1 accuracy
                _, preds_top1 = torch.max(outputs, 1)
                running_corrects_top1 += torch.sum(preds_top1 == labels.data).item()

                # Calculate top-5 accuracy
                _, preds_top5 = torch.topk(outputs, k=5, dim=1)
                correct = preds_top5.eq(labels.view(-1, 1)).any(dim=1).sum().item()
                running_corrects_top5 += correct

                # Update the progress bar
                progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_top1 = running_corrects_top1 / dataset_sizes[phase]
            epoch_acc_top5 = running_corrects_top5 / dataset_sizes[phase]

            # Print both top-1 and top-5 accuracy
            print_and_record(f'{phase} Loss: {epoch_loss:.4f} Top-1: {epoch_acc_top1:.4f} Top-5: {epoch_acc_top5:.4f}')

            # Store history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_top1'].append(epoch_acc_top1)
            history[f'{phase}_top5'].append(epoch_acc_top5)

            # Save the model if it has the best validation accuracy
            if phase == 'val' and epoch_acc_top1 > best_acc:
                best_acc = epoch_acc_top1
                torch.save(model.state_dict(), f'{MODEL_NAME}_best_weights.pth')  # Uncomment to save the best model

        # Update the learning Rate
        scheduler.step()

        # Print the epoch duration
        end_time = time.time()
        epoch_time = end_time - start_time
        print_and_record(f'Epoch time: {epoch_time:.0f}s\n')

    time_elapsed = time.time() - since
    print_and_record(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print_and_record(f'Best val Top-1 Acc: {best_acc:4f}')

    return model, history


def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy.
    """
    # Create a 1x3 grid of plots
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # Plot Top-1 accuracy
    axs[0].plot(history['train_top1'], label='Train Top-1')
    axs[0].plot(history['val_top1'], label='Val Top-1')
    axs[0].set_title('Top-1 Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    # Plot Top-5 accuracy
    axs[1].plot(history['train_top5'], label='Train Top-5')
    axs[1].plot(history['val_top5'], label='Val Top-5')
    axs[1].set_title('Top-5 Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='lower right')
    axs[1].grid(True)

    # Plot loss
    axs[2].plot(history['train_loss'], label='Train Loss')
    axs[2].plot(history['val_loss'], label='Val Loss')
    axs[2].set_title('Model Loss')
    axs[2].set_ylabel('Loss')
    axs[2].set_xlabel('Epoch')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()


def main():
    """Main function."""
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    print_and_record(f"Using device: {DEVICE}")

    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create Datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}

    # Create Dataloaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
                   for x in ['train', 'val']}

    # Get dataset sizes and class names
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Model definition
    model = models.resnet50(weights=None)  # Initialize the weights, equals to resnet50(pretrained=False)

    # Move the model to the specified device
    model = model.to(DEVICE)

    # Define loss function and optimizer, enabling label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                          nesterov=True)

    # Define a learning rate scheduler
    warmup = LinearLR(optimizer, start_factor=0.001, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    # Start training
    model_ft, history = train_model(model, DEVICE, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

    # Plot the curves
    plot_training_history(history)


if __name__ == '__main__':
    main()
