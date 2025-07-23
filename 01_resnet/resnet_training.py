import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import logging


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
WARMUP_EPOCHS = 5


def print_and_record(*args):
    """Print and record the contents at the same time."""
    print(*args)
    text = ' '.join(map(str, args))
    logging.info(text)


# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_and_record(f"Using device: {DEVICE}")

# Enable cuDNN benchmark for better performance
cudnn.benchmark = True

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
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

print_and_record("Initializing Datasets and Dataloaders...")

# Create Datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}

# Create Dataloaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=8, pin_memory=True)
               for x in ['train', 'val']}

# Get dataset sizes and class names
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Model definition
model = models.resnet50(weights=None)  # Initialize the weights, equals to resnet50(pretrained=False)

# Move the model to the specified device
model = model.to(DEVICE)

# Define loss function and optimizer, enabling label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

# Define a learning rate scheduler
warmup = LinearLR(optimizer, start_factor=0.001, total_iters=WARMUP_EPOCHS)
cosine = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS + 1, eta_min=1e-5)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    Function to train the model.
    """
    since = time.time()

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

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

                # Calculate top-1 accuracy
                _, preds_top1 = torch.max(outputs, 1)
                running_corrects_top1 += torch.sum(preds_top1 == labels.data).item()

                # Calculate top-5 accuracy
                _, preds_top5 = torch.topk(outputs, k=5, dim=1)
                running_corrects_top5 += torch.sum(preds_top5.eq(labels.view(-1, 1).expand_as(preds_top5))).item()

            if phase == 'train':
                scheduler.step()

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

        # Print the epoch duration
        end_time = time.time()
        epoch_time = end_time - start_time
        print_and_record(f'Epoch time: {epoch_time:.0f}s\n')

    time_elapsed = time.time() - since
    print_and_record(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print_and_record(f'Best val Top-1 Acc: {best_acc:4f}')

    return model, history


# Start training
model_ft, history = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)


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
    plt.show()


# Plot the curves
plot_training_history(history)


def imshow(inp, title=None):
    """
    Imshow for Tensor.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=10):
    """
    Display predictions for a few images.
    """
    was_training = model.training
    model.eval()
    images_so_far = 0

    # Get a batch of validation data
    inputs, labels = next(iter(dataloaders['val']))
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(25, 10))

    for j in range(inputs.size()[0]):
        if images_so_far < num_images:
            images_so_far += 1
            ax = plt.subplot(num_images // 5, 5, images_so_far)
            ax.axis('off')

            # Format the title to show prediction and ground truth
            pred_class = class_names[preds[j]].split(',')[0]
            true_class = class_names[labels[j]].split(',')[0]
            ax.set_title(f'Predicted: {pred_class}\n(True: {true_class})',
                         color=("green" if preds[j] == labels[j] else "red"))

            imshow(inputs.cpu().data[j])
        else:
            model.train(mode=was_training)
            return

    model.train(mode=was_training)


# Visualize some predictions
visualize_model(model_ft)
plt.ioff()
plt.show()
