import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import time


# Configure the data path and the logger
DATA_DIR = './data'
LOG_FILE = 'log.txt'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(message)s')

# Hyperparameters
BATCH_SIZE = 128
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4

# Enable cuDNN benchmark for better performance
torch.backends.cudnn.benchmark = True


def print_and_record(*args):
    """Print and record the contents at the same time."""
    print(*args)
    text = ' '.join(map(str, args))
    logging.info(text)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch_display_str):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Using tqdm for a progress bar
    train_bar = tqdm(data_loader, desc=f"Epoch {epoch_display_str} [Train]")
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip grad
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Update the progress bar
        train_bar.set_postfix(loss=loss.item(), acc=f"{correct_train / total_train:.4f}")

    epoch_loss = running_loss / total_train
    epoch_acc = correct_train / total_train
    return epoch_loss, epoch_acc


def evaluate(model, criterion, data_loader, device, epoch_display_str):
    """Evaluate the model on the test set."""
    model.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        test_bar = tqdm(data_loader, desc=f"Epoch {epoch_display_str} [Test]")
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            test_bar.set_postfix(loss=loss.item(), acc=f"{correct_test / total_test:.4f}")

    epoch_loss = running_loss / total_test
    epoch_acc = correct_test / total_test
    return epoch_loss, epoch_acc


def main():
    """Main function."""
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    print_and_record(f"Using device: {DEVICE}")

    # Configuration for the freeze & unfreeze epochs
    config = {
        'freeze_epochs': 5,
        'unfreeze_epochs': 45,
        'lr_head': 1e-3,
        'lr_finetune': 1e-4,
    }
    total_epochs = config['freeze_epochs'] + config['unfreeze_epochs']

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    # Load the pre-trained EfficientNet-B0 model
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace the classifier, for 10-class learning
    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 10)
    )

    # Move the model to the GPU
    model = model.to(DEVICE)

    # Using cross entropy as the loss function, with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0

    # STAGE 1: TRAIN THE HEAD
    print_and_record("\n--- STAGE 1: Training the classifier head ---")
    for p in model.features.parameters():
        p.requires_grad = False

    # Define the optimizer, and scheduler
    optimizer = optim.AdamW(model.classifier.parameters(), lr=config['lr_head'], weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['freeze_epochs'], eta_min=1e-6)

    for epoch in range(config['freeze_epochs']):
        epoch_display_str = f"{epoch + 1}/{total_epochs}"
        start_time = time.time()
        print_and_record(f'\nEpoch {epoch_display_str}')
        print_and_record(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, DEVICE, epoch_display_str)
        test_loss, test_acc = evaluate(model, criterion, test_loader, DEVICE, epoch_display_str)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_weights.pth')
            print_and_record(f"New best model saved with accuracy: {best_acc:.4f}")

        print_and_record(
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        print_and_record(f"Epoch time: {time.time() - start_time:.0f}s")

    # STAGE 2: FINE-TUNE THE ENTIRE MODEL
    print_and_record("\n--- STAGE 2: Fine-tuning the entire model ---")
    for p in model.parameters():
        p.requires_grad = True

    # Define the optimizer, and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['lr_finetune'], weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['unfreeze_epochs'], eta_min=1e-6)

    for epoch in range(config['unfreeze_epochs']):
        epoch_display_str = f"{epoch + 1 + config['freeze_epochs']}/{total_epochs}"
        start_time = time.time()
        print_and_record(f'\nEpoch {epoch_display_str}')
        print_and_record(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, DEVICE, epoch_display_str)
        test_loss, test_acc = evaluate(model, criterion, test_loader, DEVICE, epoch_display_str)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Save the model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_weights.pth')
            print_and_record(f"New best model saved with accuracy: {best_acc:.4f}")

        print_and_record(
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        print_and_record(f"Epoch time: {time.time() - start_time:.0f}s")

    print_and_record(f'\nTraining Completed. Best accuracy: {best_acc:.4f}')
    plot_curves(history)


def plot_curves(history):
    """Draw and save curves of training and test loss and accuracy."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Transfer Learning with EfficientNet-B0 on CIFAR-10', fontsize=20)

    # Show loss
    ax1.plot(epochs, history['train_loss'], 'o--', label='Train Loss', color='dodgerblue')
    ax1.plot(epochs, history['test_loss'], 's-', label='Test Loss', color='darkorange')
    ax1.set_title('Training and Test Loss', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Show accuracy
    ax2.plot(epochs, history['train_acc'], 'o--', label='Train Accuracy', color='dodgerblue')
    ax2.plot(epochs, history['test_acc'], 's-', label='Test Accuracy', color='darkorange')
    ax2.set_title('Training and Test Accuracy', fontsize=16)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('training_curves.png', dpi=300)
    print("\nTraining curves plot saved to 'training_curves.png'")
    plt.show()


if __name__ == '__main__':
    main()
