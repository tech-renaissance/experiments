import os
import torch
from torch import nn
from torchvision import datasets, transforms
from six.moves import urllib
import logging


BEST_ENSEMBLE = [0, 9, 12, 13, 14]
BEST_EOE = [
    [0, 2, 3, 8, 13],
    [2, 4, 5, 15, 16],
    [1, 6, 7, 8, 10, 11, 15]
]
NUM_TOTAL_MODELS = 17
LOG_FILE = 'log.txt'

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(message)s')
best_accuracy = 0.0


def print_and_record(*args):
    print(*args)
    text = ' '.join(map(str, args))
    logging.info(text)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class TorrentNet(nn.Module):
    def __init__(self):
        super(TorrentNet, self).__init__()
        self.name = 'TorrentNet'
        self.features = None
        self.classifier = None
        self.make_layers()
        self.weight_init()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def make_layers(self):
        feature_layers = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1))
        ]
        self.features = nn.Sequential(*feature_layers)

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.Dropout(p=0.5),  # MODIFIED: V10 - Dropout
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        ]
        self.classifier = nn.Sequential(*classifier_layers)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def mode_along_tensors(tensors, dim_m: int = 0) -> torch.Tensor:
    stacked = torch.stack(tensors, dim=dim_m)
    modes, _ = torch.mode(stacked, dim=dim_m)
    return modes


def test_one_model(model, device, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            compare_result = pred.eq(target.view_as(pred))
            correct += compare_result.sum().item()

        test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc


def test_one_ensemble(models, device, test_loader):
    num_models = len(models)
    if num_models not in [3, 5, 7, 9]:
        print_and_record('ERROR: Invalid number of models')
        exit(1)
    with torch.no_grad():
        for model in models:
            model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            res_list = list()
            for model in models:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                res_list.append(pred)
            final_pred = mode_along_tensors(res_list, dim_m=2)
            compare_result = final_pred.eq(target.view_as(pred))
            correct += compare_result.sum().item()

        test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc


def test_eoe(ensembles, device, test_loader):
    num_ensembles = len(ensembles)
    if num_ensembles not in [3, 5, 7, 9]:
        print_and_record('ERROR: Invalid number of ensembles')
        exit(1)
    with torch.no_grad():
        for ensemble in ensembles:
            for model in ensemble:
                model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ensemble_res_list = list()
            for ensemble in ensembles:
                res_list = list()
                for model in ensemble:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    res_list.append(pred)
                final_pred = mode_along_tensors(res_list, dim_m=2)
                ensemble_res_list.append(final_pred)
            ensemble_pred = mode_along_tensors(ensemble_res_list, dim_m=2)
            compare_result = ensemble_pred.eq(target.view_as(pred))
            correct += compare_result.sum().item()

        test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc


def main():
    os.makedirs('models', exist_ok=True)

    model_name_list = ['./models/TorrentNet_{:02d}.pth'.format(i) for i in range(NUM_TOTAL_MODELS)]
    website = 'https://tech-renaissance.cn/download/cnn_mnist/models/'
    for i in range(NUM_TOTAL_MODELS):
        urllib.request.urlretrieve('{}TorrentNet_{:02d}.pth'.format(website, i), model_name_list[i])
        print_and_record('Downloading TorrentNet_{:02d}.pth'.format(i))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='..', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=50, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_dict = dict()
    acc_list = list()
    models = list()
    for each in model_name_list:
        model = TorrentNet()
        model.load_state_dict(torch.load(each, map_location=device))
        model.to(device)
        models.append(model)
        acc = test_one_model(model, device, test_loader)
        acc_dict[each] = acc
        acc_list.append(acc)
        print_and_record('Using model: {} ({:.2f}%)'.format(each, acc))

    best_ensemble_models = list()
    print_and_record('\nTesting ensemble models: ', BEST_ENSEMBLE)
    for model_id in BEST_ENSEMBLE:
        best_ensemble_models.append(models[model_id])
    best_ensemble_acc = test_one_ensemble(best_ensemble_models, device, test_loader)
    print_and_record('\nBest ensemble accuracy: {:.2f}%'.format(best_ensemble_acc))

    best_eoe_ensembles = list()
    print_and_record('\nTesting eoe: ')
    for each_ensemble in BEST_EOE:
        print_and_record(each_ensemble)
    for model_combination in BEST_EOE:
        ensemble = list()
        for model_id in model_combination:
            ensemble.append(models[model_id])
        best_eoe_ensembles.append(ensemble)
    best_eoe_acc = test_eoe(best_eoe_ensembles, device, test_loader)
    print_and_record('\nBest eoe accuracy: {:.2f}%'.format(best_eoe_acc))


if __name__ == '__main__':
    main()
