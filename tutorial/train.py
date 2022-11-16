import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomRotation
from torchvision.datasets import MNIST

from metrics import Metrics
from model import MyNN, MyCNN


def start_training(device_name="cuda:0", lr=0.1, epochs=15, batch_size=16, num_workers=4, lr_gamma=0.7, data_path="./data"):
    # setting the device (CPU or GPU)
    device = torch.device(device_name)

    # Setting up the Data
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    train_set = MNIST(data_path, train=True, download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    valid_set = MNIST(data_path, train=False, download=True, transform=ToTensor())
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # Initializing the model, cost/loss function and the optimizer (stochastic gradient descent)
    model = MyNN().to(device)
    cost_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    metrics = Metrics()
    best_acc = 0

    for epoch in range(epochs):
        print('#' * 40, end=" ")
        print('Epoch {}/{}'.format(epoch+1, epochs), end=" ")
        print('#' * 40)
        print("Current learning rate: ", lr_scheduler.get_last_lr()[0])

        for phase in ["training", "validating"]:

            if phase == "training":
                dataloader = train_dataloader
                model.train()
            else:
                dataloader = valid_dataloader
                model.eval()

            running_loss = 0
            for i, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == "training"):

                    outputs = model(imgs)

                    loss = cost_function(outputs, labels)

                    metrics.add_batch(outputs, labels)

                    if phase == "training":
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item()
                _, acc = metrics.get_scores()
                i = i + 1 if i == 0 else i

                print(str(phase) + " Batch: {0:7d} of {1:7d} | {2:3d}% | Loss: {3:6.5f} | Acc: {4:.4f}"
                      .format(i, len(dataloader), round((i / len(dataloader) * 100)),
                              running_loss / i, acc), end="\r")

            epoch_loss = running_loss / float(len(dataloader))
            _, acc = metrics.get_scores()

            print(str(phase) + " Batch: {0:7d} of {1:7d} | {2}% | Loss: {3:6.5f} | Acc: {4:.4f}"
                  .format(len(dataloader), len(dataloader), "100", epoch_loss, acc), end="\n")

            if phase == "validating":
                lr_scheduler.step()

                if best_acc < acc:
                    print("New best weights found! Saving weights...")
                    save_weights(model.state_dict(), acc, epoch+1)
                    best_acc = acc

    print("Best Accuracy: " + str(best_acc))


def save_weights(state_dict, acc, epoch):
    model_data = {
        "model_state_dict": state_dict,
        "metrics": acc,
        "epoch_counter": epoch,
    }
    torch.save(model_data, "./best_weights.pt")


if __name__ == "__main__":
    start_training()
