import argparse
import os

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.metrics import Metrics
from src.model import *


def start_training(model_name, cuda_device="0", lr=0.1, epochs=15, batch_size=16, num_workers=4, lr_gamma=0.7,
                   data_path=os.path.abspath("data")):
    # setting the device to GPU if available else to CPU
    device = torch.device("cuda:" + cuda_device if torch.cuda.is_available() else "cpu")
    # Setting up the Data
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Downloading the dataset, creating dataset object and packing it into the dataloader
    # Training data
    train_set = MNIST(data_path, train=True, download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    # Validation data
    valid_set = MNIST(data_path, train=False, download=True, transform=ToTensor())
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # Initializing the model
    model = eval(model_name + "()").to(device)
    # Initializing the cost/loss function
    cost_function = torch.nn.CrossEntropyLoss()
    # Initializing the optimizer (stochastic gradient descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Setting a learning rate scheduler to reduce the learning rate every epoch
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    # Initializing the metrics to calculate the accuracy
    metrics = Metrics()
    best_acc = 0

    for epoch in range(epochs):
        print('#' * 40, end=" ")
        print('Epoch {}/{}'.format(epoch+1, epochs), end=" ")
        print('#' * 40)
        print("Current learning rate: ", lr_scheduler.get_last_lr()[0])

        # Every epoch has a training and validating phase
        for phase in ["training", "validating"]:

            if phase == "training":
                dataloader = train_dataloader
                model.train()
            else:
                dataloader = valid_dataloader
                model.eval()

            running_loss = 0
            # Getting the batches(n amount of images and labels randomly grouped together by the dataloader)
            for i, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                # Context-manager that sets gradient calculation on only while training
                with torch.set_grad_enabled(phase == "training"):

                    outputs = model(imgs)  # Giving the model the inputs for the forward pass

                    loss = cost_function(outputs, labels)  # calculating the loss

                    metrics.add_batch(outputs, labels)  # calculating the accuracy

                    if phase == "training":
                        loss.backward()  # Computes the gradients of current batch.
                        optimizer.step()  # Updating the parameters of the model.
                        optimizer.zero_grad()  # clearing gradients from last step.

                running_loss += loss.item()  # accumulating loss
                _, acc = metrics.get_scores()  # getting the accuracy

                i = i + 1 if i == 0 else i
                print(str(phase) + " Batch: {0:7d} of {1:7d} | {2:3d}% | Loss: {3:6.5f} | Acc: {4:.4f}"
                      .format(i, len(dataloader), round((i / len(dataloader) * 100)),
                              running_loss / i, acc), end="\r")

            epoch_loss = running_loss / float(len(dataloader))  # calculate the mean loss
            _, acc = metrics.get_scores() # getting the accuracy

            print(str(phase) + " Batch: {0:7d} of {1:7d} | {2}% | Loss: {3:6.5f} | Acc: {4:.4f}"
                  .format(len(dataloader), len(dataloader), "100", epoch_loss, acc), end="\n")

            if phase == "validating":
                lr_scheduler.step()  # reducing the learning rate after validation phase

                if best_acc < acc:  # saving new weights if they are better
                    print("New best weights found! Saving weights...")
                    save_weights(model.state_dict(), acc, epoch+1, model_name)
                    best_acc = acc

    print("Best Accuracy: " + str(best_acc))


# function for saving weights of the model
def save_weights(state_dict, acc, epoch, model_name):
    if not os.path.exists(os.path.abspath("weights")):
        os.mkdir(os.path.abspath("weights"))

    model_data = {
        "model_state_dict": state_dict,  # the parameter of the model
        "metrics": acc,
        "epoch_counter": epoch,
    }
    torch.save(model_data, os.path.abspath("weights/" + model_name + "_best_weights.pt"))


# this will only be executed when you start this script explicitly in the shell like "python train.py"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MyNN", help="Name of the model class. Default: \"MyNN\" or \"MyCNN\". "
                                                        "Classes found in src/model.py you can implement your own model there.")
    args = parser.parse_args()
    start_training(model_name=args.model)
