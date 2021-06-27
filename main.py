import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torchnet.meter.confusionmeter as cm

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import sklearn as sk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(model, criterion, optimizer, scheduler, epochs):
    starttime = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    top_accuracy = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        for i in ['train', 'val']:
            if i == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #iterator
            for inputs, labels in dataloaders[i]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(i == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if i == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[i]
            epoch_acc = running_corrects.double() / dataset_sizes[i]

            # for printing and graphs
            if i == "train":
                epoch_loss = running_loss / dataset_sizes[i]
                epoch_acc = running_corrects.double() / dataset_sizes[i]

                train_loss.append(running_loss / dataset_sizes[i])
                train_acc.append(running_corrects.double() / dataset_sizes[i])
                epoch_counter_train.append(epoch)
            if i == "val":
                epoch_loss = running_loss / dataset_sizes[i]
                epoch_acc = running_corrects.double() / dataset_sizes[i]

                val_loss.append(running_loss / dataset_sizes[i])
                val_acc.append(running_corrects.double() / dataset_sizes[i])
                epoch_counter_val.append(epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                i, epoch_loss, epoch_acc))

            # deep copy the best model
            if i == 'val' and epoch_acc > top_accuracy:
                top_accuracy = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - starttime
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(top_accuracy))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # Data augmentation and normalization for training
    # Just normalization for validation & test
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(210),
            transforms.RandomResizedCrop(110),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([
            transforms.Resize(210),
            transforms.RandomResizedCrop(110),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([
            transforms.Resize(210),
            transforms.CenterCrop(210),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}



    datasetdir = 'shipspotting'
    image_datasets = {x: datasets.ImageFolder(os.path.join(datasetdir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=8)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(image_datasets['train'].classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # lists for graph generation
    epoch_counter_train = []
    epoch_counter_val = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    model_ft = models.resnet18(pretrained=True)
    #number of features taken from pretrained model, number of classes generalised to training data
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Using Adam as the parameter optimizer
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999))

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           epochs=2)

    # Plot the losses in training & validation
    plt.figure(1)
    plt.title("Training and Validation Losses")
    plt.xlabel('Epoch (#)')
    plt.ylabel('Loss')
    plt.plot(epoch_counter_train, train_loss, color='r', label="Training Loss")
    plt.plot(epoch_counter_val, val_loss, color='g', label="Validation Loss")
    plt.legend()
    plt.show()

    # Plot the accuracies in training & validation
    plt.figure(2)
    plt.title("Training and Validation Accuracies")
    plt.xlabel('Epoch (#)')
    plt.ylabel('Accuracy')
    plt.plot(epoch_counter_train, train_acc, color='r', label="Training Accuracy")
    plt.plot(epoch_counter_val, val_acc, color='g', label="Validation Accuracy")
    plt.legend()
    plt.show()

    # Test the accuracy with test data
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

    # Class wise testing accuracy
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs, 1)
            point = (predicted == labels).squeeze()
            for j in range(len(labels)):
                label = labels[j]
                class_correct[label] += point[j].item()
                class_total[label] += 1

    # Get the confusion matrix for testing data
    confusion_matrix = cm.ConfusionMeter(num_classes)
    ground_truth = torch.eye(num_classes)
    classification_reports = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs, 1)
            confusion_matrix.add(predicted, labels)

        print(class_names)
        print(confusion_matrix.conf)

        precision = []
        recall = []
        F1 = []
        specificity = []
        for i in range(num_classes):
            ground_truth[i] = ground_truth[i] * sum(confusion_matrix.conf[i])
            classTP = confusion_matrix.conf[i][i]
            classTP_FP = sum(confusion_matrix.conf[i])
            classP = sum(confusion_matrix.conf[l][i] for l in range(num_classes))
            classFN = classP - classTP
            classTN = sum(confusion_matrix.conf[k][k] for k in range(num_classes)) - classTP
            classFP = sum(confusion_matrix.conf[i]) - classTP
            classFPR = classFP/(classFP + classTN)
            classFOR = classFN/(classFN+classTP)

            classprecision = classTP / classTP_FP
            classrecall = classTP / classP
            classF1 = (2 * classprecision * classrecall) / (classprecision + classrecall)
            classspecificity = classTN / (classTN + classFP)


            precision.append(classprecision)
            recall.append(classrecall)
            F1.append(classF1)
            specificity.append(classspecificity)
            print("Precision, \trecall, \t\tF1-score, \t\tspecificity, \t\tFPR, \t\t\tFOR of " + str(class_names[i]) + " = ")
            print(str(classprecision) + " & \t\t" + str(classrecall) + " & \t" + str(classF1) + " & \t" + str(classspecificity) + " & \t" + str(classFPR) + " & \t" + str(classFOR))

        #print(sk.multilabel_confusion_matrix(confusion_matrix.conf, ground_truth ))


    # Confusion matrix as a heatmap
    plt.figure(3)
    con_m = confusion_matrix.conf
    df_con_m = pd.DataFrame(con_m, index=[i for i in class_names], columns=[i for i in class_names])
    sn.set(font_scale=1.1)
    sn.heatmap(df_con_m, annot=True, fmt='g', annot_kws={"size": 10}, cbar=False, cmap="YlOrRd")
    plt.show()



