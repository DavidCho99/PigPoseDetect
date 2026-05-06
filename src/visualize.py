import numpy as np
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from src.model import device


'''
This codes from the example from Section 13.2
'''
################################################################
# Roughly a 95% confidence interval estimate for true accuracy
# based on the validation results per epoch.


def visualize_acc_loss(results, val_size):
    # Roughly a 95% confidence interval estimate for true accuracy
    # based on the validation results per epoch.
    # -1 b/c pre-train numbers @ pos. 0
    num_epochs = len(results["train_acc"])-1
    z = stats.norm.ppf(0.95)

    A = torch.tensor(results['val_acc'])
    dev = z*(A*(1-A)/val_size)**(1/2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Plot the accuracy on train & validate sets by epoch:
    x = [j for j in range(1, num_epochs+1)]
    ax[0].plot(x, results["train_acc"][1:], color="black", label="train")
    ax[0].plot(x, results["val_acc"][1:], color="blue", label="validation")

    best = results['epoch_used']
    y = results["val_acc"][best]
    ax[0].plot(best, y, marker='o', color="green", label=f"{y:.4f}")

    v0 = torch.tensor(results["val_acc"]) - dev
    v1 = torch.tensor(results["val_acc"]) + dev
    ax[0].fill_between(x, v0[1:], v1[1:], color="blue",
                       alpha=0.1, label="95% CI estimate")

    ax[0].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax[0].legend(loc="lower right")

    # Similarly with the loss:
    ax[1].plot(x, results["train_loss"][1:], label="train")
    ax[1].plot(x, results["val_loss"][1:], label="validation")
    y = results["val_loss"][best]
    ax[1].plot(best, y, marker='o', color="green", label=f"{y:.4f}")

    ax[1].set(title="Loss", xlabel="epoch", ylabel="loss")
    ax[1].legend(loc="upper right")

    # Save the image files. Because it run on the GPU.
    plt.savefig(
        '/content/drive/MyDrive/Pig_Project/pig_posture_GPU/outputs/training_result_plot.png')

    print("--- Plot Saved to training_result_plot.png ---")


# https://www.geeksforgeeks.org/pandas/bar-plot-in-matplotlib/
'''
import numpy as np 
import matplotlib.pyplot as plt 

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

IT = [12, 30, 1, 8, 22] 
ECE = [28, 6, 16, 5, 10] 
CSE = [29, 3, 24, 25, 17] 

br1 = np.arange(len(IT)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

plt.bar(br1, IT, color ='r', width = barWidth, 
        edgecolor ='grey', label ='IT') 
plt.bar(br2, ECE, color ='g', width = barWidth, 
        edgecolor ='grey', label ='ECE') 
plt.bar(br3, CSE, color ='b', width = barWidth, 
        edgecolor ='grey', label ='CSE') 

plt.xlabel('Branch', fontweight ='bold', fontsize = 15) 
plt.ylabel('Students passed', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(IT))], 
        ['2015', '2016', '2017', '2018', '2019'])

plt.legend()
plt.show()
'''


def visualize_errors(model, val_loader):
    model.eval()

    num_classes = 5
    correct_counts = np.zeros(num_classes, dtype=int)
    wrong_counts = np.zeros(num_classes, dtype=int)

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.argmax(dim=1)

            for i in range(len(target)):
                true_class = target[i].item()

                if pred[i] == target[i]:
                    correct_counts[true_class] += 1
                else:
                    wrong_counts[true_class] += 1

    barWidth = 0.3
    fig = plt.subplots(figsize=(12, 8))

    correct_bar = np.arange(len(correct_counts))
    wrong_bar = [x + barWidth for x in correct_bar]

    plt.bar(correct_bar, correct_counts, color='g', width=barWidth,
            edgecolor='grey', label='Correct')
    plt.bar(wrong_bar, wrong_counts, color='r', width=barWidth,
            edgecolor='grey', label='Wrong')

    plt.xlabel('Class', fontweight='bold', fontsize=15)
    plt.ylabel('Count', fontweight='bold', fontsize=15)

    plt.xticks(
        [r + barWidth / 2 for r in range(num_classes)],
        [f'Class {i}' for i in range(num_classes)]
    )
    save_path = "/content/drive/MyDrive/Pig_Project/pig_posture_GPU/outputs/validation_correct_wrong.png"
    plt.legend()
    plt.savefig(save_path)

    print(f"--- Validation correct/wrong bar graph saved to {save_path} ---")
