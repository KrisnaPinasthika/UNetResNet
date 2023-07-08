import pandas as pd
from datetime import datetime
import time
import numpy as np
import torch
from .Metrics import rmse, rel, accuracy_th
from .Loss import loss_l1, loss_depthsmoothness, loss_ssim


def getMetrics(y_true, y_pred):
    rmse_val = rmse(y_true, y_pred)
    rel_val = rel(y_true, y_pred)
    acc_1 = accuracy_th(y_true, y_pred, 1)
    acc_2 = accuracy_th(y_true, y_pred, 2)
    acc_3 = accuracy_th(y_true, y_pred, 3)
    return rmse_val, rel_val, acc_1, acc_2, acc_3


def save_state(save_state_path, model, optimizer, epoch, loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        save_state_path,
    )
    print(f"<-- Model state saved [{epoch} epoch] -->")


def load_state(save_state_path, model, optimizer):
    checkpoint = torch.load(save_state_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"<-- Model state loaded succesfully [Last epoch = {checkpoint['epoch']}] -->"
    )
    return model, optimizer


def train(
    model,
    model_name,
    max_depth,
    l1_weight,
    loader,
    epochs,
    optimizer,
    device,
    save_model=False,
    save_train_state=False,
):
    model.train()
    total_batch = loader.__len__()
    print(f"Training [Total batch : {total_batch}] [Model: {model_name}]")

    # Todo: create pandas dataframe
    hist_rmse, hist_rel, hist_acc_1, hist_acc_2, hist_acc_3 = [], [], [], [], []

    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        running_loss = 0.0
        running_rmse, running_rel, running_acc_1, running_acc_2, running_acc_3 = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Todo: forward + backward + optimize
            outputs = model(inputs)

            # Todo: scaled back
            with torch.no_grad():
                scaled_label = labels * max_depth
                scaled_output = outputs * max_depth

            # Todo: calculate loss
            loss_1 = l1_weight * loss_l1(scaled_label, scaled_output)
            loss_2 = loss_depthsmoothness(scaled_label, scaled_output)
            loss_3 = loss_ssim(
                labels, outputs, max_val=1.0, kernel_size=7, k1=0.01, k2=0.03
            )

            loss = loss_1 + loss_2 + loss_3

            loss.backward()
            optimizer.step()

            # Todo: zero the parameter gradients
            optimizer.zero_grad()

            with torch.no_grad():
                running_loss += loss.item()
                metrics = getMetrics(scaled_label, scaled_output)
                running_rmse += metrics[0]
                running_rel += metrics[1]
                running_acc_1 += metrics[2]
                running_acc_2 += metrics[3]
                running_acc_3 += metrics[4]

                if ((i + 1) % int(total_batch // 3) == 0) or ((i + 1) == total_batch):
                    print(
                        f"  Batch[{i+1}/{total_batch}] Loss : {(running_loss / (i + 1)):.4f} RMSE : {running_rmse  / (i + 1):.4f} REL : {running_rel  / (i + 1):.4f} ACC^1 : {running_acc_1  / (i + 1):.4f} ACC^2 : {running_acc_2  / (i + 1):.4f} ACC^3 : {running_acc_3  / (i + 1):.4f}"
                    )

        print(
            f"  --> Epoch {epoch + 1} Total training time : {(time.time() - start):.2f} Second"
        )
        # Todo: save artefact history and model
        hist_rmse.append(running_rmse.cpu().numpy() / total_batch)
        hist_rel.append(running_rel.cpu().numpy() / total_batch)
        hist_acc_1.append(running_acc_1.cpu().numpy() / total_batch)
        hist_acc_2.append(running_acc_2.cpu().numpy() / total_batch)
        hist_acc_3.append(running_acc_3.cpu().numpy() / total_batch)

    print(f"Total training time : {(time.time() - start):.2f} Second")

    current_date_name = datetime.now().strftime(r"%d-%m-%y")
    if save_model:
        train_df = pd.DataFrame()
        train_df["rmse"] = hist_rmse
        train_df["rel"] = hist_rel
        train_df["acc_1"] = hist_acc_1
        train_df["acc_2"] = hist_acc_2
        train_df["acc_3"] = hist_acc_3
        torch.save(
            model.state_dict(),
            f"./SavedModel/{model_name}_{epochs}_epoch_{current_date_name}.pt",
        )
        train_df.to_csv(
            f"./TrainingHistory/train_{model_name}_{epochs}_epoch_{current_date_name}.csv"
        )

    if save_train_state:
        save_state(
            save_state_path=f"./SavedTrainingState/{model_name}_{current_date_name}",
            model=model,
            optimizer=optimizer,
            epoch=epochs,
            loss=(running_loss / total_batch),
        )


def test(model, l1_weight, loader, max_depth, device):
    model.eval()
    total_batch = loader.__len__()
    print(f"Testing Phase [Total batch : {total_batch}]")

    running_loss = 0.0
    running_rmse, running_rel, running_acc_1, running_acc_2, running_acc_3 = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Todo: forward + backward + optimize
            outputs = model(inputs)

            # Todo: scaled back
            scaled_label = labels * max_depth
            scaled_output = outputs * max_depth

            # Todo: calculate loss
            loss_1 = l1_weight * loss_l1(scaled_label, scaled_output)
            loss_2 = loss_depthsmoothness(scaled_label, scaled_output)
            loss_3 = loss_ssim(
                labels, outputs, max_val=1.0, kernel_size=7, k1=0.01, k2=0.03
            )
            loss = loss_1 + loss_2 + loss_3

            running_loss += loss.item()
            metrics = getMetrics(scaled_label, scaled_output)
            running_rmse += metrics[0]
            running_rel += metrics[1]
            running_acc_1 += metrics[2]
            running_acc_2 += metrics[3]
            running_acc_3 += metrics[4]

        print(
            f"Loss : {(running_loss/total_batch):.4f} RMSE : {running_rmse/total_batch:.4f} REL : {running_rel/total_batch:.4f} ACC^1 : {running_acc_1 /total_batch:.4f} ACC^2 : {running_acc_2/total_batch:.4f} ACC^3 : {running_acc_3/total_batch:.4f}"
        )
