import torch
from EER import compute_eer
from sklearn.metrics import confusion_matrix
from multiprocessing import set_start_method
import wandb
from hyperpyyaml import load_hyperpyyaml
import os
from artifact import load_model, save_model
import numpy as np

try:
    set_start_method("spawn")
except RuntimeError:
    pass


def train(dataloader, model, loss, optim):
    cum_loss = 0
    scores = labels = torch.tensor([], device="cuda")
    for sample in dataloader:
        x = sample["feature"]
        label = sample["label"]
        pred = model(x)
        optim.zero_grad()
        l = loss(pred, label)
        l.backward()
        optim.step()
        cum_loss += l.item()
        labels = torch.cat([labels, label], axis=0)
        scores = torch.cat([scores, pred.squeeze()], axis=0)

    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    target_score = scores[labels[:, 0] == 1][:, 0]
    nontarget_score = scores[labels[:, 0] == 0][:, 0]
    eer, threshold = compute_eer(target_score, nontarget_score)
    mistakes = len(scores[np.argmax(scores, axis=1) != np.argmax(labels, axis=1)])
    cm = confusion_matrix(np.argmax(labels, axis=1), np.argmax(scores, axis=1))

    return {
        "train EER": eer,
        "train mistakes": mistakes,
        "train confusion matrix": cm,
        "train loss": cum_loss,
    }


@torch.no_grad()
def eval(dataloader, model, loss):
    scores = labels = torch.tensor([], device="cuda")
    for sample in dataloader:
        x = sample["feature"]
        label = sample["label"]
        pred = model(x)
        l = loss(pred, label)
        labels = torch.cat([labels, label], axis=0)
        scores = torch.cat([scores, pred.squeeze()], axis=0)

    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    target_score = scores[labels[:, 0] == 1][:, 0]
    nontarget_score = scores[labels[:, 0] == 0][:, 0]
    eer, threshold = compute_eer(target_score, nontarget_score)
    scores[scores >= threshold] = 1
    scores[scores < threshold] = 0
    mistakes = len(scores[np.argmax(scores, axis=1) != np.argmax(labels, axis=1)])
    cm = confusion_matrix(np.argmax(labels, axis=1), np.argmax(scores, axis=1))
    return {"test EER": eer, "test mistakes": mistakes, "test confusion matrix": cm}


if __name__ == "__main__":
    with open("/content/ASVSpoof2017-LightCNN/hyperp.yaml") as hp_file:
        hparams = load_hyperpyyaml(hp_file)

    device = hparams["DEVICE"]
    train_bs = hparams["train_batch_size"]
    test_bs = hparams["test_batch_size"]
    epochs = hparams["epochs"]
    group_name = hparams["group_name"]
    model = hparams["model"]
    train_dataset = hparams["train_dataset"]
    dev_dataset = hparams["dev_dataset"]
    train_dataloader = hparams["train_dataloader"]
    dev_dataloader = hparams["dev_dataloader"]
    loss = hparams["loss"]
    artifact = hparams["artifact"]

    wandb.login(key="2a1c0bb6f463145bf20169508da8e60d57e39c8f")
    run = wandb.init(
        project="ASVSpoof2017",
        name=hparams["artifact"],
        group=group_name,
        config={"train batch_size": train_bs, "test batch_size": test_bs},
    )

    model.to(device=device)
    run, model, epoch = load_model(artifact, model, "latest", run)
    wandb.watch(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epoch, epochs):
        model.train(True)
        train_report = train(train_dataloader, model, loss, optim)
        model.eval()
        test_report = eval(dev_dataloader, model, loss)

        wandb.log(train_report)
        wandb.log(test_report)
        save_model(run, {"model": model.state_dict(), "epoch": epoch}, artifact)
        if epoch % 5 == 0:
            wandb.alert(title="info", text=f"finished epoch {epoch}")
