import torch
from EER import compute_eer
from sklearn.metrics import confusion_matrix
from multiprocessing import set_start_method
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from typing import Literal


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
        l = loss(pred, label.squeeze())
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
def eval(dataloader, model, loss, type: Literal["dev", "eval"]):
    scores = labels = torch.tensor([], device="cuda")
    for sample in dataloader:
        x = sample["feature"]
        label = sample["label"]
        feat = model(x)
        l, pred = loss(feat, label.squeeze())
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

    if type == "dev":
        report = {"dev EER": eer, "dev mistakes": mistakes, "dev confusion matrix": cm}
    else:
        report = {
            "eval EER": eer,
            "eval mistakes": mistakes,
            "eval confusion matrix": cm,
        }

    return report


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
    train_dataloader = hparams["train_dataloader"]
    dev_dataset = hparams["dev_dataset"]
    dev_dataloader = hparams["dev_dataloader"]
    eval_dataset = hparams["eval_dataset"]
    eval_dataloader = hparams["eval_dataloader"]
    loss = hparams["loss"]
    artifact = hparams["artifact"]
    model.to(device=device)
    loss.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train(True)
        train_report = train(train_dataloader, model, loss, optim)
        model.eval()
        dev_report = eval(dev_dataloader, model, loss, type="dev")
        eval_report = eval(eval_dataloader, model, loss, type="eval")
