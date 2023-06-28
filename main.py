import torch
from EER import compute_eer
from multiprocessing import set_start_method
from hyperpyyaml import load_hyperpyyaml
<<<<<<< HEAD
import numpy as np
=======
from artifact import load_model, save_model
>>>>>>> better_implemenation
from typing import Literal


try:
    set_start_method("spawn")
except RuntimeError:
    pass

torch.nn.BCELoss(
    torch.tensor(0.9, 0.1),
)


def train(dataloader, model, loss, optim):
    cum_loss = mistakes = 0
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
        pred_log = torch.log(pred)
        scores = torch.sub(pred_log[:, 0], pred_log[:, 1])
        labels = torch.cat([labels, label], axis=0)
        scores = torch.cat([scores, pred.squeeze()], axis=0)
        mistakes += len(pred[torch.argmax(pred, dim=1) != torch.argmax(label, dim=1)])

    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    target_score = scores[labels[:, 0] == 1]
    nontarget_score = scores[labels[:, 0] == 0]
    eer, threshold = compute_eer(target_score, nontarget_score)

    return {"train EER": eer, "train mistakes": mistakes, "train loss": cum_loss}


@torch.no_grad()
def eval(dataloader, model, loss, type: Literal["dev", "eval"]):
    scores = labels = torch.tensor([], device="cuda")
    cum_loss = mistakes = 0
    for sample in dataloader:
        x = sample["feature"]
        label = sample["label"]
        pred = model(x)
        l = loss(pred, label.squeeze())
        pred_log = torch.log(pred)
        scores = torch.sub(pred_log[:, 0], pred_log[:, 1])
        labels = torch.cat([labels, label], axis=0)
        scores = torch.cat([scores, pred.squeeze()], axis=0)
        cum_loss += l.item()
        mistakes += len(pred[torch.argmax(pred, dim=1) != torch.argmax(label, dim=1)])

    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    target_score = scores[labels[:, 0] == 1]
    nontarget_score = scores[labels[:, 0] == 0]
    eer, threshold = compute_eer(target_score, nontarget_score)

    if type == "dev":
        report = {"dev EER": eer, "dev mistakes": mistakes, "dev loss": cum_loss}
    else:
        report = {"eval EER": eer, "eval mistakes": mistakes, "eval loss": cum_loss}

    return report


if __name__ == "__main__":
    with open("/content/ASVSpoof2017-LightCNN/hyperp.yaml") as hp_file:
        hparams = load_hyperpyyaml(hp_file)

    device = hparams["DEVICE"]
    model = hparams["model"]
    train_dataset = hparams["train_dataset"]
    train_dataloader = hparams["train_dataloader"]
    dev_dataset = hparams["dev_dataset"]
    dev_dataloader = hparams["dev_dataloader"]
    eval_dataset = hparams["eval_dataset"]
    eval_dataloader = hparams["eval_dataloader"]
    train_bs = hparams["train_batch_size"]
    test_bs = hparams["test_batch_size"]
    epochs = hparams["epochs"]
<<<<<<< HEAD
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
=======
    loss = hparams["loss"]
    artifact = hparams["artifact"]

    wandb.login(key="2a1c0bb6f463145bf20169508da8e60d57e39c8f")
    run = wandb.init(
        project="ASVSpoof2017",
        name=hparams["run_name"],
        group=hparams["group_name"],
        notes=hparams["notes"],
    )

    model.to(device=device)

    run, model, epoch = load_model(artifact, model, "latest", run)
    wandb.watch(model)
>>>>>>> better_implemenation
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train(True)
        train_report = train(train_dataloader, model, loss, optim)
        model.eval()
        dev_report = eval(dev_dataloader, model, loss, type="dev")
        eval_report = eval(eval_dataloader, model, loss, type="eval")
