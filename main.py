import torch
from EER import compute_eer
from sklearn.metrics import confusion_matrix
from multiprocessing import set_start_method
import wandb
from hyperpyyaml import load_hyperpyyaml
import os
from .wandb import load_model, save_model


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
    target_score = scores[labels == 1]
    nontarget_score = scores[labels == 0]
    eer, threshold = compute_eer(target_score, nontarget_score)
    scores[scores >= threshold] = 1
    scores[scores < threshold] = 0
    mistakes = len(scores[scores != labels])
    cm = confusion_matrix(labels, scores)

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
        pred = model(x.squeeze())
        l = loss(pred, label)
        labels = torch.cat([labels, label], axis=0)
        scores = torch.cat([scores, pred.squeeze()], axis=0)

    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    target_score = scores[labels == 1]
    nontarget_score = scores[labels == 0]
    eer, threshold = compute_eer(target_score, nontarget_score)
    scores[scores >= threshold] = 1
    scores[scores < threshold] = 0
    mistakes = len(scores[scores != labels])
    cm = confusion_matrix(labels, scores)
    return {"test EER": eer, "test mistakes": mistakes, "test confusion matrix": cm}


if __name__ == "__main__":
    with open("./hyperp.yaml") as hp_file:
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
    optim: torch.optim.Optimizer = hparams["optim"]
    loss = hparams["loss"]
    artifact = hparams["artifact"]
    torch.set_default_device(device)

    wandb.login(key="2a1c0bb6f463145bf20169508da8e60d57e39c8f")
    run = wandb.init(
        project="LightCNN",
        name=hparams["artifact"],
        group=group_name,
        config={"train batch_size": train_bs, "test batch_size": test_bs},
    )

    run, model, epoch = load_model(artifact, model, "latest", run)
    wandb.watch(model)
    optim.add_param_group({"model": model.state_dict()})

    for epoch in range(epoch, epochs):
        model.train(True)
        train_report = train(train_dataloader, model, model, loss, optim)
        wandb.log(train_report)
        model.eval()
        test_report = eval(dev_dataloader, model, model, loss)
        wandb.log(test_report)
        save_model(run, {"model": model.state_dict(), "epoch": epoch}, artifact)
        if epoch % 5 == 0:
            wandb.alert(title="info", text=f"finished epoch {epoch}")
