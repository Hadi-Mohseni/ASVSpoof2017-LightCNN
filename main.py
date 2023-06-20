import torch
import numpy as np
from EER import compute_eer
from sklearn.metrics import confusion_matrix
import dataset
from torch.utils.data import DataLoader
from multiprocessing import set_start_method
from encoder import ECAPA_TDNN
from loss import OCSoftmax
import wandb
from hyperpyyaml import load_hyperpyyaml
from encoder.ECAPA_TDNN import ECAPA_TDNN
from classifier.mlp import MLP2
import os
import torchaudio

try:
    set_start_method("spawn")
except RuntimeError:
    pass


def train(dataloader, encoder, classifier, loss_fn, optim):
    cum_loss = 0
    scores = labels = torch.tensor([], device="cuda")
    for sample in dataloader:
        x = sample["feature"]
        label = sample["label"]
        x, _ = encoder(x)
        pred = classifier(x.squeeze())
        optim.zero_grad()
        l = loss_fn(pred, label)
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
def eval(dataloader, encoder, classifier, loss):
    scores = labels = torch.tensor([], device="cuda")
    for sample in dataloader:
        x = sample["feature"]
        label = sample["label"]
        file_name = sample["file_name"]
        x, _ = encoder(x)
        pred = classifier(x.squeeze())
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


def set_requires_grad(model, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad = requires_grad


if __name__ == "__main__":
    with open("./hyperp.yaml") as hp_file:
        hparams = load_hyperpyyaml(hp_file)

    device = hparams["DEVICE"]
    train_bs = hparams["train_batch_size"]
    test_bs = hparams["test_batch_size"]
    learning_rate = hparams["learning_rate"]
    encoder_requires_grad = hparams["encoder_requires_grad"]
    classifier_requires_grad = hparams["classifier_requires_grad"]
    epochs = hparams["epochs"]
    group_name = hparams["group_name"]
    classifier = hparams["classifier"]
    # encoder = hparams["encoder"]
    encoder = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
    train_dataset = hparams["train_dataset"]
    dev_dataset = hparams["dev_dataset"]
    train_dataloader = hparams["train_dataloader"]
    dev_dataloader = hparams["dev_dataloader"]

    wandb.login(key="2a1c0bb6f463145bf20169508da8e60d57e39c8f")
    run = wandb.init(
        project="ASVSpoof2019-PA",
        name=hparams["artifact"],
        group=group_name,
        config={
            "train batch_size": train_bs,
            "test batch_size": test_bs,
            "learning rate": learning_rate,
            "fintune": encoder_requires_grad,
            "classifier channels": classifier.channels,
            "batch norm": classifier.bn,
            "dopout": classifier.dropout,
        },
    )

    try:
        artifact = run.use_artifact(f"{hparams['artifact']}:latest", type="model")
        artifact_dir = artifact.download()
        if os.path.exists(artifact_dir):
            checkpoint = torch.load(os.path.join(artifact_dir, "checkpoint.pt"))
            encoder.load_state_dict(checkpoint["encoder"])
            classifier.load_state_dict(checkpoint["classifier"])
            epoch = checkpoint["epoch"]
    except:
        epoch = 0

    parameters = list(classifier.parameters())
    if encoder_requires_grad:
        parameters += list(encoder.parameters())
    loss_fn = torch.nn.BCELoss()
    optim = torch.optim.SGD(parameters, learning_rate)
    loss_fn.to(device=device)
    set_requires_grad(encoder, encoder_requires_grad)
    set_requires_grad(classifier, classifier_requires_grad)

    encoder.to(device=device)
    classifier.to(device=device)
    wandb.watch(classifier)
    wandb.watch(encoder)

    for epoch in range(epoch, epochs):
        classifier.train(True)
        encoder.train(encoder_requires_grad)
        train_report = train(train_dataloader, encoder, classifier, loss_fn, optim)
        wandb.log(train_report)
        classifier.eval()
        encoder.eval()
        test_report = eval(dev_dataloader, encoder, classifier, loss_fn)
        wandb.log(test_report)

        torch.save(
            {
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "epoch": epoch,
                "train_EER": 1,
                "test_EER": 2,
            },
            "checkpoint.pt",
        )
        save_artifact = wandb.Artifact(name=hparams["artifact"], type="model")
        save_artifact.add_file("checkpoint.pt")
        run.log_artifact(save_artifact)
        if epoch % 5 == 0:
            wandb.alert(title="info", text=f"finished epoch {epoch}")
