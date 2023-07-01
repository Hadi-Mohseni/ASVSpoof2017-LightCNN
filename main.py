import torch
from EER import compute_eer
from multiprocessing import set_start_method
from hyperpyyaml import load_hyperpyyaml
from typing import Literal


try:
    set_start_method("spawn")
except RuntimeError:
    pass


def train(dataloader, model, loss, optim):
    cum_loss = mistakes = 0
    scores = labels = torch.tensor([], device=device)
    for batch_num, sample in enumerate(dataloader):
        print(f"----------- Training batch {batch_num} -----------")
        x: torch.Tensor = sample["feature"].to(device=device)
        label: torch.Tensor = sample["label"].to(device=device).squeeze()
        pred = model(x)
        optim.zero_grad()
        l = loss(pred, label)
        l.backward()
        optim.step()
        cum_loss += l.item()

        score = torch.sub(pred[:, 1], pred[:, 0])
        labels = torch.cat([labels, label], axis=0)
        scores = torch.cat([scores, score], axis=0)
        mistakes += len(pred[torch.argmax(pred, dim=1) != label])

    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    target_score = scores[labels == 1]
    nontarget_score = scores[labels == 0]
    eer, threshold = compute_eer(target_score, nontarget_score)

    return {"train EER": eer, "train mistakes": mistakes, "train loss": cum_loss}


@torch.no_grad()
def eval(dataloader, model, loss, type: Literal["dev", "eval"]):
    cum_loss = mistakes = 0
    scores = labels = torch.tensor([], device=device)
    for sample in dataloader:
        x: torch.Tensor = sample["feature"].to(device=device)
        label: torch.Tensor = sample["label"].to(device=device).squeeze()
        pred = model(x)
        l = loss(pred, label)
        cum_loss += l.item()

        score = torch.sub(pred[:, 1], pred[:, 0])
        labels = torch.cat([labels, label], axis=0)
        scores = torch.cat([scores, score], axis=0)
        mistakes += len(pred[torch.argmax(pred, dim=1) != label])

    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    target_score = scores[labels == 1]
    nontarget_score = scores[labels == 0]
    eer, threshold = compute_eer(target_score, nontarget_score)

    if type == "dev":
        report = {"dev EER": eer, "dev mistakes": mistakes, "dev loss": cum_loss}
    else:
        report = {"eval EER": eer, "eval mistakes": mistakes, "eval loss": cum_loss}
    return report


if __name__ == "__main__":
    with open("./hyperp.yaml") as hp_file:
        hparams = load_hyperpyyaml(hp_file)

    device = hparams["DEVICE"]
    train_dataset = hparams["train_dataset"]
    train_dataloader = hparams["train_dataloader"]
    dev_dataset = hparams["dev_dataset"]
    dev_dataloader = hparams["dev_dataloader"]
    eval_dataset = hparams["eval_dataset"]
    eval_dataloader = hparams["eval_dataloader"]
    model = hparams["model"]
    loss = hparams["loss"]
    train_bs = hparams["train_batch_size"]
    test_bs = hparams["test_batch_size"]
    epochs = hparams["epochs"]

    checkpoint = torch.load("checkpoint.pt")
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    loss.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        print(f"----------------- start epoch {epoch} -----------------")
        model.train(True)
        train_report = train(train_dataloader, model, loss, optim)
        print(train_report)

        model.eval()
        dev_report = eval(dev_dataloader, model, loss, type="dev")
        print(dev_report)
        eval_report = eval(eval_dataloader, model, loss, type="eval")
        print(eval_report)
