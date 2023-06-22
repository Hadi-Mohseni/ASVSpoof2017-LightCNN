from torch.nn import Module
from wandb.sdk.wandb_run import Run
import os
import torch
import wandb


def load_model(artifact: str, model: Module, version: str, run: Run):
    """
    load saved checkpoint of a model

    Parameters
    ----------
    artifact : str
        name of the artifact the model saved on.
    model : Module
        the model which the parameters should be loaded.
    version : str
        version of the artifact. e.g. "latest"
    run : Run
        wandb Run that uses artifact.

    Returns
    -------
    (run, model, epoch)
        - run: wandb Run that uses the artifact from now on.
        - model: pretrained model.
        - epoch: resuming epoch number.
    """
    try:
        artifact = run.use_artifact(f"{artifact}:{version}", type="model")
        artifact_dir = artifact.download()
        checkpoint = torch.load(os.path.join(artifact_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint["epoch"]
        print("model loaded")
    except Exception as e:
        print(str(e))
        epoch = 0

    return run, model, epoch


def save_model(run: Run, params: dict, artifact: str):
    torch.save(params, "checkpoint.pt")
    save_artifact = wandb.Artifact(name=artifact, type="model")
    save_artifact.add_file("checkpoint.pt")
    run.log_artifact(save_artifact)
