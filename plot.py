import matplotlib.pyplot as plt
import torch
import numpy as np


class Discrimination:
    def __init__(self) -> None:
        self.embeddings = torch.tensor([])
        self.labels = torch.tensor([])

    def add_sample(self, e: torch.Tensor, l: torch.Tensor):
        e = torch.tensor(e).squeeze()
        l = torch.tensor(l).squeeze()
        self.embeddings = torch.cat([self.embeddings, e], dim=0)
        self.labels = torch.cat([self.labels, l], dim=0)

    def plot(self):
        # e = self.embeddings.cpu().numpy()
        # l = self.embeddings.cpu().numpy()
        feature_size = self.embeddings.size()[1]
        bonafide_embedding = self.embeddings[self.labels == 1]
        spoof_embedding = self.embeddings[self.labels == 0]
        bonafide_sample_size = bonafide_embedding.size()[0]
        spoof_sample_size = spoof_embedding.size()[0]
        bonafide_range = (
            torch.arange(feature_size).unsqueeze(0).repeat(bonafide_sample_size, 1)
        )
        spoof_range = (
            torch.arange(feature_size).unsqueeze(0).repeat(spoof_sample_size, 1)
        )
        plt.scatter(
            bonafide_range,
            bonafide_embedding,
            c="b",
            alpha=0.01,
        )

        plt.scatter(
            spoof_range,
            spoof_embedding,
            c="r",
            alpha=0.01,
        )

        plt.show()


if __name__ == "__main__":
    train_embeddings = []
    train_label = []
    test_embeddings = []
    test_label = []
    with open("./train_embedding.txt", "r") as file:
        for line in file.readlines():
            line = line.replace("[", "")
            line = line.replace("]", "")
            embedding = line.split(",")
            embedding = list(map(float, embedding))
            train_embeddings.append(embedding)
    train_embeddings = np.array(train_embeddings)

    with open("./train_label.txt", "r") as file:
        for line in file.readlines():
            line = line.replace("[", "")
            line = line.replace("]", "")
            embedding = float(line)
            train_label.append(embedding)
    train_label = np.array(train_label)

    plotter = Discrimination()
    plotter.add_sample(train_embeddings, train_label)
    plotter.plot()

    # with open("./test_embedding.txt", "r") as file:
    #     for line in file.readlines():
    #         line = line.replace("[", "")
    #         line = line.replace("]", "")
    #         embedding = line.split(",")
    #         embedding = list(map(float, embedding))
    #         test_embeddings.append(embedding)
    # test_embeddings = np.array(test_embeddings)

    # with open("./test_label.txt", "r") as file:
    #     for line in file.readlines():
    #         line = line.replace("[", "")
    #         line = line.replace("]", "")
    #         embedding = float(line)
    #         test_label.append(embedding)
    # test_label = np.array(test_label)
