import numpy as np
from EER import compute_eer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable

# use for gamma
from sklearn.model_selection import GridSearchCV

report = PrettyTable(["mode", "EER", "threshold", "confusion matrix"])

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
    with open("./test_embedding.txt", "r") as file:
        for line in file.readlines():
            line = line.replace("[", "")
            line = line.replace("]", "")
            embedding = line.split(",")
            embedding = list(map(float, embedding))
            test_embeddings.append(embedding)
    test_embeddings = np.array(test_embeddings)

    with open("./train_label.txt", "r") as file:
        for line in file.readlines():
            line = line.replace("[", "")
            line = line.replace("]", "")
            embedding = float(line)
            train_label.append(embedding)
    train_label = np.array(train_label)

    with open("./test_label.txt", "r") as file:
        for line in file.readlines():
            line = line.replace("[", "")
            line = line.replace("]", "")
            embedding = float(line)
            test_label.append(embedding)
    test_label = np.array(test_label)

    train_embeddings = (train_embeddings - train_embeddings.mean(axis=0)) / (
        train_embeddings.var(axis=0)
    )
    test_embeddings = (test_embeddings - test_embeddings.mean(axis=0)) / (
        test_embeddings.var(axis=0)
    )

    # train_embeddings = (train_embeddings - train_embeddings.min(axis=0)) / (
    #     train_embeddings.max(axis=0) - train_embeddings.min(axis=0)
    # )
    # test_embeddings = (test_embeddings - test_embeddings.min(axis=0)) / (
    #     test_embeddings.max(axis=0) - test_embeddings.min(axis=0)
    # )
    classifier = SVC(C=500, max_iter=-1, probability=True, cache_size=100000)

    # poly degree = 2
    classifier.fit(train_embeddings, train_label)
    train_pred = classifier.predict_proba(train_embeddings)

    train_traget = train_pred[train_label == 1][:, 1]
    train_nontarget = train_pred[train_label == 0][:, 1]
    train_eer, train_threshold = compute_eer(train_traget, train_nontarget)
    train_pred = np.argmax(train_pred, axis=1)
    train_cm = confusion_matrix(train_label, train_pred)

    test_pred = classifier.predict_proba(test_embeddings)
    test_traget = test_pred[test_label == 1][:, 1]
    test_nontarget = test_pred[test_label == 0][:, 1]
    test_eer, test_threshold = compute_eer(test_traget, test_nontarget)
    test_pred = np.argmax(test_pred, axis=1)
    test_cm = confusion_matrix(test_label, test_pred)

    print(classifier.kernel)
    print(classifier)
    report.clear_rows()
    report.add_row(["train", train_eer, train_threshold, train_cm])
    print(report)
    report.clear_rows()
    report.add_row(["test", test_eer, test_threshold, test_cm])
    print(report)
