
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def predict_metrics(true, preds):
    # Convert into correct format for metric functions
    mlb = MultiLabelBinarizer()
    true = mlb.fit_transform(true)
    preds = mlb.transform(preds)

    precision = precision_score(true, preds, average="micro")
    recall = recall_score(true, preds, average="micro")
    f1 = f1_score(true, preds, average="micro")

    return {"precision": precision, "recall": recall, "f1": f1}