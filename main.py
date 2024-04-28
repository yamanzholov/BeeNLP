from turtle import reset
from data_manager import DataManager
from model import KeyWordExtractor
from metrics import predict_metrics
from utils import setup_logger
import pandas as pd

logger = setup_logger("main")

FILE_PATH = "./data/Medium_Clean.csv"
PROCESSED_FILE_PATH = "./data/medium_stories_processed.csv"
SAVE_MODEL_PATH = "./data/preds.json"
N_TAGS = 50
KEYWORDS_PER_TAG = 20
TEST_SIZE = 0.2
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


if __name__=="__main__":
    d_manager = DataManager(FILE_PATH, PROCESSED_FILE_PATH, N_TAGS)
    df_train, df_test = d_manager.get_data(TEST_SIZE)
    model_manager = KeyWordExtractor(keywords_per_tag=KEYWORDS_PER_TAG)
    model_manager.train(df_train, SAVE_MODEL_PATH)

    results = []
    # choosing the best threshold
    for threshold in THRESHOLDS:
        true, preds = model_manager.evaluate(df_test, SAVE_MODEL_PATH, threshold)
        res = predict_metrics(true, preds)
        res["threshold"] = threshold
        results.append(res)

    res_df = pd.DataFrame(results)
    res_df.to_csv("./data/results.csv", index=False)
        

    

