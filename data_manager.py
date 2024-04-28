import os
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
# library for multiprocessing in pandas https://github.com/nalepae/pandarallel
from pandarallel import pandarallel

from utils import setup_logger

# disabling ssl check due to error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

pandarallel.initialize(progress_bar=True)
logger = setup_logger("data_manager")

class DataManager:
    def __init__(self, file_path, processed_file_path,  n_tags: int = 50) -> None:
        self.file_path = file_path
        self.processed_file_path = processed_file_path
        self.n_tags = n_tags

    def extract_tags(self) -> None:
        """Converts dataframe into necessary format"""
        df = pd.read_csv(self.file_path)
        def label_with_tags(row, tag_columns):
            tags = [col.replace("Tag_", "") for col in tag_columns if row[col] == 1]
            return ",".join(tags)

        tag_columns = [col for col in df.columns if col.startswith("Tag_")]
        # This is very long operation, so we do it only once
        df["tags"] = df.parallel_apply(lambda row: label_with_tags(row, tag_columns), axis=1)
        df = df[["Title", "Subtitle", "tags"]]

        # Fills nan values with empty string
        df.fillna("", inplace=True)

        # # Adding merged tag of merged titles
        df["text"] = df["Title"]
        df["text"] = df["text"].str.cat(df["Subtitle"], sep=" ", na_rep="")
        df.drop(columns=["Title", "Subtitle"], inplace=True)

        # # Select top n_tags
        tags_expanded = df['tags'].str.split(',', expand=True).stack()
        top_tags = tags_expanded.value_counts()[:self.n_tags].keys()
        df = df[df['tags'].parallel_apply(lambda x: any(tag in top_tags for tag in x.split(',')))]

        # # preprocessing the text
        # logger.info("Preprocessing the text")
        df["text"] = df["text"].parallel_apply(lambda x: self.text_preprocess(x))
        df.to_csv(self.processed_file_path, index=False)

    def text_preprocess(self, text: str):
        """Applies stemming and tokenization and removes stopwords"""
        # We use stemming since it is more efficient with big datasets compared to lemmatization
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text.lower())
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stopwords.words("english")]
        return " ".join(words)
    
    def get_data(self, test_size: float = 0.2) -> pd.DataFrame:
        """
        Preprocesses and retrieves the dataframe.

        Args:
            test_size: size of the test set.
            n_tags: number of tags to include
        
        Returns:
            Dataframe with two columns Text and Tags
        """

        # To avoid downloading the dataset again
        if not os.path.isfile(self.processed_file_path):
            logger.info("File is not pre-processed yet, this might take a while.")
            self.extract_tags()
            
        df = pd.read_csv(self.processed_file_path)
        df.dropna(inplace=True)

        # Train-Val-Test split
        df_train, df_test = train_test_split(df, test_size=test_size)

        return df_train, df_test
        

    




