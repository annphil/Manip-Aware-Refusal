import pandas as pd
import csv
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import logging


class LoadManipDataset:
    def __init__(self, file_name, train_ratio, valid_ratio, test_ratio, split_draw=False):
        self.df = self.import_data(file_name)
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.df_train, self.df_valid, self.df_test = self.split_train_test(split_draw)
        self.techs = None
        self.vuls = None

    def import_data(sel, file_name):
        # newline='' to disable its automatic newline translation.
        with open(file_name, 'r', newline='', encoding='utf-8') as infile:
            content = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            data = []
            columns = None
            for idx, row in enumerate(content):
                if idx == 0:
                    columns = row
                else:
                    data.append(row)
        dataframe = pd.DataFrame(data, columns=columns)
        # drop certain columns
        if 'ID' in dataframe.columns:
            dataframe = dataframe.drop(['ID'], axis=1)
        return dataframe

    def split_train_test(self, draw):

        df_shuffled = self.df.sample(frac=1, random_state=17).reset_index(drop=True)
        # Calculate split sizes
        train_size = int(self.train_ratio * len(df_shuffled))
        valid_size = int(self.valid_ratio * len(df_shuffled))
        test_size = len(df_shuffled) - train_size - valid_size

        # Split the DataFrame randomly
        train = df_shuffled.iloc[:train_size]
        valid = df_shuffled.iloc[train_size:train_size + valid_size]
        test = df_shuffled.iloc[train_size + valid_size:]

        logging.info(f"-----MentalManip Dataset Information-----")
        logging.info(f"Total size = {len(df_shuffled)}, manipulative:non-manipulative ratio = {len(df_shuffled[df_shuffled['Manipulative'] == '1'])/len(df_shuffled[df_shuffled['Manipulative'] == '0']):.3f}")
        logging.info(f"Train size = {len(train)}, manipulative:non-manipulative ratio = {len(train[train['Manipulative'] == '1'])/len(train[train['Manipulative'] == '0']):.3f}")
        logging.info(f"Valid size = {len(valid)}, manipulative:non-manipulative ratio = {len(valid[valid['Manipulative'] == '1'])/len(valid[valid['Manipulative'] == '0']):.3f}")
        logging.info(f"Test size = {len(test)}, manipulative:non-manipulative ratio = {len(test[test['Manipulative'] == '1'])/len(test[test['Manipulative'] == '0']):.3f}")
        logging.info("")

        return train, valid, test
    
