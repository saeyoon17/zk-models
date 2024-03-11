import csv
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class HeartFailureDataset(Dataset):
    """Heart Failure Dataset."""

    def __init__(self, split="train"):
        """Download dataset from: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download"""
        self.dataset_path = f"./heart.csv"
        self.split = split
        data = pd.read_csv(self.dataset_path)

        """Deal with categorical values"""
        chest_pain_type_dummies = pd.get_dummies(
            data["ChestPainType"], prefix="ChestPainType"
        )
        data = pd.concat([data, chest_pain_type_dummies], axis=1)
        data.drop("ChestPainType", axis=1, inplace=True)

        resting_ecg_type_dummies = pd.get_dummies(
            data["RestingECG"], prefix="RestingECG"
        )
        data = pd.concat([data, resting_ecg_type_dummies], axis=1)
        data.drop("RestingECG", axis=1, inplace=True)

        st_slope_type_dummies = pd.get_dummies(data["ST_Slope"], prefix="ST_Slope")
        data = pd.concat([data, st_slope_type_dummies], axis=1)
        data.drop("ST_Slope", axis=1, inplace=True)

        le = LabelEncoder()
        for col in data.columns:
            if data[col].dtype == "object" or data[col].dtype == "bool":
                data[col] = le.fit_transform(data[col])

        X = data.drop("HeartDisease", axis=1)
        y = data["HeartDisease"]
        X.shape, y.shape
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, shuffle=True, test_size=0.3, random_state=42
        )

        self.x_keys = list(x_train.columns.values)
        self.x_train = x_train.values.tolist()
        self.x_test = x_test.values.tolist()
        self.y_train = y_train.values.tolist()
        self.y_test = y_test.values.tolist()

    def __len__(self):
        return len(self.x_train) if self.split == "train" else len(self.x_test)

    def __getitem__(self, idx):
        return (
            {"feat": self.x_train[idx], "label": self.y_train[idx]}
            if self.split == "train"
            else {"feat": self.x_test[idx], "label": self.y_test[idx]}
        )

    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test
