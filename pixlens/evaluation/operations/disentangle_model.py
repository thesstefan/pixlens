import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        with torch.no_grad():
            dummy_input = torch.zeros((1, 4, 45, 45))
            dummy_output = self.conv1(dummy_input)
            dummy_output = self.pool(dummy_output)
            dummy_output = self.conv2(dummy_output)
            dummy_output = self.pool(dummy_output)
            self.in_features = dummy_output.view(-1).shape[0]
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Classifier:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.label_encoder = LabelEncoder()

    def prepare_data(self) -> None:
        inputs = np.array(
            self.dataset["z_y"] - self.dataset["z_1"],
        )
        inputs_2 = np.array(
            self.dataset["z_2"] - self.dataset["z_neg"],
        )
        inputs = np.concatenate((inputs, inputs_2), axis=0)
        labels = self.label_encoder.fit_transform(
            self.dataset["attribute_type"],
        )

        inputs = torch.from_numpy(inputs).float()
        labels = torch.tensor(labels, dtype=torch.long)

        # Reshape inputs if necessary for CNN
        inputs = inputs.view(inputs.size(0), -1, 45, 45)
        x_train, x_test, y_train, y_test = train_test_split(
            inputs,
            labels,
            test_size=0.2,
            random_state=42,
        )
        self.train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=32, shuffle=True
        )
        self.test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=32,
        )

    def train_classifier(self, num_epochs: int = 50) -> None:
        num_classes = len(np.unique(self.dataset["attribute_type"]))
        model = CNNClassifier(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for _ in range(num_epochs):
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        self.model = model

    def evaluate_classifier(self) -> float:
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
