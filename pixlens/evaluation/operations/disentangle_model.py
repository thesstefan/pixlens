import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CNNClassifier(nn.Module):
    def __init__(self, num_classes) -> None:
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(...)  # Define your layers
        self.fc = nn.Linear(..., num_classes)

    def forward(self, x) -> x:
        x = self.conv1(x)
        x = self.fc(x)
        return x


class Classifier:
    def __init__(self, dataset):
        self.dataset = dataset
        self.label_encoder = LabelEncoder()

    def prepare_data(self):
        inputs = np.array(
            self.dataset["z_y"] - self.dataset["z_1"],
        )
        inputs_2 = np.array(
            self.dataset["z_2"] - self.dataset["z_neg"],
        )
        inputs = np.concatenate((inputs, inputs_2), axis=0)
        labels = self.label_encoder.fit_transform(
            self.dataset["attribute_type"]
        )

        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Reshape inputs if necessary for CNN
        inputs = inputs.view(
            inputs.size(0), -1, 28, 28
        ) 
        X_train, X_test, y_train, y_test = train_test_split(
            inputs, labels, test_size=0.2, random_state=42
        )
        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=32, shuffle=True
        )
        self.test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=32
        )

    def train_classifier(self, num_epochs=10):
        num_classes = len(np.unique(self.dataset["attribute_type"]))
        model = CNNClassifier(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        self.model = model

    def evaluate_classifier(self):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
