from collections import defaultdict
from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class ClassifierType(Enum):
    CNN = auto()
    LINEAR = auto()


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
        output: torch.Tensor = self.fc3(x)
        return output


class LinearClassifier(nn.Module):
    def __init__(self, input_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the tensor
        output: torch.Tensor = self.fc(x)
        return output


class Classifier:
    def __init__(self, dataset: pd.DataFrame, save_data: Path) -> None:
        self.dataset = dataset
        self.label_encoder = LabelEncoder()
        self.save_data = save_data
        self.checkpoint_path = save_data / "model_checkpoint.pth"
        self.classifier_type = ClassifierType.LINEAR
        self.model: nn.Module

    def save_checkpoint(self, filename: Path | None = None) -> None:
        if filename is None:
            filename = self.checkpoint_path
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: Path) -> None:
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def prepare_data(self) -> None:
        inputs_list = []
        labels_list = []

        for i in range(len(self.dataset)):
            current_attribute_new = self.dataset.loc[i, "attribute_new"]
            current_z_end = self.dataset.loc[i, "z_end"]

            # Find all rows with the same 'attribute_new'
            filtered_dataset = self.dataset.iloc[i + 1 :]

            # Find matching rows in the filtered dataset
            matching_rows = filtered_dataset[
                filtered_dataset["attribute_new"] == current_attribute_new
            ]

            for _, row in matching_rows.iterrows():
                z_end_diff = torch.abs(current_z_end - row["z_end"]).to(
                    "cpu",
                    dtype=torch.float32,
                )
                inputs_list.append(z_end_diff)

                label = self.dataset.loc[i, "attribute_type"]
                labels_list.append(label)

        # Stack all tensors in the list into a single tensor
        inputs = torch.stack(inputs_list)

        # Transform labels to numerical values
        labels = self.label_encoder.fit_transform(labels_list)
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
            TensorDataset(x_train, y_train),
            batch_size=32,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=32,
        )

    def train_classifier(self, num_epochs: int = 50) -> None:
        num_classes = len(np.unique(self.dataset["attribute_type"]))
        if self.classifier_type == ClassifierType.CNN:
            self.model = CNNClassifier(num_classes)
        elif self.classifier_type == ClassifierType.LINEAR:
            # Adjust input_features based on your input size
            input_features = 45 * 45 * 4
            self.model = LinearClassifier(input_features, num_classes)
        else:
            msg = "Invalid classifier type."
            raise ValueError(msg)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
            ):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

        # Optionally save the model after training
        self.save_checkpoint(self.checkpoint_path)

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

    def evaluate_balanced_accuracy(self) -> float:
        self.model.eval()
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                for label, prediction in zip(labels, predicted):  # noqa: B905
                    if label == prediction:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1
        class_accuracies = [
            class_correct[i] / class_total[i] for i in class_correct
        ]
        # Calculate balanced accuracy
        return sum(class_accuracies) / len(class_accuracies)

    def plot_confusion_matrix(self) -> None:
        self.model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 10))

        # Get class names from LabelEncoder
        class_names = self.label_encoder.inverse_transform(
            sorted(set(all_labels)),
        )

        # Plotting the confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.ylabel("Actual Labels")
        plt.xlabel("Predicted Labels")
        plt.title("Confusion Matrix")
        plt.savefig(self.save_data / "confusion_matrix.png")
