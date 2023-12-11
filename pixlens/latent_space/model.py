import torch
from torch import nn, optim
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def flatten_tensors(*tensors):
    flattened_tensors = [t.view(t.size(0), -1) for t in tensors if t is not None]
    return torch.cat(flattened_tensors, dim=1) if flattened_tensors else None

def polynomial_features(X, degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    return torch.tensor(X_poly, dtype=torch.float32)

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

def train_regression_model(X: torch.Tensor, Y: torch.Tensor, max_degree: int = 5, num_epochs: int = 100, learning_rate: float = 0.01):
    X_poly = polynomial_features(X, max_degree)

    model = LinearRegression(X_poly.size(1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model(X_poly)
        loss = criterion(outputs, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def get_coefficients(X: torch.Tensor, model, max_degree: int = 5):
    weights = model.linear.weight.data.numpy().flatten()
    bias = model.linear.bias.data.numpy()

    poly = PolynomialFeatures(degree=max_degree)
    _ = poly.fit_transform(X.numpy())
    feature_names = poly.get_feature_names_out()

    # Print each feature name with its corresponding weight
    for name, weight in zip(feature_names, np.concatenate(([bias[0]], weights))):
        print(f"{name}: {weight}")
