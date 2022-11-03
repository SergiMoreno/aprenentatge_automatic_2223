# -*- coding: utf-8 -*-
import torch
import math
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.datasets import make_friedman1

def train(iter, model, optimizer):
    loss_p = np.zeros((iter))
    for t in range(iter):
        y_pred = model(X_train)

        loss = loss_fn(y_pred, y_train)
        loss_p[t] = loss.item()

        model.zero_grad()

        loss.backward()
        with torch.no_grad():
            optimizer.step()

    linear_layer = model[0]

    plt.title("Funció de pèrdua a cada iteració")
    plt.plot(loss_p)
    plt.show()

X, y = make_friedman1(n_samples=2000, n_features=10, noise=0.0, random_state=33)

X = torch.from_numpy(X)
y = torch.from_numpy(y)
X = X.float()
y = y.float()

print(X.shape)
print(y.shape)

# Separar en conjunt d'entrenament i test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)


model = torch.nn.Sequential(
    torch.nn.Linear(10, 1),
    torch.nn.Flatten(0, 1))

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train(5000, model, optimizer)

y_pred = model(X_test)

plt.title("Resultats visuals 1")
plt.scatter(y_test, y_pred.detach().numpy(), c="red")
plt.plot(y_test, y_test)
plt.show()


# ANOTHER

model1 = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.Linear(5, 1),
    torch.nn.Flatten(0, 1))

optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate)

train(5000, model1, optimizer1)

y_pred = model1(X_test)

plt.title("Resultats visuals 2")
plt.scatter(y_test, y_pred.detach().numpy(), c="red")
plt.plot(y_test, y_test)
plt.show()

# ANOTHER
learning_rate = 1e-7
model2 = torch.nn.Sequential(
    torch.nn.Linear(10, 7),
    torch.nn.Linear(7, 4),
    torch.nn.Linear(4, 1),
    torch.nn.Flatten(0, 1))

optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate)

train(5000, model2, optimizer2)

y_pred = model2(X_test)

plt.title("Resultats visuals 3")
plt.scatter(y_test, y_pred.detach().numpy(), c="red")
plt.plot(y_test, y_test)
plt.show()

# ANOTHER
learning_rate = 1e-7
model3 = torch.nn.Sequential(
    torch.nn.Linear(10, 8),
    torch.nn.Linear(8, 6),
    torch.nn.Linear(6, 4),
    torch.nn.Linear(4, 2),
    torch.nn.Linear(2, 1),
    torch.nn.Flatten(0, 1))

optimizer3 = optim.SGD(model3.parameters(), lr=learning_rate)

train(5000, model3, optimizer3)

y_pred = model3(X_test)

plt.title("Resultats visuals 5 Learning -7")
plt.scatter(y_test, y_pred.detach().numpy(), c="red")
plt.plot(y_test, y_test)
plt.show()

# ANOTHER
learning_rate = 1e-6
model4 = torch.nn.Sequential(
    torch.nn.Linear(10, 8),
    torch.nn.Linear(8, 6),
    torch.nn.Linear(6, 4),
    torch.nn.Linear(4, 2),
    torch.nn.Linear(2, 1),
    torch.nn.Flatten(0, 1))

optimizer4 = optim.SGD(model4.parameters(), lr=learning_rate)

train(5000, model4, optimizer4)

y_pred = model4(X_test)

plt.title("Resultats visuals 5 Learning -6")
plt.scatter(y_test, y_pred.detach().numpy(), c="red")
plt.plot(y_test, y_test)
plt.show()

# ANOTHER
learning_rate = 1e-6
model5 = torch.nn.Sequential(
    torch.nn.Linear(10, 9),
    torch.nn.Linear(9, 8),
    torch.nn.Linear(8, 7),
    torch.nn.Linear(7, 6),
    torch.nn.Linear(6, 5),
    torch.nn.Linear(5, 4),
    torch.nn.Linear(4, 3),
    torch.nn.Linear(3, 2),
    torch.nn.Linear(2, 1),
    torch.nn.Flatten(0, 1))

optimizer5 = optim.SGD(model5.parameters(), lr=learning_rate)

train(1000, model5, optimizer5)

y_pred = model5(X_test)

plt.title("Resultats visuals 9")
plt.scatter(y_test, y_pred.detach().numpy(), c="red")
plt.plot(y_test, y_test)
plt.show()