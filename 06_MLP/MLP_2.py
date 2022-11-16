import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

etiquetes = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

class Net(nn.Module):
    def __init__(self, n_original, n_out):
        super(Net, self).__init__()

        # les capes RELU ajuden a rompre la linealitat
        self.c1 = nn.Linear(n_original, n_original // 2)
        self.c2 = nn.Linear(n_original // 2, n_original // 4)
        self.c3 = nn.Linear(n_original // 4, n_original // 8)
        self.c4 = nn.Linear(n_original // 8, n_original // 16)
        self.c5 = nn.Linear(n_original // 16, n_out)

    def forward(self, x):
        y = torch.flatten(x, 1)
        x = self.c1(y)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        output = F.softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch, log_interval=100, verbose=True):
    model.train()  # Posam la xarxa en mode entrenament

    loss_v = 0  # Per calcular la mitjana (és la vostra)

    # Bucle per entrenar cada un dels batch
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target, reduction='sum')
        loss.backward()
        optimizer.step()

        ## Informació de debug
        if batch_idx % log_interval == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Average: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss.item() / len(data)))
        loss_v += loss.item()

    loss_v /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}\n'.format(loss_v))

    return loss_v


def test(modeltest, device, test_loader):
    modeltest.eval()  # Posam la xarxa en mode avaluació

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = modeltest(data)
            #test_loss += loss(output, target)
            test_loss += F.cross_entropy(output, target, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)  # index amb la max probabilitat
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Informació de debug
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss

# Definim una seqüència (composició) de transformacions
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # mitjana, desviacio tipica (precalculats)
    ])

# Descarregam un dataset ja integrat en la llibreria Pytorch
train_data = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST('../data', train=False, transform=transform)

n_original_features = train_data.data[0].shape[0] * train_data.data[0].shape[1]

train_batch_size = 64
test_batch_size = 100

# Transformam les dades en l'estructura necessaria per entrenar una xarxa
train_loader = torch.utils.data.DataLoader(train_data, train_batch_size)
test_loader = torch.utils.data.DataLoader(test_data, test_batch_size)

iterador =  iter(train_loader) # Un iterador!!

features, labels = next(iterador)

# Extra: mostrar una graella amb tot el batch·

print("_"*50)
print(f"Feature batch shape: {features.size()}")
print(f"Labels batch shape: {labels.size()}")

torch.manual_seed(33)

# Ens permet emprar l'entorn de cuda. El podem activar a "Entorno de ejecución"
use_cuda = False
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Paràmetres bàsics
epochs = 4
lr =10e-4

n_out = len(etiquetes.keys())

model1 = Net(n_original_features, n_out).to(device)

# Stochastic gradient descent
optimizer = optim.SGD(model1.parameters(), lr=lr) #momentum

# Guardam el valor de pèrdua mig de cada època, per fer el gràfic final
train_l = np.zeros((epochs))
test_l = np.zeros((epochs))

# Bucle d'entrenament
for epoch in range(0, epochs):
    train_l[epoch] = train(model1, device, train_loader, optimizer, epoch)
    test_l[epoch] = test(modeltest=model1, device=device, test_loader=test_loader)
    #test_l[epoch] = test(model1, device, test_loader)

plt.title("Resultats de l'entrenament")
plt.plot(range(1, (epochs + 1)), train_l,  c="red", label="train")
plt.plot(range(1,  (epochs + 1)), test_l,  c="green", label="test")
plt.legend();