+++
title = '02. PyTorch Classification Exercises'
date = 2024-11-21T15:13:03-05:00
draft = false
summary = "This notebook is a exercise notebook from 'https://www.learnpytorch.io/02_pytorch_classification/#1-make-classification-data-and-get-it-ready'.."
series = ["AI",]
tags = ["AI", "Pytorch", "Machine Learning","Classification", "Deep Learning", "Neural Networks", "Artificial Intelligence"]
author= ["Me"]
+++

<a href="https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/02_pytorch_classification_exercises.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The following is a template for 02. PyTorch Classification exercises.

It's only starter code and it's your job to fill in the blanks.

Because of the flexibility of PyTorch, there may be more than one way to answer the question.

Don't worry about trying to be *right* just try writing code that suffices the question.

## Resources
* These exercises are based on [notebook 02 of the learn PyTorch course](https://www.learnpytorch.io/02_pytorch_classification/).
* You can see one form of [solutions on GitHub](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/extras/solutions) (but try the exercises below yourself first!).


```python
# Import torch
import torch

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Setup random seed
RANDOM_SEED = 42
```

    cuda


## 1. Make a binary classification dataset with Scikit-Learn's [`make_moons()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) function.
  * For consistency, the dataset should have 1000 samples and a `random_state=42`.
  * Turn the data into PyTorch tensors. 
  * Split the data into training and test sets using `train_test_split` with 80% training and 20% testing.


```python
# Create a dataset with Scikit-Learn's make_moons()
from sklearn.datasets import make_moons

SAMPLES = 1000
RANDOM_STATE = 42

x_data, y_data = make_moons(SAMPLES, random_state=RANDOM_STATE,noise=0.07)

print(f"X Length = {len(x_data)}, Shape = {x_data.shape} || Y Length = {len(y_data)}, Shape = {y_data.shape}")
```

    X Length = 1000, Shape = (1000, 2) || Y Length = 1000, Shape = (1000,)



```python
# Turn data into a DataFrame
import pandas as pd

data = {
    'X1': x_data[:, 0],
    'X2': x_data[:, 1],
    'label': y_data,
}
circles = pd.DataFrame(data)
print(circles.head(10))
circles.label.value_counts()
```

             X1        X2  label
    0 -0.033411  0.421391      1
    1  0.998827 -0.442890      1
    2  0.889592 -0.327843      1
    3  0.341958 -0.417690      1
    4 -0.838531  0.532375      0
    5  0.599064 -0.289773      1
    6  0.290090 -0.204688      1
    7 -0.038269  0.459429      1
    8  1.613771 -0.293970      1
    9  0.693337  0.827819      0





    label
    1    500
    0    500
    Name: count, dtype: int64




```python
# Visualize the data on a scatter plot
import matplotlib.pyplot as plt

plt.scatter(x=x_data[:, 0], 
            y=x_data[:, 1], 
            c=y_data, 
            cmap=plt.cm.RdYlBu);

```


    
![png](output_6_0.png)
    



```python
# Turn data into tensors of dtype float
X = torch.tensor(x_data, dtype=torch.float)
y = torch.tensor(y_data, dtype=torch.float)
```


```python
# Split the data into train and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=RANDOM_SEED)

len(X_train), len(X_test), len(y_train), len(y_test)
```




    (800, 200, 800, 200)



## 2. Build a model by subclassing `nn.Module` that incorporates non-linear activation functions and is capable of fitting the data you created in 1.
  * Feel free to use any combination of PyTorch layers (linear and non-linear) you want.


```python
import torch
from torch import nn

class MoonModelV0(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        super().__init__()
        
        self.layer1 = nn.Linear(in_features=in_features, 
                                 out_features=hidden_units)
        self.layer2 = nn.Linear(in_features=hidden_units, 
                                 out_features=hidden_units)
        self.layer3 = nn.Linear(in_features=hidden_units,
                                out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

model_0 = MoonModelV0(in_features=2,
                      out_features=1,
                      hidden_units=10).to(device)
model_0
```




    MoonModelV0(
      (layer1): Linear(in_features=2, out_features=10, bias=True)
      (layer2): Linear(in_features=10, out_features=10, bias=True)
      (layer3): Linear(in_features=10, out_features=1, bias=True)
      (relu): ReLU()
    )



## 3. Setup a binary classification compatible loss function and optimizer to use when training the model built in 2.


```python
# Setup loss function
loss_fn = nn.BCEWithLogitsLoss()

# Setup optimizer to optimize model's parameters
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)
```

## 4. Create a training and testing loop to fit the model you created in 2 to the data you created in 1.
  * Do a forward pass of the model to see what's coming out in the form of logits, prediction probabilities and labels.
  * To measure model accuray, you can create your own accuracy function or use the accuracy function in [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/).
  * Train the model for long enough for it to reach over 96% accuracy.
  * The training loop should output progress every 10 epochs of the model's training and test set loss and accuracy.


```python
# What's coming out of our model?
y_logits = model_0(X_train.to(device)[:10]).squeeze()

# logits (raw outputs of model)
print("Logits:", y_logits)
## Your code here ##

y_pred_probs = torch.sigmoid(y_logits)
# Prediction probabilities
print("Pred probs:", y_pred_probs)
## Your code here ##

# Prediction labels
y_preds = torch.round(y_pred_probs)
print("Pred labels:", y_preds)
## Your code here ##
```

    Logits: tensor([0.0019, 0.0094, 0.0161, 0.0185, 0.0284, 0.0192, 0.0291, 0.0196, 0.0258,
            0.0079], device='cuda:0', grad_fn=<SqueezeBackward0>)
    Pred probs: tensor([0.5005, 0.5024, 0.5040, 0.5046, 0.5071, 0.5048, 0.5073, 0.5049, 0.5065,
            0.5020], device='cuda:0', grad_fn=<SigmoidBackward0>)
    Pred labels: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0',
           grad_fn=<RoundBackward0>)



```python
!pip -q install torchmetrics # Colab doesn't come with torchmetrics
```


```python
# Let's calculuate the accuracy using accuracy from TorchMetrics
from torchmetrics import Accuracy

acc_fn = Accuracy(task="multiclass", num_classes=2).to(device) # send accuracy function to device
acc_fn
```




    MulticlassAccuracy()




```python
torch.manual_seed(RANDOM_SEED)

# Setup epochs
epochs = 1000


# Send data to the device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# Loop through the data
for epoch in range(epochs):
  ### Training
    model_0.train()

  # 1. Forward pass (logits output)
    y_logits = model_0(X_train).squeeze()
  
  # Turn logits into prediction probabilities
    y_pred_probs = torch.sigmoid(y_logits)

  
  # Turn prediction probabilities into prediction labels
    y_pred = torch.round(y_pred_probs)


  # 2. Calculaute the loss
    loss = loss_fn(y_logits, y_train) # loss = compare model raw outputs to desired model outputs

  # Calculate the accuracy
    acc = acc_fn(y_pred, y_train.int()) # the accuracy function needs to compare pred labels (not logits) with actual labels

  # 3. Zero the gradients
    optimizer.zero_grad()
  

  # 4. Loss backward (perform backpropagation) - https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%20propagation,to%20the%20neural%20network's%20weights.
    loss.backward()

  # 5. Step the optimizer (gradient descent) - https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21#:~:text=Gradient%20descent%20(GD)%20is%20an,e.g.%20in%20a%20linear%20regression) 
    optimizer.step()

  ### Testing
    model_0.eval() 
    with torch.inference_mode():
    # 1. Forward pass (to get the logits)
        test_logits = model_0(X_test).squeeze()
        
        # Turn the test logits into prediction labels
        y_pred_probs = torch.sigmoid(test_logits)
        test_pred = torch.round(y_pred_probs)
        
        # 2. Caculate the test loss/acc
        test_loss = loss_fn(test_logits, y_test)    
        test_acc = acc_fn(test_pred, y_test.int())

    

  # Print out what's happening every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc*100:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc*100:.2f}%")
    
```

    Epoch: 0 | Loss: 0.69532, Accuracy: 37.38% | Test loss: 0.69446, Test acc: 49.50%
    Epoch: 100 | Loss: 0.38865, Accuracy: 81.62% | Test loss: 0.40304, Test acc: 76.00%
    Epoch: 200 | Loss: 0.24289, Accuracy: 88.50% | Test loss: 0.24102, Test acc: 89.50%
    Epoch: 300 | Loss: 0.20105, Accuracy: 90.88% | Test loss: 0.19153, Test acc: 93.50%
    Epoch: 400 | Loss: 0.16584, Accuracy: 92.88% | Test loss: 0.15421, Test acc: 94.00%
    Epoch: 500 | Loss: 0.12358, Accuracy: 95.12% | Test loss: 0.11175, Test acc: 96.00%
    Epoch: 600 | Loss: 0.08452, Accuracy: 97.88% | Test loss: 0.07401, Test acc: 98.50%
    Epoch: 700 | Loss: 0.05662, Accuracy: 99.00% | Test loss: 0.04811, Test acc: 99.50%
    Epoch: 800 | Loss: 0.04003, Accuracy: 99.25% | Test loss: 0.03284, Test acc: 99.50%
    Epoch: 900 | Loss: 0.02999, Accuracy: 99.75% | Test loss: 0.02394, Test acc: 100.00%


## 5. Make predictions with your trained model and plot them using the `plot_decision_boundary()` function created in this notebook.


```python
# Plot the model predictions
import numpy as np

def plot_decision_boundary(model, X, y):
  
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/ 
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), 
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # mutli-class
    else: 
        y_pred = torch.round(torch.sigmoid(y_logits)) # binary
    
    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
```


```python
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
```


    
![png](output_20_0.png)
    


## 6. Replicate the Tanh (hyperbolic tangent) activation function in pure PyTorch.
  * Feel free to reference the [ML cheatsheet website](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh) for the formula.


```python
# Create a straight line tensor
tensor_line = torch.arange(-100, 100, 1)
plt.plot(tensor_line)
```




    [<matplotlib.lines.Line2D at 0x7f6c38b45ba0>]




    
![png](output_22_1.png)
    



```python
# Test torch.tanh() on the tensor and plot it
plt.plot(torch.tanh(tensor_line))

```




    [<matplotlib.lines.Line2D at 0x7f6c389af490>]




    
![png](output_23_1.png)
    



```python
# Replicate torch.tanh() and plot it
def tanh(x):
  return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

plt.plot(tanh(tensor_line))
```




    [<matplotlib.lines.Line2D at 0x7f6c38a513f0>]




    
![png](output_24_1.png)
    


## 7. Create a multi-class dataset using the [spirals data creation function from CS231n](https://cs231n.github.io/neural-networks-case-study/) (see below for the code).
  * Split the data into training and test sets (80% train, 20% test) as well as turn it into PyTorch tensors.
  * Construct a model capable of fitting the data (you may need a combination of linear and non-linear layers).
  * Build a loss function and optimizer capable of handling multi-class data (optional extension: use the Adam optimizer instead of SGD, you may have to experiment with different values of the learning rate to get it working).
  * Make a training and testing loop for the multi-class data and train a model on it to reach over 95% testing accuracy (you can use any accuracy measuring function here that you like) - 1000 epochs should be plenty.
  * Plot the decision boundaries on the spirals dataset from your model predictions, the `plot_decision_boundary()` function should work for this dataset too.


```python
# Code for creating a spiral dataset from CS231n
import numpy as np
import matplotlib.pyplot as plt
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()
```


    
![png](output_26_0.png)
    



```python
# Turn data into tensors
import torch
X = torch.from_numpy(X).type(torch.float) # features as float32
y = torch.from_numpy(y).type(torch.LongTensor) # labels need to be of type long

# Create train and test splits
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

len(X_train), len(X_test), len(y_train), len(y_test)
```




    (240, 60, 240, 60)




```python
# Let's calculuate the accuracy for when we fit our model
!pip -q install torchmetrics # colab doesn't come with torchmetrics
from torchmetrics import Accuracy

acc_fn = Accuracy(task="multiclass", num_classes=K).to(device)
acc_fn
```




    MulticlassAccuracy()




```python
# Prepare device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create model by subclassing nn.Module
class SpiralModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=10)
        self.linear2 = nn.Linear(in_features=10, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=K)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))
        

# Instantiate model and send it to device
model_1 = SpiralModel().to(device)
model_1

```




    SpiralModel(
      (linear1): Linear(in_features=2, out_features=10, bias=True)
      (linear2): Linear(in_features=10, out_features=10, bias=True)
      (linear3): Linear(in_features=10, out_features=3, bias=True)
      (relu): ReLU()
    )




```python
# Setup data to be device agnostic
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
print(X_train.dtype, X_test.dtype, y_train.dtype, y_test.dtype)

# Print out first 10 untrained model outputs (forward pass)
logits = model_1(X_train[:10])
print("Logits:", logits)

y_pred_probs = torch.softmax(model_1(X_train)[:10], dim=1)
print("Pred probs:", y_pred_probs)


print("Pred labels:", y_pred_probs.argmax(dim=1))
```

    torch.float32 torch.float32 torch.int64 torch.int64
    Logits: tensor([[-0.2160, -0.0600,  0.2256],
            [-0.2020, -0.0530,  0.2257],
            [-0.2223, -0.0604,  0.2384],
            [-0.2174, -0.0555,  0.2826],
            [-0.2201, -0.0502,  0.2792],
            [-0.2195, -0.0565,  0.2457],
            [-0.2212, -0.0581,  0.2440],
            [-0.2251, -0.0631,  0.2354],
            [-0.2116, -0.0548,  0.2336],
            [-0.2170, -0.0552,  0.2842]], device='cuda:0',
           grad_fn=<AddmmBackward0>)
    Pred probs: tensor([[0.2685, 0.3139, 0.4176],
            [0.2707, 0.3142, 0.4151],
            [0.2659, 0.3126, 0.4215],
            [0.2615, 0.3074, 0.4311],
            [0.2609, 0.3092, 0.4299],
            [0.2653, 0.3123, 0.4224],
            [0.2653, 0.3123, 0.4224],
            [0.2659, 0.3127, 0.4214],
            [0.2681, 0.3136, 0.4184],
            [0.2614, 0.3072, 0.4314]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
    Pred labels: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], device='cuda:0')



```python
# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.02)
```


```python
# Build a training loop for the model
epochs = 1000

# Loop over data
for epoch in range(epochs):
    
    ## Training
    model_1.train()
    
    # 1. Forward pass
    y_logits = model_1(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    
    # 2. Calculate the loss
    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train)
    
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    
    
    # 4. Loss backward
    loss.backward()
    
    
    # 5. Optimizer step
    optimizer.step()
    
    
    ## Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        
        # 2. Caculate loss and acc
        test_loss = loss_fn(test_logits, y_test)
        test_acc = acc_fn(test_pred, y_test)

    # Print out what's happening every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f} Acc: {acc:.2f} | Test loss: {test_loss:.2f} Test acc: {test_acc:.2f}")
  
```

    Epoch: 0 | Loss: 1.11 Acc: 0.32 | Test loss: 1.10 Test acc: 0.37
    Epoch: 100 | Loss: 0.45 Acc: 0.78 | Test loss: 0.53 Test acc: 0.68
    Epoch: 200 | Loss: 0.11 Acc: 0.97 | Test loss: 0.09 Test acc: 0.98
    Epoch: 300 | Loss: 0.07 Acc: 0.98 | Test loss: 0.02 Test acc: 1.00
    Epoch: 400 | Loss: 0.05 Acc: 0.98 | Test loss: 0.01 Test acc: 1.00
    Epoch: 500 | Loss: 0.04 Acc: 0.99 | Test loss: 0.01 Test acc: 1.00
    Epoch: 600 | Loss: 0.03 Acc: 0.99 | Test loss: 0.01 Test acc: 1.00
    Epoch: 700 | Loss: 0.03 Acc: 0.99 | Test loss: 0.00 Test acc: 1.00
    Epoch: 800 | Loss: 0.02 Acc: 0.99 | Test loss: 0.00 Test acc: 1.00
    Epoch: 900 | Loss: 0.02 Acc: 0.99 | Test loss: 0.00 Test acc: 1.00



```python
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
```


    
![png](output_33_0.png)
    

