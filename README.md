# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/user-attachments/assets/30d42013-4d6c-4b87-bdbd-05e5e97ca08f)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: K KESAVA SAI
### Register Number: 212223230105
```python
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1, 10)
    self.fc2 = nn.Linear(10, 15)
    self.fc3 = nn.Linear(15,25)
    self.fc4 = nn.Linear(25,45)
    self.fc5 = nn.Linear(45,60)
    self.fc6 = nn.Linear(60,85)
    self.fc7 = nn.Linear(85,1)
    self.relu = nn.ReLU()
    self.history = {'loss' : []}
  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.relu(self.fc4(x))
    x = self.relu(self.fc5(x))
    x = self.relu(self.fc6(x))
    x = self.fc7(x)
    return x
# Initialize the Model, Loss Function, and Optimizer
my_nn = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(my_nn.parameters(), lr=0.01)

def train_model(my_nn, X_train, y_train, criterion, optimizer, epochs=3000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(my_nn(X_train),y_train)
    loss.backward()
    optimizer.step()
    my_nn.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information

![image](https://github.com/user-attachments/assets/3721fa1b-c030-4e3e-831a-8469d1d2902f)


## OUTPUT
![image](https://github.com/user-attachments/assets/6907b439-0faa-4553-be83-62e94deafd63)

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/20e68fac-3771-4d32-9ada-163a25ddb39f)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/a6ee889b-6867-4106-8398-d440f6518c3b)


## RESULT

Successfully executed the code to develop a neural network regression model.
