import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score


'''
This class implements the DatasetBatchIterator class to iterate over the dataset in batches,
which shuffle the batches and return the batches in the form of tensors.

    - X: numpy array, the input features
    - Y: numpy array, the target numerical values
    - batch_size: integer, the number of samples to include in each batch
    - shuffle: boolean, whether to shuffle the batches or not

Note: This class assumes that the input features and target numerical values are numpy arrays.
'''
class DatasetBatchIterator:
    def __init__(self, X, Y, batch_size, shuffle=True, random_state=42):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        np.random.seed(random_state)
        if shuffle:
            # Set random permutation matrix to shuffle the batches
            index = np.random.permutation(X.shape[0])
            X = self.X[index]
            Y = self.Y[index]
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(X.shape[0] / batch_size))
        self._current = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        X_batch = torch.FloatTensor(self.X[k*bs:(k+1)*bs])
        Y_batch = torch.FloatTensor(self.Y[k*bs:(k+1)*bs])
        return X_batch, Y_batch.view(-1, 1)

'''
The class implements the Neural Network class, which is a neural network model that 
learns the numerical features based on their interactions in the dataset.
The model is trained using the Huber loss function and the Adam optimizer.
It's a subclass of nn.Module, which is a PyTorch's class that represents a neural network model.

    - random_state: integer, the seed for reproducibility
    - hidden_layers: tuple of integers, the number of neurons in each hidden layer of the MLP
    - dropout_rate: float, the probability that an unit of a layer will be closed
    - output_range: tuple of integers, the range of the output values
    - lr: float, the learning rate of the Adam optimizer
    - wd: float, the weight decay of the Adam optimizer
    - max_epochs: integer, the maximum number of epochs to train the model
    - early_stop_epoch_threshold: integer, the number of epochs to wait for the model's performance to improve before stopping the training process
    - batch_size: integer, the number of samples to include in each batch   
'''
class NumericalFeatureNeuralNetwork(nn.Module):
    def __init__(self, random_state=42, hidden_layers=(128,64,32,16), dropout_rate=0.2, output_range=(0,1), 
                 lr= 1e-3, wd=1e-4, max_epochs=50, early_stop_epoch_threshold=5, batch_size=64):
        super().__init__()
        
        # Set the seed for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # Set the parameters
        self.random_state = random_state
        self.output_range = output_range
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        self.early_stop_epoch_threshold = early_stop_epoch_threshold
        self.batch_size = batch_size

        ## Initialize the GPU device to compute
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.dropout_rate = dropout_rate
        self.hidden_layers = hidden_layers

        # The probability that an unit of a layer will be closed
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize output normalization parameters
        assert output_range and len(output_range) == 2, "output range has to be tuple with two integers"
        self.norm_min = min(output_range)
        self.norm_range = abs(output_range[0]-output_range[1])
        

    '''
    This function generates a MLP with the given parameters.

        - input_size: integer, the number of input features
        - hidden_layers_units: tuple of integers, the number of neurons in each hidden layer of the MLP
        - dropout_rate: float, the probability that an unit of a layer will be closed
    '''
    def __genMLP(self, input_size, hidden_layers_units, dropout_rate):
        torch.manual_seed(self.random_state)
        hidden_layers = []
        input_units = input_size

        # Construct the connection between layers
        for num_units in hidden_layers_units:
            # Connect the inputs_units layers with num_units layers linealy
            hidden_layers.append(nn.Linear(input_units, num_units, device=self.device))
            # Normalize the before layer's activities to stabilize and accelerate training
            hidden_layers.append(nn.BatchNorm1d(num_units, device=self.device))
            # Learn the relation between no connect layers
            hidden_layers.append(nn.ReLU())

            if dropout_rate:
                # Dropout layer to reduce overfitting, if exist dropout_rate
                hidden_layers.append(nn.Dropout(dropout_rate))
            input_units = num_units

        # Conect the last layers with the output
        hidden_layers.append(nn.Linear(hidden_layers_units[-1], 1, device=self.device))
        # Reproduce output between 0 and 1 
        hidden_layers.append(nn.Sigmoid())
        
        return nn.Sequential(*hidden_layers)
    

    '''
    This function initializes the random parameters of the model, this parameters will be update
    in training process, in this case initialize the weights of the neural network using Kaiming
    uniform
    '''
    def initParams(self):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        def weights_init(m):
            if type(m) == nn.Linear:
                '''
                Initialize the weights of a neural network layer using Kaiming uniform
                The Kaiming initialization method aims to set the initial weights in such 
                a way that the variance of the outputs of each layer remains approximately
                the same during forward and backward propagation.
                The bias help neural network to learn patrons more complex and fitting
                well the inputs
                '''
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.fill_(0.01)

        ## Each layer of MLP will initialize according this function 'weights_init'
        self.MLP.apply(weights_init)

    '''
    This function compute the forward pass of the model, which is the computation of the output of the model
    given the input x, contains the features of the dataset.

    The function will be auto invoked by PyTorch when we call the model.
    '''
    def forward(self, x):
        # Ensure x_batch is on the correct device
        x = x.to(device=self.device)

        # Pass the input through the MLP
        y = self.MLP(x)

        ## Normalize the output between 0 and 1
        normalized_output = y * self.norm_range + self.norm_min
        return normalized_output

    
    '''
    The training function of the model, which will update the parameters of the model using
    the Huber loss function and the Adam optimizer, in this case update the weights and biases
    of each layer of the MLP.

        - X_train: numpy array, the input features of the training dataset
        - y_train: numpy array, the target numerical values of the training dataset
    '''
    def fit(self, X_train, y_train):
        # self.initParams()
        self.train()
        num_features = X_train.shape[1]
        self.MLP = self.__genMLP(num_features, self.hidden_layers, self.dropout_rate)
        self.initParams()

        # Training loop control parameters
        no_loss_reduction_epoch_counter = 0
        min_loss = np.inf
        best_model_state = None

        # Use GPU to run the Neural Network
        self.to(self.device)

        # Loss function: Huber Error -> combination of MAE and MSE
        beta = 0.5
        loss_criterion = nn.SmoothL1Loss(reduction='sum', beta=beta)
        # Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        # Training of the model
        print('Start the NFNN MODEL training .....')
        
        # Each epoch update the parameters of the model
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            # Iterate over the batches of the dataset
            for x_batch, y_batch in DatasetBatchIterator(X_train, y_train, batch_size=self.batch_size, random_state=self.random_state):
                x_batch, y_batch = x_batch.to(device=self.device), y_batch.to(device=self.device)

                # Zero the gradients of the model's parameters
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self(x_batch)
                    # Compute the loss of the model's output with respect to the true labels
                    loss = loss_criterion(outputs, y_batch)
                    # Backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # Update the parameters of the model using the gradients
                    optimizer.step()
                epoch_loss += loss.item()
        
            epoch_loss = epoch_loss / len(X_train)
            # print(f'Epoch: {epoch+1}, Loss: {epoch_loss}')

            # Check the early stop condition of the model
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                no_loss_reduction_epoch_counter = 0
                best_model_state = self.state_dict()
            else:
                no_loss_reduction_epoch_counter += 1
            
            # If the model has not improved for the early_stop_epoch_threshold number of epochs stop the training process
            if no_loss_reduction_epoch_counter >= self.early_stop_epoch_threshold:
                print(f'Early stop at epoch {epoch+1}')
                break

        if best_model_state:
            self.load_state_dict(best_model_state)
        print(f'The NFNN Model has been trained successfully with final loss: {epoch_loss:.4f}')
        return self

    '''
    The evaluation function of the model, which will evaluate the model's accuracy using train dataset
    based on the R2 score metric.
    '''
    def score(self, X_train, y_train):
        self.eval()
        
        groud_truth, predictions = [], []

        with torch.no_grad():
            for x_batch, y_batch in DatasetBatchIterator(X_train, y_train, batch_size=self.batch_size, shuffle=False, random_state=self.random_state):
                x_batch, y_batch = x_batch.to(device=self.device), y_batch.to(device=self.device)
                # Get predictions from the model
                outputs = self(x_batch)
                groud_truth.append(y_batch.cpu().numpy())  # Collect true labels
                predictions.append(outputs.cpu().numpy())  # Collect predictions
        
        # Concatenate all predictions and true values
        groud_truth = np.concatenate(groud_truth)
        predictions = np.concatenate(predictions)

        # Using R2 score algorithm to evaluate the model's accuracy
        R2 = r2_score(groud_truth, predictions)
        return R2

    '''
    The prediction function of the model, which will predict the target variable for the given input data

    - X_test: numpy array, the input features of the test dataset
    '''
    def predict(self, X_test):
        self.eval()
        predictions = []
        with torch.no_grad():
            for x_batch, _ in DatasetBatchIterator(X_test, np.zeros((len(X_test), 1)), batch_size=self.batch_size, shuffle=False, random_state=self.random_state):
                x_batch = x_batch.to(device=self.device)
                # Get predictions from the model
                outputs = self(x_batch)
                predictions.append(outputs.cpu().numpy())  # Collect predictions
        # Concatenate all predictions
        predictions = np.concatenate(predictions)
    
        return predictions
    
    '''
    The function to get all the parameters of the model
    - deep: bool, whether to return the parameters for a deep copy of the object or not. Default is True.
    '''
    def get_params(self, deep=True):
        return {
            "random_state": self.random_state,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "output_range": self.output_range,
            "lr": self.lr,
            "wd": self.wd,
            "max_epochs": self.max_epochs,
            "early_stop_epoch_threshold": self.early_stop_epoch_threshold,
            "batch_size": self.batch_size
        }
    
    '''
    The function to set all the parameters of the model
    - params: dict, the parameters to set for the model.
    '''
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

