from sklearn.metrics import mean_squared_error
import backtrader as bt
from datetime import datetime
from ib_insync import *
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import matplotlib

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    # 3 input features: year, day_of_year, and open GO up to 2048
        self.fc1 = nn.Linear(6, 128)  # Input layer
        self.fc2 = nn.Linear(128, 256)  # Hidden layer
        self.fc3 = nn.Linear(256, 512)  # Hidden layer
        self.fc4 = nn.Linear(512, 1024)  # Additional hidden layer
        self.fc5 = nn.Linear(1024, 2048)  # Additional hidden layer
        self.fc6 = nn.Linear(2048, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 256)
        self.fc9 = nn.Linear(256, 128)
        self.fc10 = nn.Linear(128, 64)
        self.fc11 = nn.Linear(64, 3)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        
        x = self.fc11(x)
        
        
        return x


def TrainNewModel(csv_file, modelId):
    print("====================================")
    print("Loading historical data...")
    df = pd.read_csv(f"./{csv_file}")
    
    #Date,Symbol,Open,High,Low,Close,Volume ETH,Volume USD
    cuda_available = torch.cuda.is_available()
    print("CUDA Available:", cuda_available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Convert 'date' column to datetime (ensure your CSV's date format is recognized by pandas)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)

    df.dropna(inplace=True)

    # save
    # Prepare input and output data
    X = df[['year', 'day_of_year', 'open',
            'prev_close', 'prev_high', 'prev_low']].values
    y = df[['high', 'low', "close"]].values

    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    # save the scaler
    scaler_X_path = (f"./{modelId}_scaler_X.pkl")
    scaler_y_path = (f"./{modelId}_scaler_y.pkl")
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Define the model

    model = Net().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.9)
    print("====================================")
    print("Training the model...")
# Train the model
    for epoch in range(2000):  # number of epochs
        inputs, labels = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()

        if epoch % 100 == 99:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    print("====================================")
# Test the model
    print("Testing the model...")
    print("====================================")
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        X_test = X_test.to(device)
        predictions = model(X_test)
        predictions_cpu = predictions.cpu().numpy()
        predictions = scaler_y.inverse_transform(predictions_cpu)  # Scale back the predictions
        # Ensure y_test is on CPU and in NumPy format for comparison
        y_test_cpu = y_test.cpu().numpy()
        y_test = scaler_y.inverse_transform(y_test_cpu)
        for i in range(10):
            print(f'Predicted: High: {predictions[i][0]}, Low: {predictions[i][1]}, Close: {predictions[i][2]}')
            print(f'Actual: High: {y_test[i][0]}, Low: {y_test[i][1]}, Close: {y_test[i][2]}')

    print("====================================")

# Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')


# Save the model
    torch.save(model.state_dict(), f"./{modelId}_model.pth")


def predict(date_str, open_price, previous_close,  previous_high, previous_low, modelId):
    modelPath = "./" + modelId + "_model.pth"

    scaler_X_path = (f"./{modelId}_scaler_X.pkl")
    scaler_y_path = (f"./{modelId}_scaler_y.pkl")

    # Load the scalers
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    # Load the model
    model = Net()
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    # Convert the input data to the right format
    date = pd.to_datetime(date_str, utc=True)
    year = date.year
    day_of_year = date.dayofyear

    input_features = [[year, day_of_year, open_price,
                       previous_close, previous_high, previous_low]]

    input_features_scaled = scaler_X.transform(input_features)

    X = torch.tensor(input_features_scaled, dtype=torch.float32).unsqueeze(0)

    # Make the prediction

    with torch.no_grad():
        prediction = model(X)
        # Remove the batch dimension and convert to numpy
        prediction = prediction.squeeze().numpy()
        # Now prediction is 2D and can be inverse transformed
        prediction = scaler_y.inverse_transform([prediction])

        return prediction[0][0], prediction[0][1], prediction[0][2]


def __main__():
    modelId = 'eth'
    TrainNewModel('ETH_1min.csv', modelId)
    
    # Predict the next 10 days
    """  start_date = '2024-04-04'
    open_price = 170.29
    previous_close = 169.65
    plot = []
    for i in range(1000):
        high, low, close = predict(start_date, open_price, previous_close, previous_close, previous_close, modelName)
        print(f'Predicted: High: {high}, Low: {low}, Close: {close}')
        start_date = pd.to_datetime(start_date, utc=True) + pd.DateOffset(days=1)
        start_date = start_date.strftime('%Y-%m-%d')
        open_price = close
        previous_close = close
        plot.append([start_date, high, low, close])
    
    df = pd.DataFrame(plot, columns=['date', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.plot()
    
    # display the plot
    matplotlib.pyplot.show() """

__main__()
