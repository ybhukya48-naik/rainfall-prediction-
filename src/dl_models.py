import torch
import torch.nn as nn
import numpy as np
import joblib
import os

class AutoencoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoencoderLSTM, self).__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        _, (hidden, _) = self.encoder_lstm(x)
        # Repeat hidden state for decoder
        # hidden shape: (1, batch, hidden_dim)
        repeat_hidden = hidden.permute(1, 0, 2).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder_lstm(repeat_hidden)
        return decoded, hidden.squeeze(0)

def train_autoencoder(data, input_dim, hidden_dim=16, epochs=50):
    model = AutoencoderLSTM(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Convert data to tensor
    # Assuming data is (samples, seq_len, features)
    X = torch.FloatTensor(data)
    
    print("Training Autoencoder-LSTM for feature extraction...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        decoded, _ = model(X)
        loss = criterion(decoded, X)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/autoencoder_lstm.pth')
    print("Autoencoder model saved.")
    return model

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use only the last time step
        last_out = lstm_out[:, -1, :]
        out = torch.sigmoid(self.fc(last_out))
        return out

def train_bilstm(X_train, y_train, input_dim, hidden_dim=32, epochs=50):
    model = BiLSTMModel(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    X = torch.FloatTensor(X_train)
    y = torch.FloatTensor(y_train).view(-1, 1)
    
    print("Training BiLSTM model...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), 'models/bilstm_model.pth')
    return model

def build_sequences(X, y, seq_len=7):
    sequences = []
    labels = []
    for i in range(len(X) - seq_len + 1):
        seq = X[i:i+seq_len]
        label = y[i+seq_len-1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def train_lstm_timeseries(data_path='data/advanced_weather_data.csv', hidden_dim=32, epochs=30, seq_len=7):
    from advanced_data_utils import load_data, preprocess_data, generate_advanced_dummy_data
    if not os.path.exists(data_path):
        generate_advanced_dummy_data(data_path)
    df = load_data(data_path)
    X, y, features = preprocess_data(df)
    X_seq, y_seq = build_sequences(X, y.values if hasattr(y, 'values') else y, seq_len=seq_len)
    model = BiLSTMModel(X_seq.shape[2], hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq).view(-1, 1)
    print("Training Time Series LSTM model...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/timeseries_lstm.pth')
    joblib.dump({'seq_len': seq_len, 'input_dim': X_seq.shape[2]}, 'models/timeseries_lstm_meta.pkl')
    return model
