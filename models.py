import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, seq_length):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        
        self.flatten_size = 256 * (seq_length // 8)
        
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 2)  # Two categories
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: [batch, 1, seq_len]
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 2)  # Bidirectional LSTM, so hidden_size*2
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch, 1, seq_len] -> [batch, seq_len, 1]
        x = x.permute(0, 2, 1)
        
        # LSTM output: [batch, seq_len, hidden_size*2]
        output, _ = self.lstm(x)
        

        output = output[:, -1, :]
        
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

class CNNLSTMModel(nn.Module):
    def __init__(self, seq_length, hidden_size=128):
        super(CNNLSTMModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        

        self.fc = nn.Linear(hidden_size * 2, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: [batch, 1, seq_len]
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # [batch, seq_len/4, 64]
        x = x.permute(0, 2, 1)
        
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        
        return output