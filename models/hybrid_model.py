import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class HybridDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden_size=128, lstm_num_layers=1, dropout_rate=0.3):
        super(HybridDeepfakeDetector, self).__init__()
        
        # Use smaller CNN Backbone
        self.cnn_backbone = models.efficientnet_b0(weights='DEFAULT')
        # Remove the classification head
        self.cnn_features = nn.Sequential(*list(self.cnn_backbone.children())[:-2])
        
        # Freeze more layers to save memory
        for param in self.cnn_features.parameters():
            param.requires_grad = False
        
        # Get the output features from CNN
        self.cnn_output_features = 1280  # For EfficientNet-B0
        
        # Smaller LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,  # Disable bidirectional to save memory
            dropout=0.0  # No dropout for LSTM to save memory
        )
        
        # Simpler Classification Head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        # REMOVED SIGMOID - will use BCEWithLogitsLoss instead

    def forward(self, x, sequence_lengths=None):
        batch_size, seq_len, C, H, W = x.size()
        
        # Process each frame through CNN
        cnn_features = []
        for t in range(seq_len):
            with torch.no_grad():  # No gradients for CNN to save memory
                frame_features = self.cnn_features(x[:, t, :, :, :])
                frame_features = torch.mean(frame_features, dim=[2, 3])
            cnn_features.append(frame_features)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM processing
        if sequence_lengths is not None:
            packed_input = pack_padded_sequence(cnn_features, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_input)
            lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
            
            last_outputs = []
            for i, length in enumerate(sequence_lengths):
                last_outputs.append(lstm_output[i, length-1, :])
            lstm_features = torch.stack(last_outputs, dim=0)
        else:
            lstm_output, (hidden, cell) = self.lstm(cnn_features)
            lstm_features = lstm_output[:, -1, :]
        
        # Classification
        x = self.dropout(lstm_features)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Raw logits, no sigmoid
        
        return x

class SimpleCNNModel(nn.Module):
    """Simpler model for testing - Use this if hybrid still fails"""
    def __init__(self, num_classes=1):
        super(SimpleCNNModel, self).__init__()
        
        self.cnn_backbone = models.efficientnet_b0(weights='DEFAULT')
        num_features = self.cnn_backbone.classifier[1].in_features
        
        # Replace classifier - REMOVE FINAL SIGMOID
        self.cnn_backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
            # REMOVED SIGMOID - will use BCEWithLogitsLoss instead
        )
    
    def forward(self, x):
        # If input is sequence, use only first frame
        if len(x.shape) == 5:  # (batch, seq, C, H, W)
            x = x[:, 0, :, :, :]  # Take first frame only
        return self.cnn_backbone(x)