import torch
import torch.nn as nn


class DisasterTweetClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx=0):
        super().__init__()
        # Define layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass for disaster tweet classification.
        
        Args:
            x: Input tensor of word indices, shape [batch_size, seq_length]
            
        Returns:
            Output tensor of logits, shape [batch_size, 1]
        """        
        # Step 1: Embedding
        x = self.embedding(x)
        # Step 2: Pooling (mean, max, or sum)
        pooled = x.mean(dim=1)
        # Step 3: Hidden layer(s) with activation
        x = self.fc1(pooled)
        x = self.relu(x)
        # Step 4: Dropout
        x = self.dropout(x)
        # Step 5: Output layer
        output = self.fc2(x)
        return output  # shape: [batch_size, 1]
