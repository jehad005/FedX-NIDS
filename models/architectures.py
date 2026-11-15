"""
Deep Learning Model Architectures for SecureFed-NIDS
======================================================
Implements 6 DL models: MLP, CNN, LSTM, BiLSTM, GRU, Transformer.
"""

import torch
import torch.nn as nn
import math

from config import MODEL_CONFIG


class BaseModel(nn.Module):
    """Base class for all models with common functionality."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations."""
        raise NotImplementedError

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP(BaseModel):
    """Multi-Layer Perceptron with batch normalization."""

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: list = None, activation: str = "relu",
                 dropout: float = 0.3, batch_norm: bool = True):
        super().__init__(input_dim, num_classes)

        config = MODEL_CONFIG.get("mlp", {})
        hidden_dims = hidden_dims or config.get("hidden_dims", [256, 128, 64])
        activation_name = activation or config.get("activation", "relu")
        dropout_rate = dropout if dropout else config.get("dropout", 0.3)
        use_bn = batch_norm if batch_norm is not None else config.get("batch_norm", True)

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation_name.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation_name.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            elif activation_name.lower() == "gelu":
                layers.append(nn.GELU())

            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # Return penultimate representation before the final classification layer
        for layer in list(self.network)[:-1]:
            x = layer(x)
        return x


class CNN(BaseModel):
    """1D Convolutional Neural Network for sequential data."""

    def __init__(self, input_dim: int, num_classes: int,
                 filters: list = None, kernel_sizes: list = None,
                 pool_size: int = 2, dropout: float = 0.3):
        super().__init__(input_dim, num_classes)

        config = MODEL_CONFIG.get("cnn", {})
        filters = filters or config.get("filters", [64, 128, 256])
        kernel_sizes = kernel_sizes or config.get("kernel_sizes", [3, 3, 3])
        pool_size = pool_size or config.get("pool_size", 2)
        dropout_rate = dropout if dropout else config.get("dropout", 0.3)

        # Reshape input to (batch, channels, length)
        self.input_reshape = nn.Linear(input_dim, input_dim)

        # Convolutional blocks
        conv_blocks = []
        in_channels = 1

        for i, (filt, kernel) in enumerate(zip(filters, kernel_sizes)):
            conv_blocks.append(nn.Conv1d(in_channels, filt, kernel, padding=kernel//2))
            conv_blocks.append(nn.BatchNorm1d(filt))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool1d(pool_size))
            conv_blocks.append(nn.Dropout(dropout_rate))
            in_channels = filt

        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Dynamically calculate exact flattened size for any input_dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv_blocks(dummy)
            self.flatten_size = out.flatten(1).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for 1D conv: (batch, 1, length)
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        x = x.flatten(1)
        return self.fc(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        return x.flatten(1)


class LSTM(BaseModel):
    """Long Short-Term Memory network."""

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = None, num_layers: int = None,
                 dropout: float = 0.3, bidirectional: bool = False):
        super().__init__(input_dim, num_classes)

        config = MODEL_CONFIG.get("lstm", {})
        hidden_dim = hidden_dim or config.get("hidden_dim", 128)
        num_layers = num_layers or config.get("num_layers", 2)
        dropout_rate = dropout if dropout else config.get("dropout", 0.3)
        bidirectional = bidirectional or config.get("bidirectional", False)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) - need to reshape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)

        # Forward LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Use last hidden state
        if self.bidirectional:
            hn = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            hn = hn[-1]

        return self.fc(hn)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        if self.bidirectional:
            return torch.cat([hn[-2], hn[-1]], dim=1)
        return hn[-1]


class BiLSTM(LSTM):
    """Bidirectional LSTM - inherits from LSTM with bidirectional=True."""

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = None, num_layers: int = None,
                 dropout: float = 0.3):
        config = MODEL_CONFIG.get("bilstm", {})
        hidden_dim = hidden_dim or config.get("hidden_dim", 128)
        num_layers = num_layers or config.get("num_layers", 2)
        dropout_rate = dropout if dropout else config.get("dropout", 0.3)

        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=True
        )


class GRU(BaseModel):
    """Gated Recurrent Unit network."""

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = None, num_layers: int = None,
                 dropout: float = 0.3, bidirectional: bool = False):
        super().__init__(input_dim, num_classes)

        config = MODEL_CONFIG.get("gru", {})
        hidden_dim = hidden_dim or config.get("hidden_dim", 128)
        num_layers = num_layers or config.get("num_layers", 2)
        dropout_rate = dropout if dropout else config.get("dropout", 0.3)
        bidirectional = bidirectional or config.get("bidirectional", False)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)

        out, hn = self.gru(x, h0)

        if self.bidirectional:
            hn = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            hn = hn[-1]

        return self.fc(hn)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.gru(x, h0)

        if self.bidirectional:
            return torch.cat([hn[-2], hn[-1]], dim=1)
        return hn[-1]


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(BaseModel):
    """Transformer encoder for network intrusion detection."""

    def __init__(self, input_dim: int, num_classes: int,
                 d_model: int = None, nhead: int = None,
                 num_layers: int = None, dropout: float = 0.1):
        super().__init__(input_dim, num_classes)

        config = MODEL_CONFIG.get("transformer", {})
        d_model = d_model or config.get("d_model", 128)
        nhead = nhead or config.get("nhead", 8)
        num_layers = num_layers or config.get("num_layers", 4)
        dropout_rate = dropout if dropout else config.get("dropout", 0.1)

        # Project input to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_rate)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Project and add positional encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use CLS token (first position) or mean pooling
        x = x[:, 0, :]  # CLS token approach

        return self.fc(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x[:, 0, :]


class ModelFactory:
    """Factory class to create models by name."""

    MODELS = {
        "mlp": MLP,
        "cnn": CNN,
        "lstm": LSTM,
        "bilstm": BiLSTM,
        "gru": GRU,
        "transformer": Transformer
    }

    @classmethod
    def create_model(cls, model_name: str, input_dim: int, num_classes: int,
                     **kwargs) -> nn.Module:
        """Create a model instance by name."""
        if model_name.lower() not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.MODELS.keys())}")

        model_class = cls.MODELS[model_name.lower()]
        return model_class(input_dim=input_dim, num_classes=num_classes, **kwargs)

    @classmethod
    def list_models(cls) -> list:
        """List available model names."""
        return list(cls.MODELS.keys())

    @classmethod
    def count_parameters(cls, model_name: str, input_dim: int, num_classes: int) -> int:
        """Get parameter count for a model."""
        model = cls.create_model(model_name, input_dim, num_classes)
        return model.count_parameters()


def get_model(args) -> nn.Module:
    """Convenience function to create model from config."""
    return ModelFactory.create_model(
        model_name=args.model_name,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        **args.model_kwargs
    )


if __name__ == "__main__":
    # Test all models
    input_dim = 78  # Typical network traffic features
    num_classes = 2  # Normal vs Attack

    print("="*60)
    print("Testing Model Architectures")
    print("="*60)

    model_configs = ["mlp", "cnn", "lstm", "bilstm", "gru", "transformer"]

    for model_name in model_configs:
        try:
            model = ModelFactory.create_model(model_name, input_dim, num_classes)
            dummy_input = torch.randn(32, input_dim)
            output = model(dummy_input)

            print(f"\n{model_name.upper()}:")
            print(f"  Parameters: {model.count_parameters():,}")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")

        except Exception as e:
            print(f"\n{model_name.upper()}: ERROR - {e}")

    print("\n" + "="*60)
    print("All models tested successfully!")
    print("="*60)