import torch
import psutil
import torch_geometric
import torch_geometric.nn as gnn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class SatelliteImageGNN(torch.nn.Module):
    def __init__(self, scale_factor=3, hidden_channels=32, device=None):
        super().__init__()

        # Scale factor (3x3 to 1 pixel ratio)
        self.scale_factor = scale_factor

        self.output_shape = (6000, 6000)

        # Force CPU device since we're not using GPU
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device

        # Input channels = 1 (temperature value) + 2 (position encoding)
        # Output channels = scale_factor^2 (to predict values for each high-res pixel)
        self.conv1 = gnn.GCNConv(3, hidden_channels)  # 3 = 1 temp + 2 pos
        self.conv2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = gnn.GCNConv(hidden_channels, scale_factor * scale_factor)

        self.to(self.device)

        self.train_losses = []

    def create_graph(self, low_res_image):
        height, width = low_res_image.shape

        # Create node features: temperature value + position encoding
        x = torch.tensor(low_res_image, dtype=torch.float).view(-1, 1)

        # Add position encoding
        pos_h = torch.arange(height).repeat_interleave(width)
        pos_w = torch.arange(width).repeat(height)
        pos = torch.stack([pos_h, pos_w], dim=1).float()

        # Normalize positions
        pos = pos / torch.tensor([height, width])

        # Combine temperature and position features
        x = torch.cat([x, pos], dim=1).to(self.device)

        # Create edges (8-neighborhood connectivity)
        edge_index = []
        for i in range(height):
            for j in range(width):
                node_idx = i * width + j
                neighbors = [
                    (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                    (i, j - 1), (i, j + 1),
                    (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)
                ]

                for ni, nj in neighbors:
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_idx = ni * width + nj
                        edge_index.extend([[node_idx, neighbor_idx]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(self.device)

        return torch_geometric.data.Data(x=x, edge_index=edge_index)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply GNN layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Reshape output to match high-resolution grid
        batch_size = x.size(0) // (self.output_shape[0] // self.scale_factor) ** 2
        high_res_size = self.output_shape[0]

        # Reshape the output to match the high-resolution grid structure
        x = x.view(batch_size,
                   high_res_size // self.scale_factor,
                   high_res_size // self.scale_factor,
                   self.scale_factor,
                   self.scale_factor)

        # Rearrange to get final high-resolution output
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, high_res_size, high_res_size)

        return x

    def train_model(self, inputs, outputs, epochs=10, lr=0.1, batch_size=1):

        print(epochs)
        print(lr)
        print(batch_size)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        # Store output shape for reconstruction
        self.output_shape = outputs.shape[1:]

        # Process data in smaller chunks if needed
        chunk_size = batch_size

        print(f"Training on {self.device}")

        print("Entering epochs loop")
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(inputs), chunk_size):
                batch_inputs = torch.tensor(inputs[i:i + chunk_size], dtype=torch.float).to(self.device)
                batch_outputs = torch.tensor(outputs[i:i + chunk_size], dtype=torch.float).to(self.device)

                batch_losses = []
                for low_res, high_res in zip(batch_inputs, batch_outputs):
                    # Create graph from low-res input
                    graph_data = self.create_graph(low_res.cpu().numpy())

                    # Forward pass
                    pred = self(graph_data)
                    if pred.dim() == 3 and pred.size(0) == 1:
                        pred = pred.squeeze(0)

                    print(high_res.cpu().numpy())

                    loss = criterion(pred, high_res)
                    batch_losses.append(loss)


                # Average batch loss
                batch_loss = torch.mean(torch.stack(batch_losses))

                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()

                # Clear some memory
                del batch_losses, batch_loss
                torch.cuda.empty_cache()  # This won't do anything on CPU but keeping for when GPU is used

            print(f'Epoch {epoch}, Average Loss: {total_loss / len(inputs)}')
            print(f'Memory usage: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB')
            with open('training_loss.txt', 'a') as f:
                f.write(f"Epoch {epoch} = {total_loss/len(inputs)}")

        return self

    def super_resolve(self, low_res_image):
        self.eval()
        graph_data = self.create_graph(low_res_image)
        with torch.no_grad():
            super_resolved = self(graph_data)
        return super_resolved.cpu().numpy()

    def evaluate(self, test_inputs, test_outputs):
        self.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for low_res, high_res in zip(test_inputs, test_outputs):
                graph_data = self.create_graph(low_res.cpu().numpy())
                pred = self(graph_data)
                if pred.dim() == 3 and pred.size(0) == 1:
                    pred = pred.squeeze(0)

                predictions.append(pred.cpu().numpy())
                actuals.append(high_res.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        metrics = {
            'mse': mean_squared_error(actuals.flatten(), predictions.flatten()),
            'mae': mean_absolute_error(actuals.flatten(), predictions.flatten()),
            'r2': r2_score(actuals.flatten(), predictions.flatten())
        }

        # Save metrics
        with open('test_metrics.txt', 'a') as f:
            f.write(f"mse = {metrics['mse']} \nmae = {metrics['mae']} \nr2 = {metrics['r2']} \n")

        return metrics