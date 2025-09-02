from typing import List, Tuple, Optional

import flwr as fl
from flwr.common.typing import FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy


class TopKModelSelectionStrategy(fl.server.strategy.FedAvg):
    def __init__(self, k: int = 3, metric: str = "accuracy", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k  # Number of top clients to select
        self.metric = metric  # Metric to use for selection (accuracy, loss, etc.)
        self.client_metrics = {}  # Store client metrics for selection
        
    def configure_fit(
        self, server_round: int, parameters, client_manager
    ):
        """Configure the next round of training."""
        # Get all available clients
        clients = client_manager.all()
        
        # If we have historical metrics, select top K clients
        if self.client_metrics:
            # Sort clients by their performance metric (higher is better for accuracy, lower for loss)
            if self.metric == "loss":
                # For loss, we want clients with lower values
                sorted_clients = sorted(
                    self.client_metrics.items(),
                    key=lambda x: x[1],
                    reverse=False  # Ascending order for loss (lower is better)
                )
            else:
                # For accuracy and other metrics, higher is better
                sorted_clients = sorted(
                    self.client_metrics.items(),
                    key=lambda x: x[1],
                    reverse=True  # Descending order for accuracy (higher is better)
                )
            
            # Select top K clients
            selected_clients = [client for client, _ in sorted_clients[:self.k]]
            print(f"Round {server_round}: Selected top {self.k} clients based on {self.metric}: {selected_clients}")
            
            # Configure fit for selected clients
            config = {}
            for client in selected_clients:
                config[client] = super().configure_fit(server_round, parameters, client_manager)
            
            return config
        else:
            # First round, use default selection
            return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation results and update client metrics."""
        if not results:
            return None
        
        # Update client metrics based on evaluation results
        for client, result in results:
            # Handle both ClientProxy objects and string client IDs
            if hasattr(client, 'cid'):
                client_id = client.cid
            else:
                # If client is already a string (client ID), use it directly
                client_id = client
                
            metric_value = result.metrics.get(self.metric, 0)
            
            if self.metric == "loss":
                # For loss, we want to minimize, so we store the actual loss value
                self.client_metrics[client_id] = metric_value
            else:
                # For accuracy and other metrics, we store the actual value
                self.client_metrics[client_id] = metric_value
        
        print(f"Round {server_round}: Updated client metrics: {self.client_metrics}")
        
        # Call parent aggregate_evaluate
        return super().aggregate_evaluate(server_round, results, failures)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[any]:
        """Aggregate fit results from selected clients."""
        # Only aggregate results from our selected top K clients
        if self.client_metrics:
            # Filter results to only include our selected clients
            selected_client_ids = list(self.client_metrics.keys())[:self.k]
            filtered_results = [
                (client, result) for client, result in results
                if (client.cid if hasattr(client, 'cid') else client) in selected_client_ids
            ]
            
            print(f"Round {server_round}: Aggregating results from {len(filtered_results)} selected clients")
            
            # Aggregate only the selected clients' results
            return super().aggregate_fit(server_round, filtered_results, failures)
        else:
            # First round, aggregate all results
            return super().aggregate_fit(server_round, results, failures)