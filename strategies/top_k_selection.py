from typing import List, Tuple, Optional

import flwr as fl
from flwr.common.typing import FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from metrics_utils import update_task_metrics


class TopKModelSelectionStrategy(fl.server.strategy.FedAvg):
    def __init__(self, k: int = 3, metric: str = "accuracy", task_id: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k  # Number of top clients to select
        self.metric = metric  # Metric to use for selection (accuracy, loss, etc.)
        self.client_metrics = {}  # Store client metrics for selection
        self.task_id = task_id  # Task ID for per-task metrics
        self.min_loss = float('inf')  # Track minimum loss for normalization
        self.max_loss = float('-inf')  # Track maximum loss for normalization
        
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
            # Get the default configuration from parent
            default_configs = super().configure_fit(server_round, parameters, client_manager)
            
            # Find the client objects for the selected client IDs
            configs_for_selected = []
            for client_proxy, fit_ins in default_configs:
                if client_proxy.cid in selected_clients:
                    configs_for_selected.append((client_proxy, fit_ins))
            
            print(f"Round {server_round}: Configuring fit for {len(configs_for_selected)} selected clients out of {len(default_configs)} total")
            return configs_for_selected
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
            
            # Debug information about the result
            print(f"Round {server_round}: Client {client_id} - loss: {result.loss}, metrics: {result.metrics}")
                
            if self.metric == "loss":
                # For loss, we want to minimize, so we store the actual loss value from result.loss
                metric_value = result.loss
                # Validate and normalize loss value
                if metric_value < 0:
                    print(f"Warning: Negative loss value {metric_value} from client {client_id}, clamping to 0")
                    metric_value = 0.0
                elif metric_value > 10:  # Arbitrary upper bound for sanity
                    print(f"Warning: Unusually high loss value {metric_value} from client {client_id}, clamping to 10")
                    metric_value = 10.0
                
                # Update min/max for normalization tracking
                self.min_loss = min(self.min_loss, metric_value)
                self.max_loss = max(self.max_loss, metric_value)
                
                self.client_metrics[client_id] = metric_value
            else:
                # For accuracy and other metrics, we store the value from metrics
                metric_value = result.metrics.get(self.metric, 0)
                # Validate accuracy values (should be between 0 and 1)
                if self.metric == "accuracy" and (metric_value < 0 or metric_value > 1):
                    print(f"Warning: Invalid accuracy value {metric_value} from client {client_id}, clamping to [0,1]")
                    metric_value = max(0.0, min(1.0, metric_value))
                
                self.client_metrics[client_id] = metric_value
        
        print(f"Round {server_round}: Updated client metrics: {self.client_metrics}")
        if self.metric == "loss":
            print(f"Round {server_round}: Loss range - min: {self.min_loss:.6f}, max: {self.max_loss:.6f}")
        
        # Extract and store loss and accuracy metrics
        if results:
            # Calculate average loss and accuracy across all clients
            total_loss = 0
            total_accuracy = 0
            count = 0
            
            for _, result in results:
                # Loss is the first element of the tuple returned by client.evaluate()
                # result.loss contains the actual loss value
                loss_value = result.loss
                accuracy_value = result.metrics.get('accuracy', 0)
                total_loss += loss_value
                total_accuracy += accuracy_value
                count += 1
            
            if count > 0:
                avg_loss = total_loss / count
                avg_accuracy = total_accuracy / count
                if self.task_id:
                    update_task_metrics(self.task_id, server_round, avg_accuracy, avg_loss)
                print(f"Round {server_round}: Average loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}")
        
        # Call parent aggregate_evaluate
        return super().aggregate_evaluate(server_round, results, failures)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[any]:
        """Aggregate fit results from selected clients."""
        # Always aggregate all results to ensure model updates from all clients
        # The selection happens in configure_fit, but we want to aggregate all contributions
        return super().aggregate_fit(server_round, results, failures)