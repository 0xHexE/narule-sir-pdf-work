import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import FitRes, EvaluateRes
from collections import defaultdict
import random
from metrics_utils import update_task_metrics


class AdaptiveClientSelectionStrategy(fl.server.strategy.FedAvg):
    def __init__(self, selection_ratio: float = 0.5, metric: str = "accuracy",
                 history_weight: float = 0.7, exploration_factor: float = 0.1,
                 task_id: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_ratio = selection_ratio  # Ratio of clients to select
        self.metric = metric  # Metric to use for selection
        self.history_weight = history_weight  # Weight for historical performance
        self.exploration_factor = exploration_factor  # Factor for exploration vs exploitation
        self.task_id = task_id  # Task ID for per-task metrics
        
        # Store client performance history: {client_id: [metric_value1, metric_value2, ...]}
        self.client_history = defaultdict(list)
        self.client_scores = {}  # Current selection scores for clients
        
    def configure_fit(
        self, server_round: int, parameters, client_manager
    ):
        """Configure the next round of training with adaptive client selection."""
        # Get the default configuration from parent first
        default_configs = super().configure_fit(server_round, parameters, client_manager)
        
        # Extract client IDs from the default configurations
        client_ids = [client_proxy.cid for client_proxy, _ in default_configs]
        
        # Calculate number of clients to select
        num_clients_to_select = max(1, int(len(client_ids) * self.selection_ratio))
        
        if server_round == 1 or not self.client_scores:
            # First round or no history, select clients randomly
            selected_clients = random.sample(client_ids, num_clients_to_select)
            print(f"Round {server_round}: Randomly selected {num_clients_to_select} clients: {selected_clients}")
        else:
            # Calculate selection probabilities based on scores
            selection_probs = self._calculate_selection_probabilities(client_ids)
            
            # Select clients based on probabilities with some exploration
            selected_clients = self._select_clients_with_exploration(
                client_ids, selection_probs, num_clients_to_select
            )
            
            print(f"Round {server_round}: Adaptively selected {num_clients_to_select} clients: {selected_clients}")
        
        # Filter the default configurations to only include selected clients
        configs_for_selected = []
        for client_proxy, fit_ins in default_configs:
            if client_proxy.cid in selected_clients:
                configs_for_selected.append((client_proxy, fit_ins))
        
        return configs_for_selected
    
    def _calculate_selection_probabilities(self, client_ids: List[str]) -> Dict[str, float]:
        """Calculate selection probabilities based on historical performance."""
        probabilities = {}
        total_score = 0
        
        # Calculate scores for each client
        for client_id in client_ids:
            if client_id in self.client_scores:
                # Use exponential smoothing of historical performance
                score = self.client_scores[client_id]
                probabilities[client_id] = max(0.01, score)  # Ensure minimum probability
                total_score += probabilities[client_id]
            else:
                # New client, give baseline probability
                probabilities[client_id] = 1.0
                total_score += 1.0
        
        # Normalize probabilities
        if total_score > 0:
            for client_id in client_ids:
                probabilities[client_id] /= total_score
        
        return probabilities
    
    def _select_clients_with_exploration(self, client_ids: List[str], 
                                       probabilities: Dict[str, float], 
                                       num_to_select: int) -> List[str]:
        """Select clients with exploration-exploitation balance."""
        selected = []
        remaining_slots = num_to_select
        
        # First, exploit: select based on probabilities
        exploit_slots = int(num_to_select * (1 - self.exploration_factor))
        if exploit_slots > 0:
            exploit_candidates = []
            for client_id in client_ids:
                exploit_candidates.extend([client_id] * int(probabilities[client_id] * 100))
            
            if exploit_candidates:
                selected.extend(random.sample(exploit_candidates, min(exploit_slots, len(exploit_candidates))))
                remaining_slots -= len(selected)
        
        # Then, explore: randomly select from remaining clients
        if remaining_slots > 0:
            remaining_clients = [cid for cid in client_ids if cid not in selected]
            if remaining_clients:
                selected.extend(random.sample(remaining_clients, min(remaining_slots, len(remaining_clients))))
        
        return selected
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation results and update client performance history."""
        if not results:
            return None
        
        # Update client history and scores
        for client, result in results:
            # Handle both ClientProxy objects and string client IDs
            if hasattr(client, 'cid'):
                client_id = client.cid
            else:
                # If client is already a string (client ID), use it directly
                client_id = client
                
            metric_value = result.metrics.get(self.metric, 0)
            
            # Add to history
            self.client_history[client_id].append(metric_value)
            
            # Calculate smoothed score (exponential moving average)
            if len(self.client_history[client_id]) == 1:
                self.client_scores[client_id] = metric_value
            else:
                # EMA: new_score = alpha * current + (1-alpha) * previous
                alpha = self.history_weight
                prev_score = self.client_scores.get(client_id, metric_value)
                self.client_scores[client_id] = alpha * metric_value + (1 - alpha) * prev_score
        
        print(f"Round {server_round}: Updated client scores: {dict(self.client_scores)}")
        
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
    
    def get_client_performance_stats(self) -> Dict[str, Any]:
        """Get statistics about client performance for monitoring."""
        stats = {}
        for client_id, history in self.client_history.items():
            if history:
                stats[client_id] = {
                    'history_length': len(history),
                    'latest_value': history[-1],
                    'average': sum(history) / len(history),
                    'min': min(history),
                    'max': max(history),
                    'current_score': self.client_scores.get(client_id, 0)
                }
        return stats