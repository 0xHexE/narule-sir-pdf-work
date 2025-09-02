import os
import time
import json
import importlib
import traceback
from typing import Dict, Any, Optional
from celery import Celery
import flwr as fl

# Create Celery instance
app = Celery('phd_tasks')

# Load configuration from celeryconfig.py
app.config_from_object('celeryconfig')

from strategies import TopKModelSelectionStrategy, AdaptiveClientSelectionStrategy

@app.task(bind=True, name='tasks.run_federated_server')
def run_federated_server(self, strategy_config: Dict[str, Any], num_rounds: int = 2, listen_addr: str = '[::]:8787'):
    """
    Run a federated learning server with configurable strategy.
    
    Args:
        strategy_config: Dictionary containing strategy configuration
            - strategy_type: 'default', 'top_k', or 'adaptive'
            - strategy_params: Dictionary of parameters for the specific strategy
        num_rounds: Number of federated learning rounds to run
        :param num_rounds: number of rounds to run
        :param strategy_config: config for the strategy
        :param listen_addr: Server address to listen on
    """
    try:
        print(f"Starting federated learning server with strategy: {strategy_config}")
        
        # Ensure necessary directories exist
        os.makedirs('round_weights', exist_ok=True)
        os.makedirs('tasks', exist_ok=True)
        
        # Create initial running status file
        task_dir = f'tasks/{self.request.id}'
        os.makedirs(task_dir, exist_ok=True)
        running_result = {
            'status': 'running',
            'strategy_type': strategy_config.get('strategy_type', 'default'),
            'strategy_params': strategy_config.get('strategy_params', {}),
            'num_rounds': num_rounds,
            'server_address': listen_addr,
            'started_at': time.time(),
            'message': 'Task is currently running'
        }
        with open(f'{task_dir}/data.json', 'w') as f:
            json.dump(running_result, f, indent=2)
        
        # Create strategy based on configuration
        strategy_type = strategy_config.get('strategy_type', 'default')
        strategy_params = strategy_config.get('strategy_params', {})

        if strategy_type == 'top_k':
            strategy = TopKModelSelectionStrategy(task_id=self.request.id, **strategy_params)
        
        elif strategy_type == 'adaptive':
            strategy = AdaptiveClientSelectionStrategy(task_id=self.request.id, **strategy_params)

        else:
            print("Invalid strategy type. ")
            raise ValueError

        print(f"Using strategy: {strategy.__class__.__name__}")
        
        # Start Flower server
        fl.server.start_server(
            server_address=listen_addr,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            grpc_max_message_length=1024*1024*1024,
            strategy=strategy,
        )
        
        # Return success result
        result = {
            'status': 'completed',
            'strategy_used': strategy.__class__.__name__,
            'strategy_type': strategy_config.get('strategy_type', 'default'),
            'strategy_params': strategy_config.get('strategy_params', {}),
            'rounds_completed': num_rounds,
            'completed_at': time.time(),
            'task_id': self.request.id  # Include task ID in result
        }
        
        # Save result in /tasks/{taskid}/data.json format
        task_dir = f'tasks/{self.request.id}'
        os.makedirs(task_dir, exist_ok=True)
        with open(f'{task_dir}/data.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        # Get full stack trace for debugging
        error_traceback = traceback.format_exc()
        print(f"Error occurred: {str(e)}")
        print(f"Stack trace:\n{error_traceback}")
        
        error_result = {
            'status': 'failed',
            'error': str(e),
            'stack_trace': error_traceback,
            'strategy_type': strategy_config.get('strategy_type', 'default'),
            'strategy_params': strategy_config.get('strategy_params', {}),
            'failed_at': time.time(),
            'task_id': self.request.id  # Include task ID in error result
        }
        
        # Save error result in /tasks/{taskid}/data.json format
        task_dir = f'tasks/{self.request.id}'
        os.makedirs(task_dir, exist_ok=True)
        with open(f'{task_dir}/data.json', 'w') as f:
            json.dump(error_result, f, indent=2)
        
        raise self.retry(exc=e, countdown=60)

if __name__ == '__main__':
    # This allows running the worker directly for testing
    app.worker_main(['worker', '--loglevel=info'])
