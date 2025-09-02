import json
import os
from typing import List, Dict, Any

def save_task_metrics(task_id: str, metrics_data: List[Dict[str, Any]]):
    """Save metrics data to task-specific metrics file"""
    try:
        os.makedirs(f'tasks/{task_id}', exist_ok=True)
        with open(f"tasks/{task_id}/metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)
    except Exception as e:
        print(f"Error saving metrics for task {task_id}: {e}")

def load_task_metrics(task_id: str) -> List[Dict[str, Any]]:
    """Load metrics data from task-specific metrics file"""
    try:
        if os.path.exists(f"tasks/{task_id}/metrics.json"):
            with open(f"tasks/{task_id}/metrics.json", "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading metrics for task {task_id}: {e}")
    return []

def update_task_metrics(task_id: str, round_number: int, accuracy: float, loss: float):
    """Update metrics with new round data for a specific task"""
    metrics = load_task_metrics(task_id)
    
    # Check if this round already exists
    for metric in metrics:
        if metric['round'] == round_number:
            metric['accuracy'] = accuracy
            metric['loss'] = loss
            break
    else:
        # Round doesn't exist, add new entry
        metrics.append({
            'round': round_number,
            'accuracy': accuracy,
            'loss': loss
        })
    
    # Sort by round number
    metrics.sort(key=lambda x: x['round'])
    save_task_metrics(task_id, metrics)

# Keep backward compatibility (will be removed after updating strategies)
def save_metrics(metrics_data: List[Dict[str, Any]]):
    """Save metrics data to global metrics.json file (deprecated)"""
    try:
        with open("metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)
    except Exception as e:
        print(f"Error saving global metrics: {e}")

def load_metrics() -> List[Dict[str, Any]]:
    """Load metrics data from global metrics.json file (deprecated)"""
    try:
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading global metrics: {e}")
    return []

def update_metrics(round_number: int, accuracy: float, loss: float):
    """Update global metrics with new round data (deprecated)"""
    metrics = load_metrics()
    
    # Check if this round already exists
    for metric in metrics:
        if metric['round'] == round_number:
            metric['accuracy'] = accuracy
            metric['loss'] = loss
            break
    else:
        # Round doesn't exist, add new entry
        metrics.append({
            'round': round_number,
            'accuracy': accuracy,
            'loss': loss
        })
    
    # Sort by round number
    metrics.sort(key=lambda x: x['round'])
    save_metrics(metrics)