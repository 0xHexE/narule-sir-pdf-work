import os
import json
import glob
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from celery import Celery
from celery.result import AsyncResult
import time

# Create FastAPI app
app = FastAPI(title="Task Monitor", description="Monitor and manage Celery tasks")

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create Celery instance for task management
monitor_celery = Celery('phd_tasks')
monitor_celery.config_from_object('celeryconfig')

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/tasks")
async def get_tasks():
    """Get list of all tasks with their status"""
    try:
        # Get all task directories
        task_dirs = glob.glob('tasks/*')
        tasks = []
        
        # Add tasks from task directories
        for task_dir in task_dirs:
            try:
                task_id = os.path.basename(task_dir)
                data_file = f'{task_dir}/data.json'
                
                if os.path.exists(data_file):
                    with open(data_file, 'r') as f:
                        task_data = json.load(f)
                    
                    tasks.append({
                        'id': task_id,
                        'status': task_data.get('status', 'unknown'),
                        'data': task_data,
                        'file': f'tasks/{task_id}/data.json'
                    })
                
            except (json.JSONDecodeError, IOError):
                continue
        
        # Sort by status (running first, then completed, then failed) and timestamp
        status_order = {'running': 0, 'pending': 1, 'completed': 2, 'failed': 3}
        tasks.sort(key=lambda x: (
            status_order.get(x['status'], 4),
            -x['data'].get('completed_at', x['data'].get('failed_at', x['data'].get('started_at', 0)))
        ))
        
        return {"tasks": tasks}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading tasks: {str(e)}")

@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    """Get specific task details"""
    try:
        # Check for result file
        result_file = f'celery_results/server_result_{task_id}.json'
        error_file = f'celery_results/server_error_{task_id}.json'
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
            return {
                "id": task_id,
                "status": "completed",
                "data": data
            }
        elif os.path.exists(error_file):
            with open(error_file, 'r') as f:
                data = json.load(f)
            return {
                "id": task_id,
                "status": "failed",
                "data": data
            }
        else:
            # Check if task is still running
            try:
                result = AsyncResult(task_id, app=monitor_celery)
                if result.state == 'PENDING':
                    return {
                        "id": task_id,
                        "status": "pending",
                        "data": {"state": "Task is pending execution"}
                    }
                elif result.state == 'STARTED':
                    return {
                        "id": task_id,
                        "status": "running",
                        "data": {"state": "Task is currently running"}
                    }
                else:
                    return {
                        "id": task_id,
                        "status": "unknown",
                        "data": {"state": result.state}
                    }
            except:
                return {
                    "id": task_id,
                    "status": "not_found",
                    "data": {"error": "Task not found"}
                }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting task: {str(e)}")

@app.get("/api/check_port/{port}")
async def check_port(port: int):
    """Check if a port is already in use by a running task"""
    try:
        # Get all task directories
        task_dirs = glob.glob('tasks/*')
        
        for task_dir in task_dirs:
            try:
                data_file = f'{task_dir}/data.json'
                if os.path.exists(data_file):
                    with open(data_file, 'r') as f:
                        task_data = json.load(f)
                    
                    # Check if task is running and using the same port
                    if task_data.get('status') == 'running' and task_data.get('server_address'):
                        # Extract port from server address (format: "[::]:8787")
                        server_addr = task_data.get('server_address', '')
                        if server_addr.endswith(f':{port}'):
                            task_id = os.path.basename(task_dir)
                            raise HTTPException(
                                status_code=409,
                                detail=f"Port {port} is already in use",
                                headers={"X-Task-ID": task_id}
                            )
                            
            except (json.JSONDecodeError, IOError):
                continue
        
        return {"available": True, "message": f"Port {port} is available"}
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error checking port: {str(e)}")

@app.post("/api/run_task")
async def run_task(
    strategy_type: str = Form("adaptive"),
    num_rounds: int = Form(2),
    k: Optional[int] = Form(None),
    metric: Optional[str] = Form("accuracy"),
    selection_ratio: Optional[float] = Form(0.5),
    server_port: int = Form(8787)
):
    server_address = f"[::]:{server_port}"
    """Start a new federated learning task"""
    try:
        # Build strategy configuration
        strategy_config = {
            "strategy_type": strategy_type,
            "strategy_params": {}
        }
        
        if strategy_type == "top_k":
            strategy_config["strategy_params"] = {
                "k": k or 3,
                "metric": metric or "accuracy"
            }
        elif strategy_type == "adaptive":
            strategy_config["strategy_params"] = {
                "selection_ratio": selection_ratio or 0.5,
                "metric": metric or "accuracy",
                "history_weight": 0.7,
                "exploration_factor": 0.1
            }
        
        # Start the task
        result = monitor_celery.send_task(
            'tasks.run_federated_server',
            args=[strategy_config, num_rounds],
            kwargs={'listen_addr': server_address}
        )
        
        return {
            "task_id": result.id,
            "status": "started",
            "strategy_config": strategy_config,
            "num_rounds": num_rounds
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting task: {str(e)}")

@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task result file"""
    try:
        result_file = f'celery_results/server_result_{task_id}.json'
        error_file = f'celery_results/server_error_{task_id}.json'
        
        deleted = []
        if os.path.exists(result_file):
            os.remove(result_file)
            deleted.append(result_file)
        if os.path.exists(error_file):
            os.remove(error_file)
            deleted.append(error_file)
        
        if deleted:
            return {"deleted": deleted, "message": "Task files deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Task files not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting task: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    """Get metrics from the metrics.json file if it exists"""
    try:
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                metrics = json.load(f)
            return {"metrics": metrics}
        else:
            return {"metrics": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)
