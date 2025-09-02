# Celery configuration for PHD project
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Broker and backend settings
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', '6379')
redis_db = os.getenv('REDIS_DB', '0')
redis_password = os.getenv('REDIS_PASSWORD')

# Construct Redis URL with authentication if password is provided
if redis_password:
    broker_url = f'redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}'
    result_backend = f'redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}'
else:
    broker_url = f'redis://{redis_host}:{redis_port}/{redis_db}'
    result_backend = f'redis://{redis_host}:{redis_port}/{redis_db}'

# Serialization settings
accept_content = ['json']
task_serializer = 'json'
result_serializer = 'json'

# Task settings
task_ignore_result = False
task_track_started = True
task_time_limit = 30 * 60  # 30 minutes
task_soft_time_limit = 25 * 60  # 25 minutes

# Worker settings
worker_prefetch_multiplier = 1
worker_concurrency = 4  # Number of concurrent workers
worker_max_tasks_per_child = 100
worker_max_memory_per_child = 200000  # 200MB

# Beat settings (for periodic tasks)
beat_schedule = {
    'cleanup-old-results-every-hour': {
        'task': 'tasks.cleanup_old_results',
        'schedule': 3600.0,  # Every hour
        'args': (24,),  # Keep results for 24 hours
    },
}

# Task routes
task_routes = {
    'tasks.process_federated_learning_round': {'queue': 'federated'},
    'tasks.train_model_task': {'queue': 'training'},
    'tasks.data_preprocessing_task': {'queue': 'preprocessing'},
}

# Timezone
timezone = 'Asia/Dubai'
enable_utc = True

# Result expiration (1 day)
result_expires = 86400  # 24 hours in seconds
