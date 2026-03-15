import hashlib
import json
from datetime import datetime, timedelta, timezone
from models.db_schemes.celery_task_execution import CeleryTaskExecution
from models.enums.DataBaseEnum import DataBaseEnum



class IdempotencyManager:
    def __init__(self,db_client):
        self.db_client = db_client
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CELERY_TASK_EXECUTION_NAME.value]
    
    def create_args_hash(self,task_name:str,task_args:dict):
        combined_data={
            **task_args,
            "task_name":task_name
        }
        json_string = json.dumps(combined_data, sort_keys=True, default=str)
        return hashlib.sha256(json_string.encode()).hexdigest()
    
    async def create_task_record(self,task_name:str,task_args:dict,celery_task_id:str=None)->CeleryTaskExecution:
        """Create new task execution record."""
        args_hash = self.create_args_hash(task_name, task_args)
        task_record=CeleryTaskExecution(
            task_name=task_name,
            task_args_hash=args_hash,
            task_args=task_args,
            celery_task_id=celery_task_id,
            status='PENDING',
            started_at=datetime.utcnow()
        )
        
        # We need to get the collection from the database        
        #collection = self.db_client[DataBaseEnum.COLLECTION_CELERY_TASK_EXECUTION_NAME.value]
        try:
            result = await self.collection.insert_one(
                task_record.dict(by_alias=True, exclude_unset=True)
            )
            task_record.id = result.inserted_id
            return task_record
        except Exception as e:
            # Handle any insertion errors
            raise e




    async def update_task_status(self, execution_id: str, status: str, result: dict = None):
        """Update task status and result."""
        from bson.objectid import ObjectId
        
        #collection = self.db_client[DataBaseEnum.COLLECTION_CELERY_TASK_EXECUTION_NAME.value]
        
        try:
            # Build update document
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if result:
                update_data["result"] = result
            
            if status in ['SUCCESS', 'FAILURE']:
                update_data["completed_at"] = datetime.utcnow()
            
            # MongoDB equivalent of SQLAlchemy get() + update + commit()
            update_result = await self.collection.update_one(
                {"_id": ObjectId(execution_id)},
                {"$set": update_data}
            )
            
            return update_result.modified_count > 0
        except Exception as e:
            raise e
    
    async def get_existing_task(self, task_name: str, 
                                task_args: dict, celery_task_id: str) -> CeleryTaskExecution:
        """Check if task with same name and args already exists."""
        args_hash = self.create_args_hash(task_name, task_args)
        
        try:
            # MongoDB equivalent of SQLAlchemy select().where()
            result = await self.collection.find_one({
                "celery_task_id": celery_task_id,
                "task_name": task_name,
                "task_args_hash": args_hash
            })
            
            if result is None:
                return None
            
            return CeleryTaskExecution(**result)
        except Exception as e:
            raise e

    async def should_execute_task(self, task_name: str, task_args: dict,
                                  celery_task_id: str, 
                                  task_time_limit: int = 600) -> tuple[bool, CeleryTaskExecution]:
        """
        Check if task should be executed or return existing result.
        Args:
            task_time_limit: Time limit in seconds after which a stuck task can be re-executed
        Returns (should_execute, existing_task_or_none)
        """
        existing_task = await self.get_existing_task(task_name, task_args, celery_task_id)
        
        if not existing_task:
            return True, None
            
        # Don't execute if task is already completed successfully
        if existing_task.status == 'SUCCESS':
            return False, existing_task
            
        # Check if task is stuck (running longer than time limit + 60 seconds)
        if existing_task.status in ['PENDING', 'STARTED', 'RETRY']:
            if existing_task.started_at:
                time_elapsed = (datetime.utcnow() - existing_task.started_at).total_seconds()
                time_gap = 60  # 60 seconds grace period
                if time_elapsed > (task_time_limit + time_gap):
                    return True, existing_task  # Task is stuck, allow re-execution
            return False, existing_task  # Task is still running within time limit
            
        # Re-execute if previous task failed
        return True, existing_task

    async def cleanup_old_tasks(self, time_retention: int = 86400) -> int:
        """
        Delete old task records older than time_retention seconds.
        Args:
            time_retention: Time in seconds to retain tasks (default: 86400 = 24 hours)
        Returns:
            Number of deleted records
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=time_retention)
        
        try:
            # MongoDB equivalent of SQL delete().where()
            result = await self.collection.delete_many({
                "created_at": {"$lt": cutoff_time}
            })
            return result.deleted_count
        except Exception as e:
            raise e