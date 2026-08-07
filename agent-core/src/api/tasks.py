"""
api/tasks.py — Task management REST API.

Exposes active tasks for viewing and editing from the web dashboard.
"""

from dataclasses import asdict
from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Optional

import task_store
import scheduler

router = APIRouter(prefix='/tasks', tags=['tasks'])


@router.get('')
async def list_tasks():
    """List all active tasks."""
    task_store.load_all()
    tasks = task_store.active_tasks()
    return {'tasks': [asdict(t) for t in tasks]}


class TaskUpdateBody(BaseModel):
    goal: Optional[str] = None
    progress: Optional[str] = None
    check_cron: Optional[str] = None


@router.patch('/{task_id}')
async def update_task(task_id: str, body: TaskUpdateBody):
    """Update task fields (goal, progress, check_cron)."""
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        return {'ok': False, 'error': 'No fields to update'}

    task = task_store.update(task_id, **updates)
    if not task:
        return {'ok': False, 'error': f'Task {task_id} not found'}

    # Re-register cron if changed
    if 'check_cron' in updates:
        scheduler.remove_job(f'task:{task_id}')
        if updates['check_cron']:
            scheduler.add_job(
                f'task:{task_id}',
                updates['check_cron'],
                f'任务定时检查 [{task_id}]：{task.goal}。请查询实际状态并更新进展。',
            )

    return {'ok': True, 'task': asdict(task)}


@router.post('/{task_id}/done')
async def mark_done(task_id: str):
    """Mark task as completed."""
    scheduler.remove_job(f'task:{task_id}')
    task = task_store.done(task_id, summary='手动标记完成')
    return {'ok': bool(task)}


@router.delete('/{task_id}')
async def delete_task(task_id: str):
    """Delete a task."""
    scheduler.remove_job(f'task:{task_id}')
    task = task_store.done(task_id, summary='手动删除')
    return {'ok': bool(task)}


@router.delete('')
async def clear_all_tasks():
    """Clear all active tasks."""
    tasks = task_store.active_tasks()
    for t in tasks:
        scheduler.remove_job(f'task:{t.id}')
        task_store.done(t.id, summary='批量清除')
    return {'ok': True, 'cleared': len(tasks)}
