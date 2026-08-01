import pytest
import os
from datetime import datetime, timedelta
from src.automation.temporal_scheduler import DiurnalTimestampGenerator, SwarmWorkerPool
from src.persistence.submission_repository import SubmissionTracker
from src.domain.entities import ScheduledSubmissionTask, ScheduleStatus, FormAnswerSet, Persona, AnswerItem


def test_diurnal_timestamp_generation_short_window():
    now = datetime.now()
    timestamps = DiurnalTimestampGenerator.generate_schedule(
        count=5,
        start_time=now,
        duration_hours=0.5,
        apply_diurnal_curve=False,
        seed=10
    )
    assert len(timestamps) == 5
    assert all(now <= ts <= now + timedelta(hours=0.5) for ts in timestamps)
    assert timestamps == sorted(timestamps)


def test_diurnal_timestamp_generation_all_day():
    now = datetime.now()
    timestamps = DiurnalTimestampGenerator.generate_schedule(
        count=30,
        start_time=now,
        duration_hours=24.0,
        apply_diurnal_curve=True,
        seed=42
    )
    assert len(timestamps) == 30
    assert timestamps[0] >= now
    assert timestamps[-1] <= now + timedelta(hours=24.1)


def test_schedule_queue_persistence():
    db_path = ":memory:"
    tracker = SubmissionTracker(db_path=db_path)
    
    task1 = ScheduledSubmissionTask(
        form_url="https://docs.google.com/forms/d/test",
        scheduled_timestamp=(datetime.now() - timedelta(seconds=10)).isoformat(), # Already due
        answer_set_json='{"persona": {"name": "Test User", "age": 20}, "answers": []}',
        status=ScheduleStatus.PENDING
    )
    task2 = ScheduledSubmissionTask(
        form_url="https://docs.google.com/forms/d/test",
        scheduled_timestamp=(datetime.now() + timedelta(hours=5)).isoformat(), # Future task
        answer_set_json='{"persona": {"name": "Future User", "age": 21}, "answers": []}',
        status=ScheduleStatus.PENDING
    )
    
    tracker.enqueue_scheduled_tasks([task1, task2])
    all_tasks = tracker.get_all_scheduled_tasks()
    assert len(all_tasks) == 2
    
    due = tracker.get_due_scheduled_tasks()
    assert len(due) == 1
    assert due[0].task_id == task1.task_id
    
    tracker.update_task_status(task1.task_id, ScheduleStatus.COMPLETED)
    assert len(tracker.get_due_scheduled_tasks()) == 0
    
    df = tracker.get_schedule_queue_dataframe()
    assert len(df) == 2
