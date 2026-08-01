"""
Submission tracking ledger to record historical responses, ensure uniqueness, and prepare DataFrames for visualization.
"""
import os
import json
import sqlite3
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
from src.domain.entities import (
    SubmissionRecord,
    FormAnswerSet,
    Persona,
    AnswerItem,
    ScheduledSubmissionTask,
    ScheduleStatus,
)


class SubmissionTracker:
    """
    Manages a persistent SQLite database (or memory DB for testing) of completed submissions.
    Ensures no duplicate answers or identical persona names are reused in a campaign.
    """
    def __init__(self, db_path: str = "submissions_history.db"):
        self.db_path = db_path
        self._memory_conn = sqlite3.connect(":memory:") if db_path == ":memory:" else None
        self._init_db()

    def _get_connection(self):
        if self._memory_conn:
            return self._memory_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Creates required SQLite tables if they do not already exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                form_url TEXT NOT NULL,
                persona_name TEXT NOT NULL,
                persona_json TEXT NOT NULL,
                answers_json TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_campaign_queue (
                task_id TEXT PRIMARY KEY,
                form_url TEXT NOT NULL,
                scheduled_timestamp TEXT NOT NULL,
                answer_set_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)
        conn.commit()
        if not self._memory_conn:
            conn.close()

    def add_record(self, form_url: str, answer_set: FormAnswerSet, success: bool, error_message: Optional[str] = None):
        """Records a submission attempt into the ledger."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO submissions (timestamp, form_url, persona_name, persona_json, answers_json, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            form_url,
            answer_set.persona.name,
            json.dumps(answer_set.persona.model_dump()),
            json.dumps([a.model_dump() for a in answer_set.answers]),
            1 if success else 0,
            error_message
        ))
        conn.commit()
        if not self._memory_conn:
            conn.close()

    def is_persona_used(self, name: str, form_url: Optional[str] = None) -> bool:
        """Checks if a persona name has already been submitted for this form."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if form_url:
            cursor.execute("SELECT 1 FROM submissions WHERE persona_name = ? AND form_url = ?", (name, form_url))
        else:
            cursor.execute("SELECT 1 FROM submissions WHERE persona_name = ?", (name,))
        result = cursor.fetchone() is not None
        if not self._memory_conn:
            conn.close()
        return result

    def get_recent_answers_summary(self, form_url: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent answer structures to help prompt AI against creating repetitive textual patterns."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT answers_json FROM submissions 
            WHERE form_url = ? AND success = 1 
            ORDER BY id DESC LIMIT ?
        """, (form_url, limit))
        rows = cursor.fetchall()
        if not self._memory_conn:
            conn.close()
        return [json.loads(row[0]) for row in rows]

    def to_dataframe(self, form_url: Optional[str] = None) -> pd.DataFrame:
        """
        Exports submission history into a clean Pandas DataFrame for Streamlit UI chart plotting and analytics.
        Each column represents either metadata or a specific answered form question.
        """
        conn = self._get_connection()
        if form_url:
            df = pd.read_sql_query("SELECT * FROM submissions WHERE form_url = ? ORDER BY timestamp DESC", conn, params=(form_url,))
        else:
            df = pd.read_sql_query("SELECT * FROM submissions ORDER BY timestamp DESC", conn)
        if not self._memory_conn:
            conn.close()

        if df.empty:
            return pd.DataFrame()

        # Expand JSON columns into readable tabular structures
        records = []
        for _, row in df.iterrows():
            item = {
                "ID": row["id"],
                "Timestamp": row["timestamp"],
                "Persona Name": row["persona_name"],
                "Success": bool(row["success"]),
                "Error": row["error_message"]
            }
            try:
                persona = json.loads(row["persona_json"])
                item["Age"] = persona.get("age")
                item["Occupation"] = persona.get("occupation")
                item["Sentiment"] = persona.get("sentiment")

                answers = json.loads(row["answers_json"])
                for ans in answers:
                    title = ans.get("question_title", "Unknown Question")
                    val = ans.get("value")
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    item[f"Q: {title}"] = val
            except Exception:
                pass
            records.append(item)

        return pd.DataFrame(records)

    def get_stats(self, form_url: Optional[str] = None) -> Dict[str, int]:
        """Returns high-level statistics on completed submissions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        query = "SELECT COUNT(*), SUM(success) FROM submissions"
        params = []
        if form_url:
            query += " WHERE form_url = ?"
            params.append(form_url)
        cursor.execute(query, params)
        total, successful = cursor.fetchone()
        if not self._memory_conn:
            conn.close()
        total = total or 0
        successful = successful or 0
        return {
            "total_attempts": total,
            "successful_submissions": int(successful),
            "failed_attempts": total - int(successful)
        }

    def enqueue_scheduled_tasks(self, tasks: List[ScheduledSubmissionTask]):
        """Inserts multiple scheduled submission tasks into the persistent queue."""
        if not tasks:
            return
        conn = self._get_connection()
        cursor = conn.cursor()
        for t in tasks:
            cursor.execute("""
                INSERT OR REPLACE INTO scheduled_campaign_queue (
                    task_id, form_url, scheduled_timestamp, answer_set_json, status, attempts, last_error, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t.task_id,
                t.form_url,
                t.scheduled_timestamp,
                t.answer_set_json,
                t.status.value if hasattr(t.status, 'value') else str(t.status),
                t.attempts,
                t.last_error,
                t.created_at,
                t.completed_at
            ))
        conn.commit()
        if not self._memory_conn:
            conn.close()

    def get_due_scheduled_tasks(self, as_of: Optional[str] = None, limit: int = 10) -> List[ScheduledSubmissionTask]:
        """Retrieves pending tasks whose scheduled execution timestamp has arrived."""
        as_of = as_of or datetime.now().isoformat()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT task_id, form_url, scheduled_timestamp, answer_set_json, status, attempts, last_error, created_at, completed_at
            FROM scheduled_campaign_queue
            WHERE status = ? AND scheduled_timestamp <= ?
            ORDER BY scheduled_timestamp ASC
            LIMIT ?
        """, (ScheduleStatus.PENDING.value, as_of, limit))
        rows = cursor.fetchall()
        if not self._memory_conn:
            conn.close()

        tasks = []
        for r in rows:
            tasks.append(ScheduledSubmissionTask(
                task_id=r[0],
                form_url=r[1],
                scheduled_timestamp=r[2],
                answer_set_json=r[3],
                status=ScheduleStatus(r[4]),
                attempts=r[5],
                last_error=r[6],
                created_at=r[7],
                completed_at=r[8]
            ))
        return tasks

    def get_all_scheduled_tasks(self, status: Optional[str] = None) -> List[ScheduledSubmissionTask]:
        """Returns all scheduled tasks, optionally filtered by lifecycle status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if status:
            cursor.execute("""
                SELECT task_id, form_url, scheduled_timestamp, answer_set_json, status, attempts, last_error, created_at, completed_at
                FROM scheduled_campaign_queue WHERE status = ? ORDER BY scheduled_timestamp ASC
            """, (status,))
        else:
            cursor.execute("""
                SELECT task_id, form_url, scheduled_timestamp, answer_set_json, status, attempts, last_error, created_at, completed_at
                FROM scheduled_campaign_queue ORDER BY scheduled_timestamp ASC
            """)
        rows = cursor.fetchall()
        if not self._memory_conn:
            conn.close()

        tasks = []
        for r in rows:
            tasks.append(ScheduledSubmissionTask(
                task_id=r[0],
                form_url=r[1],
                scheduled_timestamp=r[2],
                answer_set_json=r[3],
                status=ScheduleStatus(r[4]),
                attempts=r[5],
                last_error=r[6],
                created_at=r[7],
                completed_at=r[8]
            ))
        return tasks

    def update_task_status(self, task_id: str, status: ScheduleStatus, error_message: Optional[str] = None):
        """Updates the status and completion metadata of a queued task."""
        conn = self._get_connection()
        cursor = conn.cursor()
        completed_at = datetime.now().isoformat() if status in (ScheduleStatus.COMPLETED, ScheduleStatus.FAILED) else None
        status_str = status.value if hasattr(status, 'value') else str(status)
        if error_message:
            cursor.execute("""
                UPDATE scheduled_campaign_queue
                SET status = ?, last_error = ?, completed_at = ?, attempts = attempts + 1
                WHERE task_id = ?
            """, (status_str, error_message, completed_at, task_id))
        else:
            cursor.execute("""
                UPDATE scheduled_campaign_queue
                SET status = ?, completed_at = ?, attempts = attempts + 1
                WHERE task_id = ?
            """, (status_str, completed_at, task_id))
        conn.commit()
        if not self._memory_conn:
            conn.close()

    def get_schedule_queue_dataframe(self) -> pd.DataFrame:
        """Returns a Pandas DataFrame summarizing all tasks in the scheduled campaign queue."""
        conn = self._get_connection()
        query = "SELECT task_id, form_url, scheduled_timestamp, status, attempts, last_error, completed_at FROM scheduled_campaign_queue ORDER BY scheduled_timestamp ASC"
        df = pd.read_sql_query(query, conn)
        if not self._memory_conn:
            conn.close()
        return df

    def clear_scheduled_tasks(self, status_filter: Optional[str] = None):
        """Deletes scheduled tasks from the queue, optionally filtering by status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if status_filter:
            cursor.execute("DELETE FROM scheduled_campaign_queue WHERE status = ?", (status_filter,))
        else:
            cursor.execute("DELETE FROM scheduled_campaign_queue")
        conn.commit()
        if not self._memory_conn:
            conn.close()

