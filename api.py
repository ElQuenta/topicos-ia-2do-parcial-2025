import ast
import sqlite3
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, Depends, Body, BackgroundTasks
import dspy
from pydantic import BaseModel

from database import setup_database
from tools import get_schema, execute_sql
from agent import create_agent

app = FastAPI(title="AI db assistant")
query_history = []
query_queue = {}

class AgentResponse(BaseModel):
    original_query: str
    sql_queries: list[str]
    agent_answer: str

class AgentAsyncStartResponse(BaseModel):
    query_id: uuid.UUID
    status: str = "pending"

class AgentAsyncFinishResponse(AgentResponse):
    query_id: uuid.UUID
    status: str = "finished"

def get_db_connection() -> sqlite3.Connection:
    return setup_database()

def get_db_schema(conn: Annotated[sqlite3.Connection,Depends(get_db_connection)]) -> str:
    return get_schema(conn)

def get_agent(conn: Annotated[sqlite3.Connection,Depends(get_db_connection)]) -> dspy.Module:
    return create_agent(conn, query_history)


def query_agent(
    agent: dspy.Module,
    user_query: str,
    db_schema: str,
    track_query: bool = False,
    query_id: uuid.UUID | None = None,
    db_conn: sqlite3.Connection | None = None
) -> AgentResponse:
    outputs = agent(question=user_query, initial_schema=db_schema)
    results = AgentResponse(
        original_query=user_query,
        sql_queries=query_history,
        agent_answer=outputs.answer
    )
    if track_query and query_id is not None:
        results_json = results.model_dump_json()
        update_query = f"UPDATE queries SET result='{results_json}', status='finished' WHERE id='{query_id}';"
        execute_sql(db_conn, update_query)
    query_history.clear()

    return results

@app.post("/database/natural_queries")
def query_database(
    db_schema: Annotated[str, Depends(get_db_schema)],
    agent: Annotated[dspy.Module, Depends(get_agent)],
    user_query : str = Body(..., embed=True),
) -> AgentResponse:
    response = query_agent(agent, user_query, db_schema)
    return response

@app.post("/database/async_queries")
def async_query_database(
db_schema: Annotated[str, Depends(get_db_schema)],
    agent: Annotated[dspy.Module, Depends(get_agent)],
    background_tasks: BackgroundTasks,
    db_conn: Annotated[sqlite3.Connection, Depends(get_db_connection)],
    timestamp: datetime | None = Body(default_factory=datetime.now),
    user_query : str = Body(..., embed=True),
) -> AgentAsyncStartResponse:
    query_id = uuid.uuid4()
    insert_query_str = f"INSERT INTO queries (id, status, result) VALUES ('{query_id}', 'pending', '')"
    execute_sql(db_conn, insert_query_str)
    background_tasks.add_task(query_agent, agent, user_query, db_schema, True, query_id, db_conn)
    return AgentAsyncStartResponse(query_id=query_id)

@app.get("/database/async_queries")
def get_async_query_result(
    db_conn: Annotated[sqlite3.Connection, Depends(get_db_connection)],
    query_id : uuid.UUID,
) -> AgentAsyncStartResponse | AgentAsyncFinishResponse:
    result = execute_sql(db_conn, f"SELECT * FROM queries WHERE id = '{query_id}'")
    rows = ast.literal_eval(result)
    print(f"Fetched rows for query_id {query_id}: {rows}")
    if not rows or not rows[0][2]:
        return AgentAsyncStartResponse(
            query_id=query_id,
            status="pending",
        )

    response = AgentResponse.model_validate_json(rows[0][2])
    return AgentAsyncFinishResponse(
        original_query=response.original_query,
        sql_queries=response.sql_queries,
        agent_answer=response.agent_answer,
        query_id=query_id
    )
