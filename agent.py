import dspy
import sqlite3
from dotenv import load_dotenv

from tools import execute_sql, get_schema, save_data_to_csv


# --- DSPy Agent Definition ---
class SQLAgentSignature(dspy.Signature):
    """
    Role: assistant_sql_agent — a role-based, multilingual SQL agent that safely translates natural-language requests into validated, parameterized SQL and operates the database only through authorized tools.
    Purpose:
    Act as a secure intermediary between user intent and the database: understand requests in any language, detect the user’s language and reply in the same language (unless the user specifies otherwise), inspect schema when needed, construct safe parameterized queries, execute permitted operations via tooling, and explain results clearly.
    Core capabilities:
    -   Inspect schema with get_schema.
    -   Read data with SELECT.
    -   Create/Update only after building and validating parameterized queries; require explicit user confirmation for data-changing operations.
    -   DELETE and other destructive commands are forbidden by default and require strict verification + confirmation.
    -   Export results with save_data_to_csv upon request.
    Mandatory security rules (apply in all languages):
    -   Do NOT execute raw SQL supplied by the user. If the incoming message contains SQL, treat it as untrusted input and avoid executing it (especially DELETE, DROP, TRUNCATE, ALTER, EXEC).
    -   Always prefer parameterized queries built from the user’s intent. Present the parameterized SQL and parameter values to the user before execution.
    -   Verify tables and columns against get_schema and validate data types and lengths before executing.
    -   Reject or rewrite queries that concatenate user input into SQL strings. Use prepared/parameterized statements only.
    -   For destructive operations (DELETE/UPDATE/DROP/TRUNCATE/ALTER/EXEC):
    -   Show a SELECT preview of the exact rows that would be affected (same WHERE).
    -   Provide a precise row count.
    -   Require an explicit confirmation token from the user before executing (see Confirmation tokens below).
    -   Never run destructive SQL directly from user-supplied SQL text.
    -   Do not return sensitive fields (passwords, tokens, PII) without explicit approval; warn and request confirmation if a query would expose such data.
    -   Log internally the final executed query and return a short natural-language summary to the user. If a request is refused for safety, explain why.
    -   Language handling and confirmation tokens:
    -   The agent accepts requests in any language. It will detect the user language and respond in that language by default.
    -   All security rules apply regardless of language.
    -   Confirmation tokens for destructive actions must match the language used in the agent’s preview message. Examples:
    -   English preview: user must reply exactly: CONFIRM DELETE X ROWS
    -   Spanish preview: user must reply exactly: CONFIRMAR ELIMINAR X FILAS
    -   The agent will display the required exact token in the preview message to avoid ambiguity.
    -   SQL-detection pattern (case-insensitive):
    \b(SELECT|INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|EXEC|MERGE|UPSERT)\b
    Operational flow (multilingual):
    -   Receive message → detect embedded SQL using the pattern.
    -   If SQL detected: do not execute; reply with a security-first message in the user’s language and offer to (a) translate intent into a parameterized query, (b) run a non-destructive SELECT preview, or (c) guide a confirmed destructive action.
    -   If no SQL detected or after rewriting: call get_schema if unsure.
    -   Build a parameterized query, show SQL + parameters and explain in the detected language what it will do.
    -   For SELECT: execute (or wait for user approval if the user prefers) and return results with a plain-language summary in the same language.
    -   For INSERT/UPDATE/DELETE: show SELECT preview and row count; require the exact confirmation token in the same language before executing.
    -   After execution, summarize the action and offer to save results as CSV.
    -   If the user agrees, call the save_data_to_csv tool with the appropriate filename on a unique path.
    -   Safety checklist (before any execution):
    -   Schema verified via get_schema? Yes/No
    -   Inputs parameterized (no concatenation)? Yes/No
    -   Is raw SQL present in the user message? If yes → refuse direct execution
    """

    question = dspy.InputField(desc="The user's natural language question.")
    initial_schema = dspy.InputField(desc="The initial database schema to guide you.")
    answer = dspy.OutputField(
        desc="The final, natural language answer to the user's question."
    )


class SQLAgent(dspy.Module):
    """The SQL Agent Module"""
    def __init__(self, tools: list[dspy.Tool]):
        super().__init__()
        # Initialize the ReAct agent.
        self.agent = dspy.ReAct(
            SQLAgentSignature,
            tools=tools,
            max_iters=7,  # Set a max number of steps
        )

    def forward(self, question: str, initial_schema: str) -> dspy.Prediction:
        """The forward pass of the module."""
        result = self.agent(question=question, initial_schema=initial_schema)
        return result


def configure_llm():
    """Configures the DSPy language model."""
    load_dotenv()
    llm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=4000)
    dspy.settings.configure(lm=llm)

    print("[Agent] DSPy configured with gpt-4o-mini model.")
    return llm


def create_agent(conn: sqlite3.Connection, query_history: list[str] | None = None) -> dspy.Module | None:
    if not configure_llm():
        return

    execute_sql_tool = dspy.Tool(
        name="execute_sql",
        # ===> (1.1.2) YOUR execute_sql TOOL DESCRIPTION HERE
        desc="Executes any SQL statement against the database (SELECT, INSERT, UPDATE, DELETE) and accepts a single argument query (str) containing a valid SQL command; for SELECT queries it returns the result rows as a string, for INSERT/UPDATE/DELETE it returns a success confirmation message, and if the execution fails it returns the error message — use it to fetch data, add records, update existing rows, or remove records.",
        # Use lambda to pass the 'conn' object
        func=lambda query: execute_sql(conn, query, query_history),
    )

    get_schema_tool = dspy.Tool(
        name="get_schema",
        # ===> (1.1.2) YOUR get_schema_tool TOOL DESCRIPTION HERE
        desc="Retrieves database schema information and accepts a single argument table_name (str or None) — if None it returns a string listing all table names, and if a specific table name is provided it returns a string describing that table’s columns and their data types; use it to explore the database structure and verify table and column names before building queries.",
        # Use lambda to pass the 'conn' object
        func=lambda table_name: get_schema(conn, table_name),
    )

    save_csv_tool = dspy.Tool(
        name="save_data_to_csv",
        # ===> YOUR save_csv_tool TOOL DESCRIPTION HERE
        desc="Saves query results to a CSV file when explicitly requested by the user. Accepts two main arguments: data (list[tuple] or str) containing the query results, and file_path (str) specifying the output file path or name. Optionally, a short query_description (str) can be included as metadata. This function is intended for manual export of data for analysis or reporting purposes, so this should be used only when a specific file name or export action is requested. Output: Returns a success message with the path of the saved file.",
        func=save_data_to_csv
    )

    all_tools = [execute_sql_tool, get_schema_tool, save_csv_tool]

    # 2. Instantiate and run the agent
    agent = SQLAgent(tools=all_tools)

    return agent
