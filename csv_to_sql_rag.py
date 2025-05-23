import os
import pandas as pd
from sqlalchemy import create_engine, MetaData, inspect,text
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI
import json
from typing import List, Dict, Any
import logging
from fastapi import FastAPI,UploadFile,File
import os,tempfile
import mysql.connector

app = FastAPI()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db(db_name: str, user: str, password: str, host: str, port: str) -> tuple:
    """Initialize database connection and create metadata table."""
    db_params = {
        'database': db_name,
        'user': user,
        'password': password,
        'host': host,
        'port': port
    }
    
    # Create SQLAlchemy engine for MySQL
    engine = create_engine(
        f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}'
    )
    
    # Initialize metadata table
    try:
        with mysql.connector.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(f"USE `{db_name}`;")
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS metadata_table (
                        table_name VARCHAR(255) PRIMARY KEY,
                        metadata JSON
                    )
                ''')
            conn.commit()
    except Exception as e:
        logger.error(f"Error initializing metadata table: {str(e)}")
        raise
        
    return engine, db_params

def store_metadata(db_params: dict, table_name: str, df: pd.DataFrame) -> None:
    """Store table metadata in the MySQL database."""
    metadata = {
        'table_name': table_name,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'row_count': len(df),
        'description': f"Table containing {len(df)} rows with columns: {', '.join(df.columns)}"
    }
    
    try:
        with mysql.connector.connect(**db_params) as conn:
            with conn.cursor() as cur:
                # Ensure we're using the correct database
                cur.execute(f"USE `{db_params['database']}`;")
                cur.execute(
                '''
                INSERT INTO metadata_table (table_name, metadata)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE metadata = %s;
                ''',
                (table_name, json.dumps(metadata), json.dumps(metadata))
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error storing metadata: {str(e)}")
        raise

def csv_to_sql(engine, db_params: dict, csv_path: str, table_name: str) -> None:
    """Convert CSV to SQL table and store metadata."""
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Create SQL table
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        # Extract and store metadata
        store_metadata(db_params, table_name, df)
        
        logger.info(f"Successfully converted {csv_path} to SQL table {table_name}")
    except Exception as e:
        logger.error(f"Error converting CSV to SQL: {str(e)}")
        raise

def create_embeddings(db_params: dict, openai_api_key: str) -> Chroma:
    """Create embeddings for all table metadata."""
    global metadata_records
    try:
        # Fetch all metadata
        with mysql.connector.connect(**db_params) as conn:
            with conn.cursor() as cur:
                # Ensure we're using the correct database
                cur.execute(f"USE `{db_params['database']}`;")
                cur.execute('SELECT table_name, metadata FROM metadata_table')
                metadata_records = cur.fetchall()
        # Prepare documents for embedding
        documents = []
        for table_name, metadata in metadata_records:
            if isinstance(metadata, str):
                metadata_dict = json.loads(metadata)
            else:
                metadata_dict = metadata
            doc = f"Table: {table_name}\n"
            doc += f"Description: {metadata_dict['description']}\n"
            doc += f"Columns: {', '.join(metadata_dict['columns'])}\n"
            doc += f"Data Types: {json.dumps(metadata_dict['dtypes'])}"
            documents.append(doc)

        # Create embeddings
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_text('\n'.join(documents))
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = Chroma.from_texts(texts, embeddings)
        logger.info("Successfully created embeddings for metadata")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

def preprocess_query(query: str, client: OpenAI) -> str:
    """Preprocess and regenerate the user query to make it more SQL-friendly."""
    try:
        system_message = """You are a query reformulation assistant. Your task is to take a user's potentially vague, incomplete, or poorly structured query and reformulate it into a clear, detailed natural language query that can be effectively converted to SQL by a downstream agent.
                        Instructions:

                        Analyze the user's intent: Understand what the user is trying to accomplish with their data query
                        Identify missing details: Determine what information might be needed for a complete SQL query
                        Make reasonable assumptions: Fill in likely details based on common database operations and business logic
                        Structure the query clearly: Organize the reformulated query in a logical, SQL-friendly format

                        Reformulation Guidelines:

                        Be specific about data sources: If the user mentions "sales" or "customers", specify likely table names
                        Clarify aggregations: If the user wants "totals" or "counts", specify what should be summed or counted
                        Define time periods: If the user mentions "recent" or "last month", specify exact date ranges
                        Specify sorting/ordering: Add reasonable default sorting if not specified
                        Include filtering logic: Make explicit any implicit filters or conditions
                        Clarify grouping: Specify what data should be grouped by if aggregations are involved."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        
        reformulated_query = response.choices[0].message.content
        logger.info(f"Original query: {query}")
        logger.info(f"Reformulated query: {reformulated_query}")
        return reformulated_query
    except Exception as e:
        logger.error(f"Error preprocessing query: {str(e)}")
        return query

def generate_sql(engine, vector_store: Chroma, query: str, client: OpenAI, max_attempts: int = 5) -> str:
    """Generate SQL from natural language query using RAG."""
    try:
        # Preprocess the query first
        reformulated_query = preprocess_query(query, client)
        
        # Retrieve relevant metadata
        relevant_docs = vector_store.similarity_search(reformulated_query, k=1)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        # Construct system message
        system_message = f"""You are a SQL expert specialized in MySQL. Based on the given database schema and user question, generate a valid, precise, and efficient MySQL query.

Database Schema:
{context}

Instructions:
    0. Try to query one table at a time
    1. Create syntactically correct PostgreSQL queries that directly answer the user's question. I want directly executable queries
    2. CRITICAL: Use EXACT table and column names as provided in the schema - no spelling variations, no case changes, no typos. Copy and paste names directly from the schema when possible.
    3. Use backticks (`) to quote identifiers (table and column names) to preserve case sensitivity and avoid keyword conflicts.
    4. Before finalizing your response, verify that every table and column name in your query exactly matches those in the provided schema.
    5. When appropriate, use advanced MySQL features like:
        - Common Table Expressions (WITH clauses) for complex queries
        - Window functions (OVER, PARTITION BY) for analytical queries
        - JSON functions when dealing with JSON data
    6. Cast values properly when needed (e.g., ::date, ::integer).
    7. Include appropriate JOIN conditions to prevent cartesian products.
    8. Consider performance implications - use efficient filtering early in the query.
    9. For aggregation queries, ensure all non-aggregated columns appear in GROUP BY clauses.
    10. Output only the executable SQL query - no explanations, comments, or markdown.

User Query:
{reformulated_query}"""

        # Generate SQL using OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": reformulated_query}
            ],
            temperature=0
        )
        sql_query = response.choices[0].message.content
        
        # Validate and execute SQL
        for attempt in range(max_attempts):
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql_query))
                    return sql_query
            except Exception as e:
                if attempt == max_attempts - 1:
                    return  None
                # Try to correct the SQL
                correction_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You are a SQL expert. The user will provide a faulty PostgreSQL query and an error message. "
                "Your task is to return ONLY the corrected SQL query. "
                "Do NOT include any explanation or commentary. Just return the corrected query as plain SQL. PostgreSQL query that failed with this error: {str(e)}"},
                        {"role": "user", "content": f"Original query: {sql_query}"}
                    ],
                    temperature=0
                )
                sql_query = correction_response.choices[0].message.content
         
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        raise

def execute_query(engine, sql_query: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame."""
    
    try:
        return pd.read_sql(sql_query, engine)
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise

def get_answer_from_sql_results(sql_results, user_query: str, client: OpenAI) -> str:

    result_str = sql_results

    prompt = f"""
You are a helpful assistant. Based on the following SQL query result, answer the user's question clearly and concisely.

User's question:
{user_query}

SQL query result:
{result_str}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing answers based on database query results."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()
    return answer


def visualization_detector(query, results, client):
    # Construct a detailed prompt that gives clear instructions
    prompt = f"""
    Analyze the following query and results to determine if visualization would be helpful:
    
    QUERY: {query}
    
    RESULTS: {results}
    
    Please respond in valid JSON format with the following fields:
    1. "needs_visualization": (boolean) true if visualization would be helpful, false otherwise
    2. "visualization_type": (string) the most appropriate visualization type if needed (e.g., "bar_chart", "line_chart", "pie_chart", "scatter_plot", "table", etc.), otherwise null
    3. "reasoning": (string) brief explanation for your decision
    4. "formatted_data": (object) the data formatted specifically for the suggested visualization type, otherwise null
    
    Consider these factors when deciding:
    - Numeric data comparisons often benefit from visualization
    - Time series data is good for line charts
    - Categorical comparisons work well with bar charts
    - Part-to-whole relationships suit pie charts
    - Tables are better for detailed data that needs precise values
    - Complex relationships between multiple variables may require scatter plots
    - Give json such that it should be directly given to frontend to render consider json would be dynamic use generic key value pairs
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a data analysis expert specializing in determining when data visualization would enhance understanding."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  
            temperature=0.1  
        )
        
        visualization_analysis = json.loads(response.choices[0].message.content)
        
        return visualization_analysis
        
    except Exception as e:
        # Return a fallback response in case of errors
        return {
            "needs_visualization": False,
            "visualization_type": None,
            "reasoning": f"Error analyzing visualization needs: {str(e)}",
            "formatted_data": None
        }


def main():
    # Initialize OpenAI client
    client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
    )
    engine, db_params = init_db(
        db_name=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST", "localhost"),  
        port=os.environ.get("DB_PORT", "3306")        
    )

    # Convert CSV to SQL
    csv_to_sql(engine, db_params, "/Users/prabhaskalyan/Downloads/aemtek_lims_data.csv", "ex_db")
    
    # # Create embeddings
    vector_store = create_embeddings(db_params, client.api_key)
    
    # # Generate and execute SQL from natural language
    questions = [
    "Compare actuals to spec: Summarize yesterday’s fermenter KPIs: average FERM1_TEMP_C vs spec (31–33 °C), average FERM1_PH vs spec (4.2-4.8), and flag any 5-min intervals out of range.",
]
    for query in questions:
        sql_query = generate_sql(engine, vector_store, query, client)
        if sql_query:
            print(sql_query)
            results = execute_query(engine, sql_query)
            answer = get_answer_from_sql_results(results,query,client)
            print(query+"\n")
            print(answer)
            if visualization_detector(query,results,client)['needs_visualization']:
                print(visualization_detector(query,results,client))

@app.post("/test")
def test():
    return "All good"


@app.post("/upload")
def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "Please upload a CSV file"}
    engine, db_params = init_db(
        db_name=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST", "localhost"),  # default to localhost
        port=os.environ.get("DB_PORT", "5432")        # default to 5432
    )
    try:
        contents = file.file.read().decode("utf-8")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        table_name = os.path.splitext(file.filename)[0]
        csv_to_sql(engine, db_params, tmp_file_path, table_name)
        os.remove(tmp_file_path)

        return {"message": f"CSV uploaded and data inserted into table '{table_name}'."}

    except Exception as e:
        return {"error": str(e)}


@app.post("/query")
def query(query):
    client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
    )

    engine, db_params = init_db(
        db_name=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST", "localhost"),  # default to localhost
        port=os.environ.get("DB_PORT", "5432")        # default to 5432
    )
    vector_store = create_embeddings(db_params, client.api_key)
    sql_query = generate_sql(engine, vector_store, query, client)
    if sql_query:
        print(sql_query)
        results = execute_query(engine, sql_query)
        answer = get_answer_from_sql_results(results,query,client)
        print(query+"\n")
        print(answer)
        
        if visualization_detector(query,results,client)['needs_visualization']:
            print(visualization_detector(query,results,client))
            return {"answer":answer,"json":visualization_detector(query,results,client)}
        else:
            return {"answer":answer}

        
    else:
        return "Answer not found. Try again"






