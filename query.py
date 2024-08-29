import logging
from sqlalchemy.exc import SQLAlchemyError
from sshtunnel import BaseSSHTunnelForwarderError, SSHTunnelForwarder
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    Date,
    Boolean,
    select,
    column,
    insert,
    engine, 
    inspect,
    text
)

from llama_index.legacy import SQLDatabase
from llama_index.llms.openai import OpenAI
from llama_index.core.objects import (
    ObjectIndex,
    SQLTableNodeMapping,   
    SQLTableSchema,
)
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.retrievers import NLSQLRetriever
from llama_index.core.query_engine import NLSQLTableQueryEngine

import requests
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
import csv

from sshtunnel import SSHTunnelForwarder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_question(question: str):
    try:

        # Load environment variables from .env file
        load_dotenv()

        
        server_ip = os.getenv("SERVER_IP")
        ssh_username = os.getenv("SSH_USERNAME")
        ssh_password = os.getenv("SSH_PASSWORD")
        # Connect to the server clone via SSH Tunnel
        server = SSHTunnelForwarder(
            (server_ip, 22),
            ssh_username=ssh_username,
            ssh_password=ssh_password,
            remote_bind_address=('localhost', 5432),
            local_bind_address=('localhost', 6543),
        )
        
        server.start()
        local_port = str(server.local_bind_port)

        dbschema = 'expanded_hts_prep,public'
        engine = create_engine('postgresql://emmanuel:5gmuPHOxaQpSEQt&*^46@127.0.0.1:{}/lamisplus_ods_dwh'.format(local_port),
        connect_args={'options': '-csearch_path={}'.format(dbschema)})

        llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

        period_df = pd.read_sql("SELECT * FROM expanded_hts_prep.period;", engine)
        this_period = period_df['periodcode'][period_df['is_active']].to_string(index=False).lower()
        this_table = "expanded_hts_weekly_" + this_period

        sql_database = SQLDatabase(engine, include_tables=[this_table])

        with open('data-dictionary.csv', 'r') as file:
            reader = csv.DictReader(file)
            data_dict = {row["parent"] + row["name"]: row for row in reader}

        hts_table_text = (
        "This table gives information regarding HIV Testing Services (HTS)."
        "Users may ask about positive tests and positivity rates. Positive tests are those where finalhivtestresult is positive. Negative tests are those where finalhivtestresult is negative or missing or null"
        "If a user asks about positivity rate (or similar), calculate it as the number of positive tests divided by the total number of tests for the specified group"
        )

        table_node_mapping = SQLTableNodeMapping(sql_database)
        table_schema_objs = [
            (SQLTableSchema(table_name=this_table, context_str=('description of the table: ' + hts_table_text + '. These are columns in the table and their descriptions: ' + str(data_dict))))
        ]

        

        # Access the OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Ensure the key is loaded before using it
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please check your .env file.")

        # Use the API key
        os.environ["OPENAI_API_KEY"] = openai_api_key


        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=[this_table]
        )

        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
        )

        

        custom_txt2sql_prompt = ("""Given an input question, construct a syntactically correct SQL query to run, then look at the results of the query and return a comprehensive and detailed answer. Ensure that you:
                - Select only the relevant columns needed to answer the question.
                - Use correct column and table names as provided in the schema description. Avoid querying for columns that do not exist.
                - Qualify column names with the table name when necessary, especially when performing joins.
                - Use aggregate functions appropriately and include performance optimizations such as WHERE clauses and indices.
                - Add additional related information for the user.
                - Use background & definitions provided for more detailed answer. Follow the instructions.
                - Your are provided with several tables each for a different proram area, ensure you retrive the relevant table.
                - do not hallucinate column names. If you can't  find a column name, do not write the sql query say I'm not sure.

                Special Instructions:
                - Default to using averages for aggregation if not specified by the user question.
                - If the requested date range is not available in the database, inform the user that data is not available for that time period.
                - Use bold and large fonts to highlight keywords in your answer.
                - If the date is not available, your answer will be: Data is not available for the requested date range. Please modify your query to include a valid date range.
                - Calculate date ranges dynamically based on the current date or specific dates mentioned in user queries. Use relative time expressions such as "last month" or "past year".
                - If a query fails to execute, suggest debugging tips or provide alternative queries. Ensure to handle common SQL errors gracefully."
                - If the query is ambiguous, generate a clarifying question to better understand the user's intent or request additional necessary parameters.
                - Users may ask about positive tests and positivity rates. Positive tests are those where finalhivtestresult is positive.
                - Negative tests are those where either test1_result OR confirmatory_result are No 
                - If a user asks about positivity rate (or similar), calculate it as the number of positive tests divided by the total number of tests for the specified group
                - If a user asks about the percentage of something (for example, percentage of positive tests) calculate it as the number of patients for which that attribute is true divided by the total number of patients

                Additional Instructions:
                            
                Please confirm the variables names in the schema before generating a query

                You are required to use the following format, each taking one line:
                Question: Question here
                SQLQuery: SQL Query to run
            
                            
                The text-to-SQL system that might be required to handle queries related to calculating proportions within a dataset. Your system should be able to generate SQL queries to calculate the proportion of a certain category within a dataset table.

                hints only gives you the columns, please use the hint to calculate proportions.  When the input question requests information that includes multiple answers (for example, asking about states with the highest rates, greatest number of positives, etc.) by default, you should arrange those answers from highest to lowest unless otherwise told in the question.
                
                Example 1 :
                If a user asks, "In what states were the positivity rates highest excluding states with a rate of 100%?", your system should generate a SQL query like:

                'SELECT stateofresidence, COUNT(*) AS TotalTests, SUM(CASE WHEN finalhivtestresult = "Positive" THEN 1 ELSE 0 END) AS TotalPositives, SUM(CASE WHEN finalhivtestresult = "Positive" THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS PositivityRate FROM {tablename} GROUP BY stateofresidence HAVING PositivityRate < 100 ORDER BY PositivityRate DESC;' 
                
                Example 2:
                If a user asks, "What is the proportion of clients offered Prep who accepted Prep", your system should generate a SQL query like:

                SELECT SUM(CASE WHEN prep_offered = 'Yes' THEN 1 ELSE 0 END) AS TotalOfferedPrep, SUM(CASE WHEN prep_offered = 'Yes' AND prep_accepted = 'Yes' THEN 1 ELSE 0 END) AS TotalAcceptedPrep, SUM(CASE WHEN prep_offered = 'Yes' AND prepaccepted = 'Yes' THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN prepoffered = 'Yes' THEN 1 ELSE 0 END) AS ProportionAcceptedPrep FROM {tablename};
            """).format(tablename=this_table)

        first_identified_table = table_schema_objs[0]

        nl_sql_retriever = NLSQLRetriever(
            sql_database,
            tables=[this_table],
            return_raw=True
        )

        custom_prompt = ("Please calculate proportion when asked to, generate sql query that contains both the numbers and proportion. Only output sql query, do not attempt to generate an answer"
            f"You can refer to {custom_txt2sql_prompt} for examples and instructions on how to generate a SQL statement."
            f"Write a SQL query to answer the following question: {question}."#, Using the table {first_identified_table}."
            "Please take note of the column names which are in quotes and their description. It is essential that you consider the data dictionary and explanations of the variables" 
            "before crafting the query. Match the question to columns in the data and pay special attention to the description of that column in the schema."
            "Skip all operations/groupings that require dividing by zero."
            #"The following variables are booleans and can only be equal to True/1 or False/0: {cols}"
        )


        # Generate SQL query
        response = nl_sql_retriever.retrieve_with_metadata(custom_prompt)
        response_list, metadata_dict = response

        try:
            # Extract data and column keys
            data = list(response[1]['result'])
            col_keys = response[1]['col_keys']

            # Debug information
            logger.info("Fetched data: %s", data)
            logger.info("Column keys: %s", col_keys)

            # Check if data is not empty and column keys match the data columns
            if len(data) > 0 and len(col_keys) == len(data[0]):
                output_df = pd.DataFrame(data)
                output_df.columns = col_keys
            else:
                # Log the mismatch
                logger.error("Mismatch in data and column keys. Data length: %d, Column keys length: %d", len(data), len(col_keys))
                output_df = pd.DataFrame()  # Create an empty DataFrame as fallback

        except KeyError as e:
            logger.error("Key error while processing response: %s", e)
            output_df = pd.DataFrame()  # Create an empty DataFrame as fallback

        except ValueError as e:
            logger.error("Value error while creating DataFrame: %s", e)
            output_df = pd.DataFrame()  # Create an empty DataFrame as fallback

        except Exception as e:
            logger.error("Unexpected error: %s", e)
            output_df = pd.DataFrame()  # Create an empty DataFrame as fallback

        # Continue with the rest of the code
        query_string = "Question: \n" + question + "\n" + "\n" + "Query: \n" + metadata_dict.get("sql_query", "No SQL query") + "\n"
        query = metadata_dict.get("sql_query", "No SQL query")

        logger.info(query_string)
        logger.info(output_df)

        return query_string, output_df, question, query


    except BaseSSHTunnelForwarderError as e:
        logger.error("SSH Tunnel error: %s", e)
        raise
    except SQLAlchemyError as e:
        logger.error("Database error: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise
    finally:
        if 'server' in locals() and server.is_active:
            server.stop()

