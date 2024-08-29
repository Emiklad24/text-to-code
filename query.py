#Imports
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
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
import os
import csv

from sshtunnel import SSHTunnelForwarder


def process_question(question: str):
    #Connect to the server clone via SSH Tunnel --> Adjust to private key location/login as needed
    server = SSHTunnelForwarder(
        ('******', 2422),
        ssh_username="****",
        ssh_password='*****',
        ssh_private_key='******',
        ssh_private_key_password = "****",
        remote_bind_address=('localhost', 5432),
        local_bind_address=('localhost',6543), # could be any available port
        )

    #Create engine connection from that 
    server.start()
    local_port = str(server.local_bind_port)

    #Specify schema --> need to identify expanded_hts_prep else tables hidden
    dbschema='expanded_hts_prep,public' # Searches left-to-right 
    #engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format("isaac", "*!NX5nEr!bU)bYrm", "127.0.0.1", local_port, "lamisplus_ods_dwh"))
    engine = create_engine('postgresql://isaac:*!NX5nEr!bU)bYrm@127.0.0.1:6543/lamisplus_ods_dwh',
    connect_args={'options': '-csearch_path={}'.format(dbschema)})


    #Choose llm
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

    #Specify current table (generated from is_current table variable in period table)
    period_df = pd.read_sql("SELECT * FROM expanded_hts_prep.period;", engine)
    this_period = period_df['periodcode'][period_df['is_current']].to_string(index = False).lower()
    this_table = "expanded_hts_weekly_" + this_period
    #this_table = "expanded_hts_weekly_2024w32"

    #Create SQLDatabase object
    sql_database = SQLDatabase(engine, include_tables=[this_table])

    #Ingest data dictionary/generate dictionary from it for use in table description

    #Read data dictionary from CSV file 
    with open('Nigeria_Text2Code_DataDictionary.csv', 'r') as file:
        reader = csv.DictReader(file)

        data_dict = {}
        for row in reader:
            key = row["parent"] + row["name"]  
            data_dict[key] = row


    #Manually set context text
    hts_table_text = (
        "This table gives information regarding HIV Testing Services (HTS)."
        "Users may ask about positive tests and positivity rates. Positive tests are those where finalhivtestresult is positive. Negative tests are those where finalhivtestresult is negative or missing or null"
        "If a user asks about positivity rate (or similar), calculate it as the number of positive tests divided by the total number of tests for the specified group"
    )

    #Set up table schema
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        (SQLTableSchema(table_name=this_table, context_str=('description of the table: ' + hts_table_text + '. These are columns in the table and their descriptions: ' + str(data_dict))))
    ]

    #Set open AI key *Flag if needs to be hidden
    os.environ["OPENAI_API_KEY"] = ""


    #Create query engine to generate SQL queries
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[this_table]
    )

    #Create object index
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )

    #Set up custom prompt
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

    #Keep table identification variable for when we expand to multiple tables
    first_identified_table = table_schema_objs[0]

    #Set up retriever from table
    nl_sql_retriever = NLSQLRetriever(
        sql_database,
        tables = [this_table],
        return_raw = True
    )

    # question = "What is the proportion of clients offered Prep who accepted Prep for each key population target group?"#Please return the states with the lowest positivity rates and the rates themselves"

    #Put in larger prompt
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

    #Put response into data frame
    output_df = pd.DataFrame(list(response[1]['result']))
    output_df.columns = response[1]['col_keys']

    #Put question/query into single string
    query_string = "Question: \n"+ question + "\n" + "\n"+ "Query: \n" + metadata_dict["sql_query"] + "\n"
    query = metadata_dict["sql_query"]
    #Print relevant info
    print(query_string)
    print(output_df)

    
    return query_string, output_df, question, query