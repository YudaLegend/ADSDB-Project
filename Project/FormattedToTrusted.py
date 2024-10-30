import duckdb


def UnifyTables(tables_to_combine, table_name, con, trusted_conn):
    # Obtain the column names of the first table to ensure the structure
    first_table = tables_to_combine[0]
    columns_query = f"SELECT * FROM {first_table} LIMIT 0"  # Obtain the column names
    columns = con.execute(columns_query).df().columns.tolist()
    
    # Create the query to combine the tables
    combine_query = f"SELECT {', '.join(columns)} FROM {first_table}"
    
    # Ensure that each table has the same column structure in the same position
    for table in tables_to_combine[1:]:
        combine_query += f" UNION ALL SELECT {', '.join(columns)} FROM {table}"

    # Execute the query in the formatted zone database to obtain the combined data
    combined_data = con.execute(combine_query).df()

    # Save the DataFrame directly in the trusted zone database
    trusted_conn.register('combined_data_df', combined_data)  # Register the DataFrame like a temporal table

    # Create a new table in the trusted zone using like a temporal table
    trusted_conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM combined_data_df")


def connectToDatabases():
    # Specify the path where DuckDB will store the database
    db_path_formatted = './Project/Formatted Zone/formatted_zone.duckdb'

    # Connect to the DuckDB database
    con = duckdb.connect(database=db_path_formatted, read_only=False)

    # Specify the path where DuckDB will store the database
    db_path_trusted = './Project/Trusted Zone/trusted_zone.duckdb'

    trusted_conn =  duckdb.connect(database=db_path_trusted, read_only=False)

    return con, trusted_conn

