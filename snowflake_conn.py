import snowflake.connector

def get_connection():
    """
    Establishes a connection to the Snowflake database.

    Returns:
        snowflake.connector.SnowflakeConnection: A connection object to the Snowflake database.
    """
    conn = snowflake.connector.connect(
        account = "BXHDBDX-GETWEAVE",
        user = "SARA.SONG",
        authenticator = "externalbrowser",
        role = "ANALYST",
        warehouse = "ANALYST_WH",
        database = "PROD",
        schema = "PUBLIC"
    )
    return conn 