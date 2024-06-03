import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_RAW_DIR = os.path.join(BASE_DIR, 'job_raw')
JOB_DIR = os.path.join(BASE_DIR, 'job')
if not os.path.exists(JOB_DIR):
    os.makedirs(JOB_DIR, exist_ok=True)

TIME_MEAS_TEMP = """
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := '{}';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := '{}';

	-- Start the timer
    start_time := clock_timestamp();

	-- Execute the query
    EXECUTE query;

    -- End the timer
    end_time := clock_timestamp();

    -- Call the function with the query name and the query
    RETURN QUERY SELECT query_name, EXTRACT(EPOCH FROM (end_time - start_time));
END;
$$ LANGUAGE plpgsql;

SELECT * FROM get_query_results();
"""

query_files = []
for root, dirs, files in os.walk(JOB_RAW_DIR):
    query_files = [file for file in files if file.endswith('.sql')]
    break

for query_file in query_files:
    with open(os.path.join(JOB_RAW_DIR, query_file), 'r') as f:
        query:str = f.read()

    # replace all single quote with double single quote
    query = query.replace("\'", "\'\'")

    time_query = TIME_MEAS_TEMP.format(query_file, query)
    time_query_file = os.path.join(JOB_DIR, f'{query_file.split(".")[0]}.sql')
    with open(time_query_file, 'w') as f:
        f.write(time_query)