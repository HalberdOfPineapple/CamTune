
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := '5c.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := 'SELECT MIN(t.title) AS american_movie
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info AS mi,
     title AS t
WHERE ct.kind = ''production companies''
  AND mc.note NOT LIKE ''%(TV)%''
  AND mc.note LIKE ''%(USA)%''
  AND mi.info IN (''Sweden'',
                  ''Norway'',
                  ''Germany'',
                  ''Denmark'',
                  ''Swedish'',
                  ''Denish'',
                  ''Norwegian'',
                  ''German'',
                  ''USA'',
                  ''American'')
  AND t.production_year > 1990
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND mc.movie_id = mi.movie_id
  AND ct.id = mc.company_type_id
  AND it.id = mi.info_type_id;

';

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
