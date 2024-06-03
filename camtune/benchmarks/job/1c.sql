
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := '1c.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := 'SELECT MIN(mc.note) AS production_note,
       MIN(t.title) AS movie_title,
       MIN(t.production_year) AS movie_year
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info_idx AS mi_idx,
     title AS t
WHERE ct.kind = ''production companies''
  AND it.info = ''top 250 rank''
  AND mc.note NOT LIKE ''%(as Metro-Goldwyn-Mayer Pictures)%''
  AND (mc.note LIKE ''%(co-production)%'')
  AND t.production_year >2010
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND it.id = mi_idx.info_type_id;

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
