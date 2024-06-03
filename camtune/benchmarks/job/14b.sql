
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := '14b.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := 'SELECT MIN(mi_idx.info) AS rating,
       MIN(t.title) AS western_dark_production
FROM info_type AS it1,
     info_type AS it2,
     keyword AS k,
     kind_type AS kt,
     movie_info AS mi,
     movie_info_idx AS mi_idx,
     movie_keyword AS mk,
     title AS t
WHERE it1.info = ''countries''
  AND it2.info = ''rating''
  AND k.keyword IN (''murder'',
                    ''murder-in-title'')
  AND kt.kind = ''movie''
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
  AND mi_idx.info > ''6.0''
  AND t.production_year > 2010
  AND (t.title LIKE ''%murder%''
       OR t.title LIKE ''%Murder%''
       OR t.title LIKE ''%Mord%'')
  AND kt.id = t.kind_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mi_idx.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mi_idx.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id;

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
