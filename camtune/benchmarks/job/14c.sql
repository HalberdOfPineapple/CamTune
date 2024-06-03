
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := '14c.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := 'SELECT MIN(mi_idx.info) AS rating,
       MIN(t.title) AS north_european_dark_production
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
  AND k.keyword IS NOT NULL
  AND k.keyword IN (''murder'',
                    ''murder-in-title'',
                    ''blood'',
                    ''violence'')
  AND kt.kind IN (''movie'',
                  ''episode'')
  AND mi.info IN (''Sweden'',
                  ''Norway'',
                  ''Germany'',
                  ''Denmark'',
                  ''Swedish'',
                  ''Danish'',
                  ''Norwegian'',
                  ''German'',
                  ''USA'',
                  ''American'')
  AND mi_idx.info < ''8.5''
  AND t.production_year > 2005
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
