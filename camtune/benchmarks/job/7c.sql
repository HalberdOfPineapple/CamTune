
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := '7c.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := 'SELECT MIN(n.name) AS cast_member_name,
       MIN(pi.info) AS cast_member_info
FROM aka_name AS an,
     cast_info AS ci,
     info_type AS it,
     link_type AS lt,
     movie_link AS ml,
     name AS n,
     person_info AS pi,
     title AS t
WHERE an.name IS NOT NULL
  AND (an.name LIKE ''%a%''
       OR an.name LIKE ''A%'')
  AND it.info =''mini biography''
  AND lt.link IN (''references'',
                  ''referenced in'',
                  ''features'',
                  ''featured in'')
  AND n.name_pcode_cf BETWEEN ''A'' AND ''F''
  AND (n.gender=''m''
       OR (n.gender = ''f''
           AND n.name LIKE ''A%''))
  AND pi.note IS NOT NULL
  AND t.production_year BETWEEN 1980 AND 2010
  AND n.id = an.person_id
  AND n.id = pi.person_id
  AND ci.person_id = n.id
  AND t.id = ci.movie_id
  AND ml.linked_movie_id = t.id
  AND lt.id = ml.link_type_id
  AND it.id = pi.info_type_id
  AND pi.person_id = an.person_id
  AND pi.person_id = ci.person_id
  AND an.person_id = ci.person_id
  AND ci.movie_id = ml.linked_movie_id;

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
