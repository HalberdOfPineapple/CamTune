
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := '19a.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := 'SELECT MIN(n.name) AS voicing_actress,
       MIN(t.title) AS voiced_movie
FROM aka_name AS an,
     char_name AS chn,
     cast_info AS ci,
     company_name AS cn,
     info_type AS it,
     movie_companies AS mc,
     movie_info AS mi,
     name AS n,
     role_type AS rt,
     title AS t
WHERE ci.note IN (''(voice)'',
                  ''(voice: Japanese version)'',
                  ''(voice) (uncredited)'',
                  ''(voice: English version)'')
  AND cn.country_code =''[us]''
  AND it.info = ''release dates''
  AND mc.note IS NOT NULL
  AND (mc.note LIKE ''%(USA)%''
       OR mc.note LIKE ''%(worldwide)%'')
  AND mi.info IS NOT NULL
  AND (mi.info LIKE ''Japan:%200%''
       OR mi.info LIKE ''USA:%200%'')
  AND n.gender =''f''
  AND n.name LIKE ''%Ang%''
  AND rt.role =''actress''
  AND t.production_year BETWEEN 2005 AND 2009
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND mc.movie_id = mi.movie_id
  AND mi.movie_id = ci.movie_id
  AND cn.id = mc.company_id
  AND it.id = mi.info_type_id
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND n.id = an.person_id
  AND ci.person_id = an.person_id
  AND chn.id = ci.person_role_id;

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