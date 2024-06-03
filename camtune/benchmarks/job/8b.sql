
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := '8b.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := 'SELECT MIN(an.name) AS acress_pseudonym,
       MIN(t.title) AS japanese_anime_movie
FROM aka_name AS an,
     cast_info AS ci,
     company_name AS cn,
     movie_companies AS mc,
     name AS n,
     role_type AS rt,
     title AS t
WHERE ci.note =''(voice: English version)''
  AND cn.country_code =''[jp]''
  AND mc.note LIKE ''%(Japan)%''
  AND mc.note NOT LIKE ''%(USA)%''
  AND (mc.note LIKE ''%(2006)%''
       OR mc.note LIKE ''%(2007)%'')
  AND n.name LIKE ''%Yo%''
  AND n.name NOT LIKE ''%Yu%''
  AND rt.role =''actress''
  AND t.production_year BETWEEN 2006 AND 2007
  AND (t.title LIKE ''One Piece%''
       OR t.title LIKE ''Dragon Ball Z%'')
  AND an.person_id = n.id
  AND n.id = ci.person_id
  AND ci.movie_id = t.id
  AND t.id = mc.movie_id
  AND mc.company_id = cn.id
  AND ci.role_id = rt.id
  AND an.person_id = ci.person_id
  AND ci.movie_id = mc.movie_id;

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
