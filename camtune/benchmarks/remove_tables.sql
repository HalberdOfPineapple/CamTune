DO $$
DECLARE
    i int;
BEGIN
    -- Dropping tables in the range sbtest10 to sbtest118
    FOR i IN 2..9 LOOP
        EXECUTE format('DROP TABLE IF EXISTS public.sbtest%s CASCADE', i);
        EXECUTE format('DROP SEQUENCE IF EXISTS public.sbtest%s_id_seq CASCADE', i);
    END LOOP;
END $$;
