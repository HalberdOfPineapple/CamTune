EXPLAIN -- using 1433771997 as a seed to the RNG


select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1994-01-01'
	and l_shipdate < date '1994-01-01' + interval '1' year
	and l_discount between 0.08 - 0.01 and 0.08 + 0.01
	and l_quantity < 24;

