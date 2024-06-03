#!/usr/bin/env bash
# args: $1 = benchmark, $2 = config file, $3 = output directory
# example: java -jar benchbase.jar -b ycsb -c config/postgres/sample_ycsb_config.xml --execute=true

cd /home/wl446/benchbase/target/benchbase-postgres
java -jar benchbase.jar -s 1 -b $1 -c $2 -d $3 --execute=true