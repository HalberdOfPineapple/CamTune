from typing import Dict
from .base_database import BaseDatabase
from .postgresql_db import PostgresqlDB
from .workloads import *

DBMS_SET: Dict[str, BaseDatabase] = {
    'postgresql': PostgresqlDB,
}