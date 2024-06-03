import re
import math
import psycopg2

from camtune.utils import print_log
from camtune.database.dbconnector import DBConnector

from .variables import *
DBCONFIG = get_db_config()


class PostgresqlConnector(DBConnector):
    def __init__(
            self, 
            db_host:str=None,
            db_port:int=None,
            db_user:str=None,
            db_pwd:str=None,
            db_name:str=None,
    ):
        self.db_host = db_host if db_host else DBCONFIG.db_host
        self.db_port = db_port if db_port else DBCONFIG.db_port
        self.db_user = db_user if db_user else DBCONFIG.db_user
        self.db_pwd = db_pwd if db_pwd else DBCONFIG.db_pwd
        self.db_name = db_name if db_name else DBCONFIG.db_name

        self.conn: psycopg2.extensions.connection = None

    def connect(self):
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(
                host=DBCONFIG.db_host,
                user=DBCONFIG.db_user,
                password=DBCONFIG.db_pwd,
                database=DBCONFIG.db_name,
                port=DBCONFIG.db_port,
            )
            self.cursor = self.conn.cursor()

    def close_db(self):
        if self.cursor:
            self.cursor.close()

        if self.conn:
            self.conn.close()

    def fetch_results(self, sql, json=True):
        results = False
        self.connect()

        self.cursor.execute(sql)
        self.conn.commit()

        results = self.cursor.fetchall()

        if json:
            columns = [col[0] for col in self.cursor.description]
            results = [dict(zip(columns, row)) for row in results]

        self.close_db()
        return results

    def execute(self, sql):
        self.connect()

        self.cursor.execute(sql)

        self.close_db()
        return True

    def get_knob_value(self, k):
        sql = 'SHOW {};'.format(k)
        r = self.fetch_results(sql)

        if len(r) == 0 or k not in r[0]:
            raise ValueError(
                f"[PostgresqlConnector] Knob {k} is not correctly detected on DBMS")
        
        # sample r: [{'backend_flush_after': '856kB'}]
        return r[0][k]

    def set_knob_value(self, k, v):
        # If the knob is not set to the value, set it to the value by executing SQL command 'SET'
        if str(v).isdigit():
            sql = "SET {}={}".format(k, v)
        else:
            sql = "SET {}='{}'".format(k, v)

        try:
            self.execute(sql)
        except:
            print_log(f"[PostgresqlConnector] Failed when setting up knob {k} to {v}")