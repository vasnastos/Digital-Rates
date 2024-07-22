import sqlite3
import pandas as pd
import os
from typing import List, Tuple, Any
import argparse,json

def read_generated_data(file):
    try:
        if os.path.isdir(file):
            subdata=list()
            filenames=[os.path.join(file,filename) for filename in os.listdir(file)]
            for filename in filenames:
                subdata.append(pd.read_csv(filepath_or_buffer=filename))
            return pd.concat(subdata)
        elif os.path.isfile(file):
            return pd.read_csv(filepath_or_buffer=file)
        else:
            raise NotImplementedError("Method with file:{file} not implement yet")
    except FileNotFoundError as fe:
        print(str(fe))
            
class Database:
    def __init__(self,db_name) -> None:
        self.database_name=db_name
        self.connection=None
        try:
            self.connection=sqlite3.connect(self.database_name)
        except sqlite3.Error as e:
            print(e)
        self.cursor=self.connection.cursor()
        
    def create_table(self,table_name:str,columns: List[str]):
        columns_with_types=", ".join(columns)
        query=f"CREATE TABLE IF NOT EXISTS {table_name}({columns_with_types})"
        self.cursor.execute(query)
        self.connection.commit()
    
    def insert(self,table_name:str,columns:List[str],values:Tuple[Any]):
        placeholders=", ".join(["?" for _ in values])
        columns_joined=", ".join(columns)
        query=f"INSERT INTO {table_name} ({columns_joined}) VALUES({placeholders})"
        self.cursor.execute(query,values)
        self.connection.commit()
    
    def fetch_all(self,table_name):
        query=f"SELECT * FROM {table_name}"
        self.cursor.execute(query)
        data=self.cursor.fetchall()
        return data
    
    def fetch_all_as_dataframe(self,table_name:str):
        query=f'SELECT * FROM {table_name}'
        return pd.read_sql_query(query,self.connection)

    def delete(self,table_name:str,condition:str):
        query=f'DELETE FROM {table_name} WHERE {condition}'
        self.cursor.execute(query)
        self.connection.commit()
    
    def column_names_except_autoincrement_ones(self,table_name:str):
        query=f"PRAGMA table_info({table_name})"
        self.cursor.execute(query)
        columns_info=self.cursor.fetchall()
        return [info[1] for info in columns_info if not (info[5]==1 and info[1].lower()=="id")]
        
    def close(self):
        self.cursor.close()
        self.connection.close()

class QNDatabase:
    def __init__(self) -> None:
        pass


def get_args():
    parser=argparse.ArgumentParser(prog="DigitalRates-database")
    parser.add_argument("--config",type=str,help="load database configuration")
    return parser.parse_args()


if __name__=='__main__':
    args=get_args()
    with open(args.config,'r') as f:
        config=json.load(f)
    
    if "database" in config:
        db=Database(config["database"]["db_name"])
        for table in config["database"]["tables"]:
            columns_with_types=[f"{column['name']} {column['type']} {column.get('constraints','')}" for column in table["columns"]]
            db.create_table(table["table_name"],columns_with_types)
            if "sample_data" in table:
                data=read_generated_data(table["sample_data"])
                for _,row in data.iterrows():
                    db.insert(table["table_name"],columns=[column["name"] for column in table["columns"] if 'AUTOINCREMENT' not in column.get('constraints','')],values=list(row.values))
        db.close()
        
    if "insert" in config:
        db=Database(db_name=config["insert"]["db_path"])
        table_name=config["insert"]["table_name"]
        files=config["insert"]["data"].split(";")
        columns_without_autoincrement=db.column_names_except_autoincrement_ones()
        for file in files:
            with open(file,'r') as reader:
                for line in reader:
                    data=line.strip().split(",")
                    db.insert(table_name,columns=columns_without_autoincrement,values=data)
        db.close()
    
    db.close()