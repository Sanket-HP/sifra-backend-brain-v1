# tasks/auto_sqlgen.py

import pandas as pd
import numpy as np


class AutoSQLGenerator:
    """
    -------------------------------------------------------
    Auto SQL Query Generator for SIFRA AI
    -------------------------------------------------------
    Features:
        ✔ Auto-generate SQL table schema
        ✔ Auto-generate SQL INSERT statements
        ✔ Auto SQL analysis queries
        ✔ SQL prediction formula for Linear Regression models
        ✔ MongoDB query templates (NoSQL)
    -------------------------------------------------------
    """

    def __init__(self):
        print("[TASK] Auto SQL Generator Ready")

    # -------------------------------------------------------
    # Generate SQL schema (CREATE TABLE)
    # -------------------------------------------------------
    def generate_schema(self, df, table_name="sifra_table"):
        schema = f"CREATE TABLE {table_name} (\n"

        for col in df.columns:
            dtype = df[col].dtype

            if np.issubdtype(dtype, np.integer):
                sql_type = "INT"
            elif np.issubdtype(dtype, np.floating):
                sql_type = "FLOAT"
            else:
                sql_type = "TEXT"

            schema += f"    {col} {sql_type},\n"

        schema = schema.rstrip(",\n") + "\n);"
        return schema

    # -------------------------------------------------------
    # SQL Insert generator
    # -------------------------------------------------------
    def generate_insert(self, df, table_name="sifra_table"):
        columns = ", ".join(df.columns)
        values = []

        for _, row in df.iterrows():
            row_vals = []
            for v in row:
                if isinstance(v, str):
                    row_vals.append(f"'{v}'")
                else:
                    row_vals.append(str(v))
            values.append("(" + ", ".join(row_vals) + ")")

        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES\n"
        insert_query += ",\n".join(values) + ";"

        return insert_query

    # -------------------------------------------------------
    # SQL Analysis Queries
    # -------------------------------------------------------
    def generate_analysis_queries(self, df, table_name="sifra_table"):
        queries = []

        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                queries.append(f"SELECT AVG({col}) AS avg_{col} FROM {table_name};")
                queries.append(f"SELECT MIN({col}), MAX({col}) FROM {table_name};")
                queries.append(f"SELECT COUNT(*) AS count_{col} FROM {table_name};")

        # Top rows
        queries.append(f"SELECT * FROM {table_name} LIMIT 10;")

        return queries

    # -------------------------------------------------------
    # SQL Prediction Query for Linear Regression
    # -------------------------------------------------------
    def generate_ml_query(self, model, feature_names, table_name="sifra_table"):
        """
        Generates SQL formula like:
        SELECT (coef1*x + coef2*y + intercept) AS predicted FROM table;
        """

        if not hasattr(model, "coef_"):
            return "Model is not linear regression — SQL prediction unavailable."

        coef = model.coef_
        intercept = model.intercept_

        # Build equation
        formula_parts = []
        for c, f in zip(coef, feature_names):
            formula_parts.append(f"{float(c)} * {f}")

        formula = " + ".join(formula_parts) + f" + {float(intercept)}"

        sql = f"SELECT {formula} AS predicted_value FROM {table_name};"
        return sql

    # -------------------------------------------------------
    # NoSQL (MongoDB) Queries
    # -------------------------------------------------------
    def generate_mongo_queries(self, df, collection_name="sifra_collection"):
        queries = {
            "insert_example": {
                "insertMany": df.to_dict(orient="records")
            },
            "find_all": {
                "find": {}
            },
            "range_query_example": {
                "find": {
                    df.columns[0]: {"$gt": df.iloc[:, 0].mean()}
                }
            }
        }
        return queries

    # -------------------------------------------------------
    # MAIN WRAPPER
    # -------------------------------------------------------
    def run(self, dataset):
        """
        Input: pandas DataFrame
        Output: SQL code bundle
        """

        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)

        schema = self.generate_schema(dataset)
        insert = self.generate_insert(dataset)
        analysis = self.generate_analysis_queries(dataset)
        mongo = self.generate_mongo_queries(dataset)

        return {
            "status": "success",
            "schema": schema,
            "insert_statements": insert,
            "analysis_queries": analysis,
            "mongo_queries": mongo,
            "message": "SQL generation completed."
        }

