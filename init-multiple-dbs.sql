CREATE EXTENSION IF NOT EXISTS dblink;

DO
$do$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow') THEN
      PERFORM dblink_exec('dbname=postgres user=' || current_user,
         'CREATE DATABASE airflow');
   END IF;
END
$do$;

DO
$do$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow') THEN
      PERFORM dblink_exec('dbname=postgres user=' || current_user,
         'CREATE DATABASE mlflow');
   END IF;
END
$do$;
