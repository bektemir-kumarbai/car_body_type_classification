from decouple import config

db_username = config("POSTGRES_USER")
db_password = config("POSTGRES_PASSWORD")
db_host = config("POSTGRES_HOST")
db_name = config("POSTGRES_DB")
db_port = config("POSTGRES_PORT")

app_port = config("APP_PORT")
