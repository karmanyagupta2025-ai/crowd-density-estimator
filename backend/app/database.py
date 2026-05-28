from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGO_URI")

client = MongoClient(
    uri,
    server_api=ServerApi('1'),
    tls=True,
    tlsAllowInvalidCertificates=True
)

db = client["crowd_db"]

predictions_collection = db["predictions"]

try:
    client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print(e)