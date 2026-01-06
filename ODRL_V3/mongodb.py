# mongodb.py
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from bson import ObjectId
from typing import List, Dict, Any

class MongoDBManager:
    def __init__(self,
                 mongo_uri: str = "mongodb://localhost:27017/",
                 mongo_db_name: str = "default_rules_db"
                ):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[mongo_db_name]
        print(f"MongoDBManager initialized for DB: '{self.db.name}'")

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

    async def fetch_all_rules(self, collection_name: str, query: Dict = None, projection: Dict = None) -> List[Dict]:
        collection = self.get_collection(collection_name)
        if query is None:
            query = {}
        try:
            cursor = collection.find(query, projection)
            documents = await cursor.to_list(length=None)
            print(f"Fetched {len(documents)} documents from {collection.name}")
            return documents
        except Exception as e:
            print(f"Error fetching from collection '{collection.name}': {e}")
            raise RuntimeError(f"数据库读取失败: {str(e)}")

    async def insert_generated_odrl(self, collection_name: str, odrl_data: Dict[str, Any]) -> ObjectId:
        collection = self.get_collection(collection_name)
        try:
            document_to_insert = {**odrl_data, "generation_time": datetime.now()}
            result = await collection.insert_one(document_to_insert)
            print(f"Inserted document ID {result.inserted_id} into {collection.name}")
            return result.inserted_id
        except Exception as e:
            print(f"Error inserting ODRL into collection '{collection.name}': {e}")
            raise RuntimeError(f"ODRL 数据数据库插入失败: {str(e)}")

    async def insert_many_generated_odrls(self, collection_name: str, odrl_data_list: List[Dict]) -> List[ObjectId]:
        collection = self.get_collection(collection_name)
        if not odrl_data_list:
            return []
        try:
            documents = [
                {**odrl_doc, "generation_time": datetime.now()} for odrl_doc in odrl_data_list
            ]
            result = await collection.insert_many(documents)
            print(f"Inserted {len(result.inserted_ids)} documents into {collection.name}")
            return result.inserted_ids
        except Exception as e:
            print(f"Error inserting multiple ODRLs into collection '{collection.name}': {e}")
            raise RuntimeError(f"ODRL 数据数据库批量插入失败: {str(e)}")

    async def close_connection(self):
        if self.client:
            self.client.close()
            print(f"MongoDB connection to {self.db.name} closed.")