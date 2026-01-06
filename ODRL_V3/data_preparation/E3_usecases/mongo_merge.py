# -*- coding: utf-8 -*-

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# --- 全局配置变量 (请替换成您的实际值) ---

# 源数据库 1 的信息
SOURCE_MONGO_URI_1 = "mongodb://localhost:27017/"
SOURCE_DB_NAME_1 = "odrl3_final_e3"
SOURCE_COLLECTION_NAME_1 = "e3_41nano_direct_3"

# 源数据库 2 的信息
SOURCE_MONGO_URI_2 = "mongodb://localhost:27017/"
SOURCE_DB_NAME_2 = "odrl3_final_e3"
SOURCE_COLLECTION_NAME_2 = "e3_41nano_split_3"

# 源数据库 3 的信息
# 如果这个源和前一个在同一个MongoDB实例上，URI可以保持一致
SOURCE_MONGO_URI_3 = "mongodb://localhost:27017/"
SOURCE_DB_NAME_3 = "odrl3_final_e3"
SOURCE_COLLECTION_NAME_3 = "e3_41nano_rewrite_4"

# 目标数据库的信息
DESTINATION_MONGO_URI = "mongodb://localhost:27017/"
DESTINATION_DB_NAME ="odrl3_final_e3"
DESTINATION_COLLECTION_NAME = "e3_41nano_all_4_334"


def aggregate_collections():
    """
    连接到各个MongoDB源，读取文档，并全部汇聚到目标集合中。
    """
    # 将所有源信息放入一个列表中，方便循环处理
    sources = [
        {"uri": SOURCE_MONGO_URI_1, "db": SOURCE_DB_NAME_1, "coll": SOURCE_COLLECTION_NAME_1},
        {"uri": SOURCE_MONGO_URI_2, "db": SOURCE_DB_NAME_2, "coll": SOURCE_COLLECTION_NAME_2},
        {"uri": SOURCE_MONGO_URI_3, "db": SOURCE_DB_NAME_3, "coll": SOURCE_COLLECTION_NAME_3},
    ]

    destination_client = None
    total_inserted_count = 0

    try:
        # 1. 连接到目标数据库
        print(f"正在连接到目标数据库: {DESTINATION_MONGO_URI}...")
        destination_client = MongoClient(DESTINATION_MONGO_URI)
        # 验证连接
        destination_client.admin.command('ping')
        destination_db = destination_client[DESTINATION_DB_NAME]
        destination_collection = destination_db[DESTINATION_COLLECTION_NAME]
        print("目标数据库连接成功。")

        # 可选：如果希望每次运行时都清空目标集合，取消下面这行代码的注释
        # print(f"正在清空目标集合: '{DESTINATION_COLLECTION_NAME}'...")
        # destination_collection.delete_many({})

        # 2. 遍历所有源并迁移数据
        for i, source_info in enumerate(sources, 1):
            source_client = None
            print(f"\n--- 开始处理源 {i} ---")
            print(f"数据库: '{source_info['db']}', 集合: '{source_info['coll']}'")

            try:
                # 连接源数据库
                print(f"正在连接到源 {i}: {source_info['uri']}...")
                source_client = MongoClient(source_info['uri'])
                source_client.admin.command('ping')
                source_db = source_client[source_info['db']]
                source_collection = source_db[source_info['coll']]
                print(f"源 {i} 连接成功。")

                # 读取所有文档
                print("正在读取文档...")
                # 使用 list() 将所有文档加载到内存中
                documents_cursor = source_collection.find({})
                documents_to_insert = []
                
                # 关键步骤：移除原始 _id，以便MongoDB在目标集合中生成新的_id
                for doc in documents_cursor:
                    if '_id' in doc:
                        del doc['_id']
                    documents_to_insert.append(doc)
                
                doc_count = len(documents_to_insert)
                print(f"从源 {i} 读取了 {doc_count} 个文档。")

                # 如果有文档，则批量插入到目标集合
                if documents_to_insert:
                    print("正在将文档批量插入目标集合...")
                    result = destination_collection.insert_many(documents_to_insert, ordered=False)
                    inserted_count = len(result.inserted_ids)
                    total_inserted_count += inserted_count
                    print(f"成功插入 {inserted_count} 个文档到 '{DESTINATION_COLLECTION_NAME}'。")
                else:
                    print("没有文档需要插入。")

            except ConnectionFailure as e:
                print(f"错误：无法连接到源 {i} 的MongoDB。请检查URI: {source_info['uri']}\n{e}")
                continue # 继续处理下一个源
            except OperationFailure as e:
                print(f"错误：操作失败。请检查数据库/集合名称或权限。\n{e}")
                continue
            finally:
                # 确保关闭当前源的连接
                if source_client:
                    source_client.close()
                    print(f"源 {i} 的连接已关闭。")

        print(f"\n--- 任务完成 ---")
        print(f"总共成功插入了 {total_inserted_count} 个文档到数据库 '{DESTINATION_DB_NAME}' 的集合 '{DESTINATION_COLLECTION_NAME}' 中。")

    except ConnectionFailure as e:
        print(f"致命错误：无法连接到目标数据库。请检查URI: {DESTINATION_MONGO_URI}\n{e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    finally:
        # 确保关闭目标数据库的连接
        if destination_client:
            destination_client.close()
            print("目标数据库连接已关闭。")


if __name__ == "__main__":
    aggregate_collections()