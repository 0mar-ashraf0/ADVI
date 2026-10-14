import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings

async def deep_scan():
    settings = get_settings()
    client = AsyncIOMotorClient(settings.MONGO_URI)
    dbs = await client.list_database_names()
    
    print("🔍 Starting Deep Scan for ADVI data...")
    found_something = False

    for db_name in dbs:
        if db_name in ['admin', 'local', 'config']: continue
        db = client[db_name]
        collections = await db.list_collection_names()
        
        for col_name in collections:
            count = await db[col_name].count_documents({})
            if count > 0:
                sample = await db[col_name].find_one()
                print(f"📂 Found: Database [{db_name}] -> Collection [{col_name}] ({count} docs)")
                print(f"   📝 Sample keys: {list(sample.keys())}")
                found_something = True
    
    if not found_something:
        print("❌ No data found in any collection. MongoDB is completely empty!")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(deep_scan())