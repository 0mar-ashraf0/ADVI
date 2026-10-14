import asyncio
from qdrant_client import QdrantClient
from motor.motor_asyncio import AsyncIOMotorClient

from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory

async def rescue_and_migrate():
    print("🚁 Starting Rescue Mission: Extracting data from Qdrant...")
    settings = get_settings()
    
    # 1. الاتصال بـ Qdrant (المسار المحلي اللي إنت شغال عليه)
    client_qdrant = QdrantClient(path=settings.VECTOR_DB_PATH)
    
    # 2. الاتصال بـ MongoDB و OpenAI
    mongo_conn = AsyncIOMotorClient(settings.MONGO_URI)
    db = mongo_conn[settings.MONGO_DB_NAME]
    
    llm_factory = LLMProviderFactory(settings)
    embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    # 3. هات كل الـ Collections اللي في Qdrant
    collections = client_qdrant.get_collections().collections
    print(f"🔍 Found {len(collections)} collections in Qdrant.")

    for coll in collections:
        coll_name = coll.name
        print(f"\n📦 Extracting from: {coll_name}")
        
        # سحب كل البيانات من Qdrant (بما فيها النص الأصلي)
        points, _ = client_qdrant.scroll(collection_name=coll_name, limit=1000, with_payload=True)
        
        if not points:
            print(f"⚠️ Collection {coll_name} is empty.")
            continue

        print(f"📑 Found {len(points)} points. Re-embedding and saving to MongoDB...")
        
        new_vectors = []
        texts = []
        metadatas = []
        ids = []

        for p in points:
            # استخراج النص من الـ Payload (حسب التسمية في مشروع ADVI)
            text = p.payload.get("text") or p.payload.get("page_content") or p.payload.get("chunk_text")
            if not text: continue
            
            # 1. توليد الـ Embedding الجديد (384)
            new_vector = embedding_client.embed_text(text=text)
            
            # 2. تجهيز البيانات للرفع
            new_vectors.append(new_vector)
            texts.append(text)
            metadatas.append(p.payload)
            ids.append(p.id)

            # 3. حفظ نسخة احتياطية في MongoDB (عشان متضيعش تاني)
            await db.chunks.update_one(
                {"vector_id": str(p.id)},
                {"$set": {
                    "chunk_text": text,
                    "chunk_metadata": p.payload,
                    "project_id": coll_name.replace("collection_", ""),
                    "dimension": settings.EMBEDDING_MODEL_SIZE
                }},
                upsert=True
            )

        # 4. مسح الـ Collection القديمة وعمل واحدة جديدة بالـ Dimension الصح
        print(f"♻️ Recreating {coll_name} with {settings.EMBEDDING_MODEL_SIZE} dimensions...")
        client_qdrant.delete_collection(coll_name)
        
        from qdrant_client.http import models as rest
        client_qdrant.create_collection(
            collection_name=coll_name,
            vectors_config=rest.VectorParams(size=settings.EMBEDDING_MODEL_SIZE, distance=rest.Distance.COSINE)
        )

        # 5. رفع البيانات الجديدة
        client_qdrant.upsert(
            collection_name=coll_name,
            points=rest.Batch(
                ids=ids,
                vectors=new_vectors,
                payloads=metadatas
            )
        )
        print(f"✅ Collection {coll_name} is now updated and backed up!")

    print("\n🎉 MISSION ACCOMPLISHED! Your vector DB is now 384 and MongoDB is repopulated.")
    mongo_conn.close()

if __name__ == "__main__":
    asyncio.run(rescue_and_migrate())