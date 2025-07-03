#!/usr/bin/env python3

from main import DocumentSearchSystem, Document
import os
from dotenv import load_dotenv

load_dotenv()

def test_pinecone_debug():
    print("🔍 Testing Pinecone with Debug")
    print("=" * 50)
    
    try:
        rag = DocumentSearchSystem()
        
        # Check initial state
        stats = rag.index.describe_index_stats()
        print(f"Initial Pinecone vectors: {stats.total_vector_count}")
        
        # Create simple test document
        test_doc = Document(
            id="simple_test",
            title="Simple Test Document",
            content="This is a test document about asset loss policy. When employees lose equipment, they must report it immediately.",
            source="Test"
        )
        
        print(f"\nProcessing document: {test_doc.title}")
        print(f"Content length: {len(test_doc.content)}")
        print(f"Document ID: {test_doc.id}")
        
        # Check if already processed
        if test_doc.id in rag.processed_docs:
            print(f"Document {test_doc.id} already processed, removing from cache")
            del rag.processed_docs[test_doc.id]
        
        # Process step by step
        print("\n1. Chunking content...")
        chunks = rag.chunk_content_adaptively(test_doc.content)
        print(f"   Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i}: {chunk[:100]}...")
        
        print("\n2. Creating embeddings...")
        vectors_to_upsert = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{test_doc.id}_chunk_{i}"
            print(f"   Processing chunk {i} with ID: {chunk_id}")
            
            # Create embedding
            embedding = rag.embedder.encode([chunk])[0].tolist()
            print(f"   Embedding created: {len(embedding)} dimensions")
            
            # Prepare metadata
            metadata = {
                'title': test_doc.title,
                'content': chunk,
                'source': test_doc.source,
                'parent_doc_id': test_doc.id,
                'chunk_index': i
            }
            
            vectors_to_upsert.append((chunk_id, embedding, metadata))
        
        print(f"\n3. Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        
        if vectors_to_upsert:
            result = rag.index.upsert(vectors=vectors_to_upsert)
            print(f"   Upsert result: {result}")
            
            # Add to processed docs
            rag.processed_docs[test_doc.id] = True
            rag.save_cache()
            print("   Cache saved")
        
        # Check final state
        print("\n4. Checking final state...")
        stats = rag.index.describe_index_stats()
        print(f"   Final Pinecone vectors: {stats.total_vector_count}")
        
        if stats.total_vector_count > 0:
            print("   ✅ Success! Document indexed to Pinecone")
            
            # Test search
            print("\n5. Testing search...")
            query = "asset loss policy"
            results = rag.search_documents(query, top_k=3)
            print(f"   Search results for '{query}': {len(results)} found")
            
            for result in results:
                print(f"     - {result.title}: {result.content[:50]}...")
        else:
            print("   ❌ No vectors found in Pinecone")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pinecone_debug()