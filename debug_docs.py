#!/usr/bin/env python3

from main import DocumentSearchSystem
import os
from dotenv import load_dotenv

load_dotenv()

def debug_document_processing():
    print("🔍 Debugging Document Processing")
    print("=" * 50)
    
    try:
        rag = DocumentSearchSystem()
        
        # Test Supabase vector manager directly
        print("\n1. Testing SupabaseVectorManager...")
        supabase_mgr = rag.supabase_vector
        
        # Call process_storage_documents to force processing
        print("   Calling process_storage_documents...")
        supabase_mgr.process_storage_documents()
        
        print("   ✅ process_storage_documents completed")
        
        # Check Pinecone after processing
        print("\n2. Checking Pinecone vectors...")
        stats = rag.index.describe_index_stats()
        print(f"   Pinecone vectors after processing: {stats.total_vector_count}")
        
        if stats.total_vector_count > 0:
            print("   ✅ Documents found in Pinecone")
            
            # Test search functionality
            print("\n3. Testing search...")
            query = "policy assets"
            results = rag.search_documents(query, top_k=3)
            print(f"   Search results for '{query}': {len(results)} documents")
            if results:
                for i, result in enumerate(results):
                    print(f"     {i+1}. {result.title} (score: {result.embedding})")
        else:
            print("   ❌ No documents found in Pinecone")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_document_processing()