#!/usr/bin/env python3

from main import DocumentSearchSystem, Document
import os
from dotenv import load_dotenv

load_dotenv()

def test_pinecone_direct():
    print("🔍 Testing Direct Pinecone Indexing")
    print("=" * 50)
    
    try:
        rag = DocumentSearchSystem()
        
        # Create test documents manually
        test_docs = [
            Document(
                id="test_asset_policy",
                title="01_Device_Security_and_Lost_Equipment_Policy.pdf",
                content="""
                Asset Management and Lost Equipment Policy
                
                If an employee loses any company assets including laptops, phones, or equipment:
                1. Report the loss immediately to IT department
                2. File an incident report within 24 hours
                3. The employee may be responsible for replacement costs
                4. Security review will be conducted
                5. New equipment will be issued after approval
                
                Lost equipment must be reported to prevent security breaches.
                """,
                source="Test Data"
            ),
            Document(
                id="test_policy_general",
                title="Asset Policy General",
                content="""
                Company policy for asset loss and equipment management.
                All employees must follow proper procedures when equipment is lost or stolen.
                Immediate reporting is required for all lost company property.
                """,
                source="Test Policy"
            )
        ]
        
        print(f"Creating {len(test_docs)} test documents...")
        
        # Process and store directly to Pinecone
        for doc in test_docs:
            print(f"Processing: {doc.title}")
            rag.process_and_store_documents([doc])
        
        # Check Pinecone stats
        stats = rag.index.describe_index_stats()
        print(f"\nPinecone vectors after indexing: {stats.total_vector_count}")
        
        if stats.total_vector_count > 0:
            print("✅ Documents successfully indexed to Pinecone!")
            
            # Test search
            print("\n🔍 Testing search...")
            query = "What's the policy if someone loses some assets"
            results = rag.search_documents(query, top_k=3)
            
            print(f"Search results for '{query}':")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.title}")
                print(f"     Content preview: {result.content[:100]}...")
                print(f"     Similarity: {getattr(result, 'similarity', 'N/A')}")
                print()
                
        else:
            print("❌ No documents indexed to Pinecone")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pinecone_direct()