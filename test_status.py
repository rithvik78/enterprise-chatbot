#!/usr/bin/env python3

import requests
import time
import json

def test_status_system():
    """Test the dynamic status system"""
    print("Testing dynamic status system...")
    
    # Start a query
    query_data = {
        "question": "who reports to Sarah Chen",
        "chat_history": []
    }
    
    print("Sending query...")
    response = requests.post('http://localhost:8080/api/chat', 
                           headers={'Content-Type': 'application/json'},
                           json=query_data)
    
    if response.status_code == 200:
        data = response.json()
        query_id = data.get('query_id')
        print(f"Got query ID: {query_id}")
        
        if query_id:
            # Poll status updates
            print("\nPolling status updates...")
            for i in range(20):  # Poll for up to 10 seconds
                status_response = requests.get(f'http://localhost:8080/api/status/{query_id}')
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"Status {i+1}: {status['step']} - {status['message']}")
                    
                    if status['step'] == 'complete':
                        print("Query completed!")
                        break
                else:
                    print(f"Status check failed: {status_response.status_code}")
                
                time.sleep(0.5)
        
        # Show final results
        print(f"\nFinal answer: {data['answer'][:100]}...")
        print(f"Sources found: {len(data.get('sources', []))}")
        for i, source in enumerate(data.get('sources', [])[:3]):
            print(f"  {i+1}. {source['title']} - {source['score']}% match")
    else:
        print(f"Query failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_status_system()