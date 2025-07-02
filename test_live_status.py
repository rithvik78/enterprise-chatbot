#!/usr/bin/env python3

import requests
import time
import json
import threading

def poll_status():
    """Poll the latest status every 300ms"""
    print("Starting status polling...")
    for i in range(40):  # Poll for up to 12 seconds
        try:
            response = requests.get('http://localhost:8080/api/status/latest')
            if response.status_code == 200:
                status = response.json()
                print(f"Status {i+1}: {status.get('step', 'unknown')} - {status.get('message', 'No message')}")
            else:
                print(f"Status {i+1}: No active queries")
        except:
            print(f"Status {i+1}: Error checking status")
        
        time.sleep(0.3)

def test_live_status():
    """Test the live status system"""
    print("Testing live status system...")
    
    # Start status polling in background
    status_thread = threading.Thread(target=poll_status, daemon=True)
    status_thread.start()
    
    # Give a moment for status polling to start
    time.sleep(0.5)
    
    # Start query
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
        print(f"\nQuery completed!")
        print(f"Answer preview: {data['answer'][:100]}...")
        print(f"Sources: {len(data.get('sources', []))}")
        for i, source in enumerate(data.get('sources', [])[:3]):
            print(f"  {i+1}. {source['title']} - {source['score']}% match")
    else:
        print(f"Query failed: {response.status_code}")
    
    # Wait for status thread to finish
    time.sleep(2)

if __name__ == "__main__":
    test_live_status()