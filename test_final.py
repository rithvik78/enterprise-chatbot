#!/usr/bin/env python3

import requests
import time
import json
import threading

def poll_status():
    """Poll the latest status every 300ms"""
    print("📊 Starting status monitoring...")
    for i in range(50):  # Poll for up to 15 seconds
        try:
            response = requests.get('http://localhost:8080/api/status/latest')
            if response.status_code == 200:
                status = response.json()
                step = status.get('step', 'unknown')
                message = status.get('message', 'No message')
                print(f"🔄 Status {i+1}: [{step.upper()}] {message}")
                
                if step == 'complete':
                    print("✅ Query processing completed!")
                    break
            else:
                print(f"⏳ Status {i+1}: Waiting for query to start...")
        except Exception as e:
            print(f"❌ Status {i+1}: Error - {e}")
        
        time.sleep(0.3)

def test_final_system():
    """Test the complete system with live status and detailed answers"""
    print("🚀 Testing Final System: Live Status + Detailed Answers")
    print("=" * 60)
    
    # Start status polling in background
    status_thread = threading.Thread(target=poll_status, daemon=True)
    status_thread.start()
    
    # Give status polling a moment to start
    time.sleep(0.5)
    
    # Test query
    query_data = {
        "question": "who reports to Sarah Chen",
        "chat_history": []
    }
    
    print("📤 Sending query: 'who reports to Sarah Chen'")
    print("-" * 60)
    
    start_time = time.time()
    response = requests.post('http://localhost:8080/api/chat', 
                           headers={'Content-Type': 'application/json'},
                           json=query_data)
    end_time = time.time()
    
    print("-" * 60)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Query completed in {end_time - start_time:.1f} seconds!")
        print("=" * 60)
        print("📝 DETAILED ANSWER:")
        print("=" * 60)
        print(data['answer'])
        print("=" * 60)
        
        sources = data.get('sources', [])
        print(f"📚 SOURCES ({len(sources)} total):")
        for i, source in enumerate(sources[:5]):
            print(f"  {i+1}. {source['title']} - {source['score']}% match")
        if len(sources) > 5:
            print(f"  ... and {len(sources) - 5} more sources")
            
    else:
        print(f"❌ Query failed: {response.status_code}")
        print(response.text)
    
    # Wait for status thread to finish
    time.sleep(2)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_final_system()