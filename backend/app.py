#!/usr/bin/env python3

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import threading
import uuid
import time
from openai import OpenAI

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from main import DocumentSearchSystem

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

print("Starting RAG Backend API...")
rag_system = None

# Global status tracking for live updates
query_status = {}

def update_query_status(query_id, step, message, details=None):
    query_status[query_id] = {
        'step': step,
        'message': message,
        'details': details,
        'timestamp': time.time()
    }

def get_query_status(query_id):
    return query_status.get(query_id, None)

def initialize_rag():
    global rag_system
    try:
        print("Loading RAG system...")
        rag_system = DocumentSearchSystem()
        rag_system._skip_judge_evaluation = True
        rag_system.update_knowledge_base()
        print("RAG system ready!")
    except Exception as e:
        print(f"ERROR: RAG system failed to load: {e}")

threading.Thread(target=initialize_rag, daemon=True).start()
print("Backend API server ready on port 8080")

def rewrite_query(question):
    rewrite_prompt = f"""You are an expert at query expansion for document search. Given a user question, rewrite it in 3 different styles to improve retrieval of relevant documents.

Original question: "{question}"

Create 3 variations:
1. Formal/Technical style: Use professional terminology and specific keywords
2. Conversational style: Rephrase as natural language people might use in conversation  
3. Keyword-focused style: Extract key concepts and create a search-optimized version

Return only the 3 rewritten queries, one per line, without labels or numbers.
"""
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Agentic RAG",
            },
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[
                {"role": "user", "content": rewrite_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        response = completion.choices[0].message.content.strip()
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        queries = [question]
        for line in lines[:3]:
            if line and line != question:
                queries.append(line)
        
        return queries[:4]
        
    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return [question]

def judge_and_filter_results(results, question, max_results=50):
    if not results or len(results) == 0:
        return results
    
    evaluation_prompt = f"""You are an expert at evaluating search result relevance. Given a user question and a list of search results, score each result's relevance on a scale of 0-100.

Question: "{question}"

Search Results to evaluate:
"""
    
    for i, result in enumerate(results[:20]):
        content_preview = result.get('content', '')[:200]
        title = result.get('title', 'Untitled')
        source = result.get('source', 'Unknown')
        
        evaluation_prompt += f"""

{i+1}. Title: {title}
Source: {source}
Content Preview: {content_preview}..."""
    
    evaluation_prompt += f"""

Please provide ONLY a comma-separated list of scores (0-100) for each result in order. For example: 85,92,15,67,45...

Be strict - only give high scores (80+) to results that directly answer the question. Give medium scores (40-79) to somewhat relevant results. Give low scores (0-39) to irrelevant results."""
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Agentic RAG Judge",
            },
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        scores_text = completion.choices[0].message.content.strip()
        
        try:
            scores = [int(float(s.strip())) for s in scores_text.split(',') if s.strip()]
        except:
            import re
            scores = [int(m.group()) for m in re.finditer(r'\d+', scores_text)]
        
        scored_results = []
        for i, result in enumerate(results[:len(scores)]):
            if i < len(scores):
                result_copy = result.copy()
                result_copy['judge_score'] = scores[i]
                result_copy['original_score'] = result.get('score', 0)
                scored_results.append(result_copy)
        
        scored_results.sort(key=lambda x: x.get('judge_score', 0), reverse=True)
        
        # For now, let's be very inclusive and use LLM filtering smartly
        # Include ALL results first, then let LLM decide what's relevant
        filtered_results = scored_results.copy()
        
        # If we have too many results, use LLM to filter intelligently
        if len(filtered_results) > max_results:
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
                
                # Create a concise prompt for LLM filtering
                results_summary = []
                for i, r in enumerate(filtered_results[:30]):  # Evaluate top 30
                    content_preview = r.get('content', '')[:80].replace('\n', ' ')
                    results_summary.append(f"{i+1}. {r.get('title', 'Unknown')} (Score: {r.get('judge_score', 0)}%) - {content_preview}...")
                
                relevance_prompt = f"""Question: "{question}"

Which results are relevant? Return ONLY the numbers (1,2,3...) of relevant results:

{chr(10).join(results_summary)}

Numbers of relevant results:"""
                
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Agentic RAG Filter",
                    },
                    model="meta-llama/llama-3.1-8b-instruct:free",
                    messages=[{"role": "user", "content": relevance_prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                
                response = completion.choices[0].message.content.strip()
                relevant_indices = []
                
                # Parse the response more robustly
                import re
                numbers = re.findall(r'\b(\d+)\b', response)
                for num in numbers:
                    try:
                        idx = int(num) - 1  # Convert to 0-based index
                        if 0 <= idx < len(filtered_results):
                            relevant_indices.append(idx)
                    except:
                        continue
                
                # If LLM selected results, use those; otherwise keep high-scoring ones
                if relevant_indices:
                    filtered_results = [filtered_results[idx] for idx in relevant_indices[:max_results]]
                else:
                    # Fallback: keep all high-scoring results
                    filtered_results = [r for r in filtered_results if r.get('judge_score', 0) >= 50][:max_results]
                    
            except Exception as e:
                print(f"LLM filtering failed, keeping all high-scoring results: {e}")
                # Fallback: keep results with reasonable scores
                filtered_results = [r for r in filtered_results if r.get('judge_score', 0) >= 40][:max_results]
        else:
            # If we have reasonable number of results, keep them all
            filtered_results = filtered_results[:max_results]
        
        print(f"Judge filtered: {len(results)} -> {len(filtered_results)} results")
        return filtered_results
        
    except Exception as e:
        print(f"Judge filtering failed: {e}")
        return results[:max_results]

def judge_and_filter_results_with_status(results, question, query_id, max_results=50):
    """Judge and filter results with live status updates"""
    if not results or len(results) == 0:
        return results
    
    update_query_status(query_id, "judge_filtering", f"AI Judge analyzing {len(results)} results")
    
    evaluation_prompt = f"""You are an expert at evaluating search result relevance. Given a user question and a list of search results, score each result's relevance on a scale of 0-100.

Question: "{question}"

Search Results to evaluate:
"""
    
    for i, result in enumerate(results[:20]):
        content_preview = result.get('content', '')[:200]
        title = result.get('title', 'Untitled')
        source = result.get('source', 'Unknown')
        
        evaluation_prompt += f"""

{i+1}. Title: {title}
Source: {source}
Content Preview: {content_preview}..."""
    
    evaluation_prompt += f"""

Please provide ONLY a comma-separated list of scores (0-100) for each result in order. For example: 85,92,15,67,45...

Be strict - only give high scores (80+) to results that directly answer the question. Give medium scores (40-79) to somewhat relevant results. Give low scores (0-39) to irrelevant results."""
    
    try:
        update_query_status(query_id, "judge_filtering", "AI Judge scoring relevance")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Agentic RAG Judge",
            },
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        scores_text = completion.choices[0].message.content.strip()
        
        try:
            scores = [int(float(s.strip())) for s in scores_text.split(',') if s.strip()]
        except:
            import re
            scores = [int(m.group()) for m in re.finditer(r'\d+', scores_text)]
        
        update_query_status(query_id, "judge_filtering", f"AI Judge scored {len(scores)} results")
        
        scored_results = []
        for i, result in enumerate(results[:len(scores)]):
            if i < len(scores):
                result_copy = result.copy()
                result_copy['judge_score'] = scores[i]
                result_copy['original_score'] = result.get('score', 0)
                scored_results.append(result_copy)
        
        scored_results.sort(key=lambda x: x.get('judge_score', 0), reverse=True)
        
        # For now, let's be very inclusive and use LLM filtering smartly
        # Include ALL results first, then let LLM decide what's relevant
        filtered_results = scored_results.copy()
        
        # If we have too many results, use LLM to filter intelligently
        if len(filtered_results) > max_results:
            update_query_status(query_id, "judge_filtering", "AI Judge filtering most relevant results")
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
                
                # Create a concise prompt for LLM filtering
                results_summary = []
                for i, r in enumerate(filtered_results[:30]):  # Evaluate top 30
                    content_preview = r.get('content', '')[:80].replace('\n', ' ')
                    results_summary.append(f"{i+1}. {r.get('title', 'Unknown')} (Score: {r.get('judge_score', 0)}%) - {content_preview}...")
                
                relevance_prompt = f"""Question: "{question}"

Which results are relevant? Return ONLY the numbers (1,2,3...) of relevant results:

{chr(10).join(results_summary)}

Numbers of relevant results:"""
                
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Agentic RAG Filter",
                    },
                    model="meta-llama/llama-3.1-8b-instruct:free",
                    messages=[{"role": "user", "content": relevance_prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                
                response = completion.choices[0].message.content.strip()
                relevant_indices = []
                
                # Parse the response more robustly
                import re
                numbers = re.findall(r'\b(\d+)\b', response)
                for num in numbers:
                    try:
                        idx = int(num) - 1  # Convert to 0-based index
                        if 0 <= idx < len(filtered_results):
                            relevant_indices.append(idx)
                    except:
                        continue
                
                # If LLM selected results, use those; otherwise keep high-scoring ones
                if relevant_indices:
                    filtered_results = [filtered_results[idx] for idx in relevant_indices[:max_results]]
                    update_query_status(query_id, "judge_filtering", f"AI Judge selected {len(filtered_results)} most relevant results")
                else:
                    # Fallback: keep all high-scoring results
                    filtered_results = [r for r in filtered_results if r.get('judge_score', 0) >= 50][:max_results]
                    update_query_status(query_id, "judge_filtering", f"Kept {len(filtered_results)} high-scoring results")
                    
            except Exception as e:
                print(f"LLM filtering failed, keeping all high-scoring results: {e}")
                # Fallback: keep results with reasonable scores
                filtered_results = [r for r in filtered_results if r.get('judge_score', 0) >= 40][:max_results]
                update_query_status(query_id, "judge_filtering", f"Fallback: kept {len(filtered_results)} results")
        else:
            # If we have reasonable number of results, keep them all
            filtered_results = filtered_results[:max_results]
            update_query_status(query_id, "judge_filtering", f"Using all {len(filtered_results)} results")
        
        print(f"Judge filtered: {len(results)} -> {len(filtered_results)} results")
        return filtered_results
        
    except Exception as e:
        print(f"Judge filtering failed: {e}")
        update_query_status(query_id, "judge_filtering", "Judge filtering failed, using original results")
        return results[:max_results]

@app.route('/')
def home():
    return jsonify({'status': 'RAG Backend API is running'})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '').strip()
        chat_history = data.get('chat_history', [])
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Generate unique query ID for status tracking
        query_id = str(uuid.uuid4())
        
        if rag_system is None:
            return jsonify({
                'answer': f"RAG system is still loading... Please wait a moment and try again.\n\nYour question: '{question}' will be answered once the system is ready.",
                'sources': [{
                    'icon': '<svg viewBox="0 0 24 24" width="16" height="16" fill="#ffc107"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>',
                    'title': 'System Loading',
                    'type': 'Please wait...',
                    'score': 0,
                    'view_url': None
                }]
            })

        print(f"Processing query: '{question}'")
        
        # Update status: Starting query processing
        update_query_status(query_id, "query_processing", "Processing your question", {"question": question})

        # Update status: Query expansion
        update_query_status(query_id, "query_expansion", "Expanding query for better search")
        rewritten_queries = rewrite_query(question)
        if len(rewritten_queries) > 1:
            print(f"Query rewritten into {len(rewritten_queries)} variants")
            update_query_status(query_id, "query_expansion", f"Created {len(rewritten_queries)} search variants")

        # Update status: Searching documents
        update_query_status(query_id, "document_search", "Searching through documents and databases")
        all_results = []
        for i, query in enumerate(rewritten_queries):
            update_query_status(query_id, "document_search", f"Searching with variant {i+1}/{len(rewritten_queries)}")
            search_results = rag_system.semantic_search(query, top_k=25, chat_history=chat_history)
            combined_results = search_results.get('combined_results', [])
            all_results.extend(combined_results)
            
        update_query_status(query_id, "document_search", f"Found {len(all_results)} potential matches")

        if not all_results:
            return jsonify({
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': []
            })

        # Update status: Removing duplicates
        update_query_status(query_id, "deduplication", "Removing duplicate results")
        seen_ids = set()
        unique_results = []
        for result in all_results:
            result_id = f"{result.get('title', '')}_{result.get('source', '')}"
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
                
        update_query_status(query_id, "deduplication", f"Filtered to {len(unique_results)} unique results")

        # Update status: Judge filtering
        update_query_status(query_id, "judge_filtering", "AI Judge evaluating result relevance")
        filtered_results = judge_and_filter_results_with_status(unique_results, question, query_id, max_results=30)
        
        # Update status: Generating answer
        update_query_status(query_id, "answer_generation", "Generating comprehensive answer")
        best_results = {'combined_results': filtered_results[:25]}
        
        answer = rag_system.generate_answer(question, best_results, chat_history)
        
        # Update status: Complete
        update_query_status(query_id, "complete", "Answer ready")

        icon_map = {
            'airtable': '<svg viewBox="0 0 24 24" width="16" height="16" fill="#28a745"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
            'document': '<svg viewBox="0 0 24 24" width="16" height="16" fill="#007bff"><path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/></svg>',
            'employee': '<svg viewBox="0 0 24 24" width="16" height="16" fill="#28a745"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
            'policy': '<svg viewBox="0 0 24 24" width="16" height="16" fill="#dc3545"><path d="M12,1L3,5V11C3,16.55 6.84,21.74 12,23C17.16,21.74 21,16.55 21,11V5L12,1M12,7C13.4,7 14.8,8.6 14.8,10V11.5C14.8,12.61 13.91,13.5 12.8,13.5H11.2C10.09,13.5 9.2,12.61 9.2,11.5V10C9.2,8.6 10.6,7 12,7Z"/></svg>',
            'asset': '<svg viewBox="0 0 24 24" width="16" height="16" fill="#fd7e14"><path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M7.07,18.28C7.5,17.38 8.12,16.5 8.91,15.77L10.5,12.65L9.04,10.8C8.64,10.5 8.26,10.15 7.92,9.75H8.91L10.5,11.9L12.09,9.75H13.08C12.74,10.15 12.36,10.5 11.96,10.8L10.5,12.65L12.09,15.77C12.88,16.5 13.5,17.38 13.93,18.28C13.3,18.67 12.57,18.9 11.77,18.96V19H12.23V18.96C13.03,18.9 13.76,18.67 14.39,18.28C17.24,16.09 17.81,12.14 15.62,9.29C13.43,6.44 9.5,5.87 6.65,8.06C3.8,10.25 3.23,14.2 5.42,17.05C5.97,17.7 6.64,18.23 7.39,18.62C7.32,18.5 7.18,18.4 7.07,18.28Z"/></svg>'
        }

        sources = []
        for result in filtered_results[:50]:
            source_type = 'employee' if 'employee' in result.get('type', '').lower() else 'document'
            if 'airtable' in result.get('source', '').lower():
                source_type = 'airtable'
            
            icon = icon_map.get(source_type, icon_map['document'])
            
            # For Airtable results, use original score instead of judge score
            if 'airtable' in result.get('source', '').lower():
                source_score = result.get('original_score', result.get('score', 0.5))
                if isinstance(source_score, float):
                    source_score = int(source_score * 100)
            else:
                source_score = result.get('judge_score')
                if source_score is None:
                    source_score = int(result.get('score', 0.5) * 100)
            
            sources.append({
                'title': result.get('title', 'Untitled'),
                'type': result.get('type', 'Document'),
                'icon': icon,
                'score': source_score,
                'view_url': result.get('view_url')
            })

        return jsonify({
            'answer': answer,
            'sources': sources,
            'query_id': query_id
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'ready': rag_system is not None,
        'rag_system_loaded': rag_system is not None
    })

@app.route('/api/status/<query_id>')
def get_status(query_id):
    """Get the current status of a query"""
    status = get_query_status(query_id)
    if status:
        return jsonify(status)
    else:
        return jsonify({'error': 'Query not found'}), 404

@app.route('/api/status/latest')
def get_latest_status():
    """Get the most recent query status"""
    if not query_status:
        return jsonify({'message': 'Processing your question...', 'step': 'waiting'}), 404
    
    # Get the most recent status entry
    latest_query_id = max(query_status.keys(), key=lambda x: query_status[x]['timestamp'])
    latest_status = query_status[latest_query_id]
    
    return jsonify(latest_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)