# TechFlow Assistant

A multi-modal LLM system assistant that helps employees find information about colleagues, policies, assets, and tickets using advanced RAG (Retrieval-Augmented Generation) technology. The system combines document search with live Airtable data through SQL agents to provide accurate, real-time answers.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   Create a `.env` file with:
   ```
   ANTHROPIC_API_KEY=your_anthropic_key
   OPENROUTER_API_KEY=your_openrouter_key
   PINECONE_API_KEY=your_pinecone_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   AIRTABLE_BASE_ID=your_airtable_base_id
   AIRTABLE_API_KEY=your_airtable_api_key
   ```

3. **Run the application:**
   ```bash
   # Start backend
   python backend/app.py
   
   # Start frontend (in new terminal)
   cd frontend && python -m http.server 3000
   ```

4. **Access the application:**
   Open http://localhost:3000 in your browser

## Requirements

- Python 3.8+
- Valid API keys for Anthropic Claude, OpenRouter, Pinecone, Supabase, and Airtable