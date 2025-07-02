# TechFlow Assistant

A multi-modal LLM system assistant that helps employees find information about colleagues, policies, assets, and tickets using advanced RAG (Retrieval-Augmented Generation) technology. The system combines document search with live Airtable data through SQL agents to provide accurate, real-time answers.

## Setup

1. **Configure environment variables:**
   Create a `.env` file with:
   ```
   BASE_ID=your_airtable_base_id
   AIRTABLE_API_KEY=your_airtable_api_key
   PINECONE_API_KEY=your_pinecone_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
   OPENROUTER_API_KEY=your_openrouter_key
   CLAUDE_API_KEY=your_claude_api_key
   ```

2. **Run the application:**
   ```bash
   ./start_all.sh
   ```

3. **Access the application:**
   Open http://localhost:3000 in your browser

## Requirements

- Python 3.8+
- Valid API keys for Anthropic Claude, OpenRouter, Pinecone, Supabase, and Airtable