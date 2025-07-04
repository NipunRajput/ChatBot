import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app)

# Configuration
GROQ_API_KEY =os.getenv("GROQ_API")
GROQ_MODEL = os.getenv("MODEL")
GROQ_API_URL = os.getenv("API_URL")
EMBEDDING_MODEL =os.getenv("MODEL_UPI")

# Load and preprocess knowledge base
def load_knowledge_base():
    try:
        with open("knowledge.txt", "r", encoding="utf-8") as f:
            content = f.read().strip()
            
            # Parse the Global Garner knowledge base
            products = {}
            lines = content.split('\n')
            current_product = None
            current_section = None
            in_features = False
            in_timeline = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect product headers (numbered items)
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    current_product = line.split('.', 1)[1].strip()
                    products[current_product] = {
                        'title': current_product,
                        'description': '',
                        'features': [],
                        'usage_timeline': [],
                        'details': [],
                        'category': '',
                        'target_users': ''
                    }
                    in_features = False
                    in_timeline = False
                    current_section = None
                
                # Handle main sections
                elif line.startswith('Description:'):
                    current_section = 'description'
                    desc_text = line.replace('Description:', '').strip()
                    if desc_text and current_product:
                        products[current_product]['description'] = desc_text
                        
                elif line.startswith('Key Features:'):
                    in_features = True
                    current_section = 'features'
                    
                elif line.startswith('Usage Timeline:'):
                    in_timeline = True
                    current_section = 'timeline'
                    
                elif line.startswith('Category:'):
                    if current_product:
                        products[current_product]['category'] = line.replace('Category:', '').strip()
                        
                elif line.startswith('Target Users:'):
                    if current_product:
                        products[current_product]['target_users'] = line.replace('Target Users:', '').strip()
                
                # Handle features and descriptions
                elif current_product and current_section == 'description' and not line.startswith(('Key Features:', 'Usage Timeline:', 'Category:', 'Target Users:')):
                    if products[current_product]['description']:
                        products[current_product]['description'] += ' ' + line
                    else:
                        products[current_product]['description'] = line
                        
                elif current_product and in_features and line and not line.startswith(('Usage Timeline:', 'Category:', 'Target Users:')):
                    if ':' in line:
                        products[current_product]['features'].append(line)
                    elif line.startswith(('•', '-', '*')):
                        products[current_product]['features'].append(line)
                        
                elif current_product and in_timeline and line and not line.startswith(('🏢', '🧩', 'Category:', 'Target Users:')):
                    if '|' in line or line.startswith(('Stage', 'Activity', 'Initial', 'Daily', 'Continuous', 'Onboarding', 'Ongoing', 'Post-')):
                        products[current_product]['usage_timeline'].append(line)
                
                # Handle general details
                elif current_product and not in_features and not in_timeline and current_section != 'description':
                    if line not in ['Key Features:', 'Usage Timeline:', 'Description:'] and not line.startswith(('🏢', '🧩', 'Category:', 'Target Users:')):
                        products[current_product]['details'].append(line)
            
            return products
                
    except FileNotFoundError:
        print("Error: knowledge.txt file not found!")
        return {}
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return {}

# Load knowledge base
products = load_knowledge_base()

# Prepare chunks for vector search
knowledge_chunks = []
chunk_metadata = []  # Store metadata for each chunk

if products:
    for product_id, product_data in products.items():
        chunks = []
        
        # Create comprehensive chunks for Global Garner products
        if isinstance(product_data, dict):
            # Main product info
            chunks.append(f"Product: {product_data.get('title', product_id)}")
            
            # Description chunk
            if product_data.get('description'):
                chunks.append(f"Description: {product_data['description']}")
            
            # Category and target users
            if product_data.get('category'):
                chunks.append(f"Category: {product_data['category']}")
            if product_data.get('target_users'):
                chunks.append(f"Target Users: {product_data['target_users']}")
            
            # Features - each feature as separate chunk for better retrieval
            for feature in product_data.get('features', []):
                if feature.strip():
                    chunks.append(f"{product_id} Feature: {feature}")
            
            # Usage timeline
            if product_data.get('usage_timeline'):
                timeline_text = ' '.join(product_data['usage_timeline'])
                chunks.append(f"{product_id} Usage Timeline: {timeline_text}")
            
            # Additional details
            if product_data.get('details'):
                for detail in product_data['details']:
                    if detail.strip():
                        chunks.append(f"{product_id} Detail: {detail}")
        else:
            # Handle string data
            chunks = [f"{product_id}: {str(product_data)}"]
        
        # Add chunks to knowledge base
        for chunk in chunks:
            if chunk.strip() and not chunk.endswith(": ") and len(chunk) > 10:  # Skip empty/short chunks
                knowledge_chunks.append(chunk)
                chunk_metadata.append(product_id)

# Initialize embedding model if chunks exist
if knowledge_chunks:
    print(f"Loaded {len(knowledge_chunks)} knowledge chunks")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    chunk_embeddings = embedder.encode(knowledge_chunks)
else:
    print("Warning: No knowledge chunks loaded!")
    embedder = None
    chunk_embeddings = None

def retrieve_context(query, top_k=5):
    """Retrieve most relevant context using vector similarity"""
    if not knowledge_chunks or embedder is None:
        return "No knowledge base available."
    
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get top matches with similarity scores
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Filter by minimum similarity threshold
    min_similarity = 0.1  # Adjust this threshold as needed
    relevant_chunks = []
    
    for idx in top_indices:
        if similarities[idx] >= min_similarity:
            relevant_chunks.append(knowledge_chunks[idx])
    
    if not relevant_chunks:
        return "No relevant information found in knowledge base."
    
    return "\n".join(relevant_chunks)

def ask_groq(query, context):
    """Query Groq API with strict knowledge base constraints"""
    prompt = (
        "You are a confident sales assistant for Global Garner Sales Services Limited and Global Garner Consultancy Services. "
        "Provide clear, direct answers without using uncertain language like 'seems', 'appears', 'might', or 'possibly'. "
        "Be definitive and professional in your responses.\n\n"
        
        "CRITICAL: You must ONLY use the provided context to answer questions. "
        "If the context doesn't contain information to answer the question, "
        "respond with 'I don't have that specific information in my knowledge base.' "
        "Never make up, assume, or infer information not explicitly stated in the context.\n\n"
        
        "COMPANY STRUCTURE:\n"
        "Global Garner Sales Services Limited (Parent Company - Founded 2016 by Vikas Rawat):\n"
        "- Universal Cashback App (Consumer mobile app with up to 100% real cashback)\n"
        "- Post-Paid Sales Service (Commission-only business model for vendors)\n"
        "- GG Vendor App & Dashboard (SaaS tool for vendor management)\n"
        "- GG Mall (Online marketplace)\n"
        "- UPOS Digital Mall (Franchise-as-a-Service platform)\n\n"
        
        "Global Garner Consultancy Services (Subsidiary - Launched 2025):\n"
        "- GG LocalPulse (Content monetization platform for creators)\n"
        "- GG EduCore360 (School management software)\n"
        "- GG PrepMaster (Test preparation platform)\n"
        "- GG EduSphere (University LMS platform)\n"
        "- GG PharmaSphere360 (Pharmacy inventory management)\n"
        "- GG VendoStream (Video e-commerce & vendor operations)\n"
        "- FranchiseBuilder360 (Franchise management system)\n\n"
        
        f"CONTEXT FROM KNOWLEDGE BASE:\n{context}\n\n"
        f"CUSTOMER QUESTION: {query}\n\n"
        
        "INSTRUCTIONS:\n"
        "1. Answer directly using ONLY the context provided above\n"
        "2. Be specific about product features, pricing, and benefits\n"
        "3. If discussing costs, mention the commission-only/no-upfront-cost model where applicable\n"
        "4. For SaaS products, clarify which company (Sales Services vs Consultancy Services) offers them\n"
        "5. Always end your response with: 'For detailed information and personalized assistance, contact us at +91 6354917511 or email ggcs@globalgarner.com'\n\n"
        
        "Provide a helpful, confident response , don't write this for any response= Based on the context provided,:"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Very low for strict factual responses
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with AI service: {str(e)}"
    except KeyError:
        return "Error: Unexpected response format from AI service"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if not knowledge_chunks:
        return jsonify({"error": "Knowledge base not loaded"}), 500

    try:
        context = retrieve_context(query)
        answer = ask_groq(query, context)
        
        return jsonify({
            "response": answer,
            "context_used": context.split("\n"),  # For debugging
            "knowledge_chunks_available": len(knowledge_chunks)
        })
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "knowledge_chunks_loaded": len(knowledge_chunks) if knowledge_chunks else 0,
        "embedder_loaded": embedder is not None
    })

if __name__ == "__main__":
    if not knowledge_chunks:
        print("Warning: Starting server without knowledge base!")
    app.run(host="0.0.0.0", port=5000, debug=True)