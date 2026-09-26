import numpy as np
from sentence_transformers import SentenceTransformer, util

# Sample SEC 10-K / Earnings Call Transcript Chunks
EARNINGS_TRANSCRIPT_CHUNKS = [
    "Apple announced quarterly revenue of $85.8 billion, up 5% year-over-year, driven by record Services revenue.",
    "Management highlighted that iPhone revenue reached $39.3 billion despite foreign exchange headwinds.",
    "Our gross margin for the quarter was 46.3%, benefiting from favorable product mix and cost efficiencies.",
    "Key supply chain risks include geopolitical tension and component availability shortages in Asia.",
    "Capital expenditure for AI infrastructure and data centers is projected to increase by 20% in the upcoming fiscal year.",
    "Services segment growth was led by strong subscription adoption across Apple Music, iCloud, and Apple Pay."
]

class FinancialRAGAnalyzer:
    def __init__(self):
        print("--> Loading Sentence-Transformer Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("--> Indexing financial transcript chunks into vector space...")
        self.doc_embeddings = self.model.encode(EARNINGS_TRANSCRIPT_CHUNKS, convert_to_tensor=True)

    def query_transcript(self, user_query, top_k=2):
        print(f"\n--> Querying Vector Database for: '{user_query}'")
        query_embedding = self.model.encode(user_query, convert_to_tensor=True)
        
        # Compute Cosine Similarity between query and transcript embeddings
        cosine_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        top_results = np.argsort(cosine_scores.cpu().numpy())[::-1][:top_k]

        print("Top Relevant Context Chunks Retrieved:")
        retrieved_chunks = []
        for rank, idx in enumerate(top_results):
            score = cosine_scores[idx].item()
            chunk_text = EARNINGS_TRANSCRIPT_CHUNKS[idx]
            print(f" [{rank+1}] Match Score: {score:.4f} | '{chunk_text}'")
            retrieved_chunks.append(chunk_text)
            
        return retrieved_chunks

if __name__ == "__main__":
    rag_engine = FinancialRAGAnalyzer()
    
    # Test Query 1: Supply Chain Risks
    rag_engine.query_transcript("What are the main risks facing the company's supply chain?")
    
    # Test Query 2: Revenue & Financial Performance
    rag_engine.query_transcript("How did the Services segment and gross margin perform?")
