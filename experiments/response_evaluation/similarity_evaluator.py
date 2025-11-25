import torch  
from sentence_transformers import SentenceTransformer, util  
from bert_score import score as bert_score_fn  
  
class SimilarityEvaluator:  
    """Evaluates similarity between baseline and control responses."""  
      
    def __init__(self, device="cuda"):  
        self.device = device  
        print("Loading sentence transformer model...")  
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)  
        self._bert_score_loaded = False  
      
    def compute_cosine_similarity(self, text1, text2):  
        """Compute cosine similarity using sentence transformers."""  
        embeddings = self.sentence_model.encode([text1, text2], convert_to_tensor=True)  
        similarity = util.cos_sim(embeddings[0], embeddings[1])  
        return float(similarity)  
      
    def compute_bert_score(self, text1, text2):  
        """Compute BERT score between two texts."""  
        if not self._bert_score_loaded:  
            print("Loading BERT score model (first use only)...")  
            self._bert_score_loaded = True  
          
        P, R, F1 = bert_score_fn([text2], [text1], lang="en", verbose=False)  
          
        return {  
            "precision": float(P[0]),  
            "recall": float(R[0]),  
            "f1": float(F1[0])  
        }