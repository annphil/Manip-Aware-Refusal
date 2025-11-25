import torch  
from sentence_transformers import SentenceTransformer  
from bert_score import score as bert_score_fn  
import numpy as np  
  
class SimilarityEvaluator:  
    """Evaluates similarity between baseline and control responses."""  
      
    def __init__(self):  
        self.sentence_model = None  
        self._bert_score_loaded = False  
      
    def compute_cosine_similarity(self, text1, text2):  
        """  
        Compute cosine similarity using sentence transformers.  
          
        Args:  
            text1: First text  
            text2: Second text  
              
        Returns:  
            float: Cosine similarity score  
        """  
        if self.sentence_model is None:  
            print("Loading sentence transformer model (first use only)...")  
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  
          
        embeddings = self.sentence_model.encode([text1, text2])  
          
        # Compute cosine similarity  
        similarity = np.dot(embeddings[0], embeddings[1]) / (  
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])  
        )  
          
        return float(similarity)  
      
    def compute_bert_score(self, text1, text2):  
        """  
        Compute BERT score between two texts.  
          
        Args:  
            text1: Reference text  
            text2: Candidate text  
              
        Returns:  
            Dict with precision, recall, f1  
        """  
        if not self._bert_score_loaded:  
            print("Loading BERT score model (first use only)...")  
            self._bert_score_loaded = True  
          
        P, R, F1 = bert_score_fn([text2], [text1], lang="en", verbose=False)  
          
        return {  
            "precision": float(P[0]),  
            "recall": float(R[0]),  
            "f1": float(F1[0])  
        }
