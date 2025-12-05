import pandas as pd  
import json  
  
def refactor_reft_results(csv_path, output_format="table"):  
    # Refactor ReFT evaluation results into key-value format.  
 
    df = pd.read_csv(csv_path)  
    results = []  
    for idx, row in df.iterrows():  
        result = {  
            "sample_id": idx + 1,  
            "dialogue": row["dialogue"],  
            "true_label": row["true_label"],  
            "reft_response": row["reft_response"],  
            "baseline_response": row["baseline_response"],  
            "metrics": {  
                "cosine_similarity": row["cosine_similarity"],  
                "bert_precision": row["bert_precision"],  
                "bert_recall": row["bert_recall"],  
                "bert_f1": row["bert_f1"],  
                "reft_mean_score": row["reft_mean_score"],  
                "reft_individual_scores": row["reft_individual_scores"],  
                "reft_agreement": row["reft_agreement"]  
            }  
        }  
        results.append(result)  
        
    with open("./reft_results_key_value.json", "w") as f:  
        json.dump(results, f, indent=2)  
        
    print("Saved key-value format to: ./reft_results_key_value.json")  
 
if __name__ == "__main__": 
    refactor_reft_results("./reft_evaluation_results.csv", "key_value")  
    