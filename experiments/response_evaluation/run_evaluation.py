import csv  
import pandas as pd  
from tqdm import tqdm  
from control_token_generator import ControlTokenGenerator  
from similarity_evaluator import SimilarityEvaluator  
import argparse  
  
def load_dataset(file_path):  
    """Load MentalManip dataset."""  
    dialogues = []  
    ground_truth_labels = []  
      
    with open(file_path, 'r', encoding='utf-8') as f:  
        reader = csv.reader(f)  
        next(reader)  # Skip header  
        for row in reader:  
            dialogues.append(row[1])  # Dialogue  
            ground_truth_labels.append(int(row[2]))  # Ground truth (for comparison)  
      
    return dialogues, ground_truth_labels  
  
def run_evaluation(dataset_path, output_path, detection_model="roberta_ft/mentalmanip",  
                   generation_model="meta-llama/Llama-2-13b-chat-hf", max_samples=None):  
    """Run response evaluation with RoBERTa detection and Llama generation."""  
      
    # Load dataset  
    print(f"Loading dataset from: {dataset_path}")  
    dialogues, ground_truth_labels = load_dataset(dataset_path)  
      
    if max_samples:  
        dialogues = dialogues[:max_samples]  
        ground_truth_labels = ground_truth_labels[:max_samples]  
      
    print(f"Evaluating {len(dialogues)} samples...")  
      
    # Initialize models  
    generator = ControlTokenGenerator(  
        detection_model_path=detection_model,  
        generation_model_name=generation_model  
    )  
    evaluator = SimilarityEvaluator()  
      
    # Results storage  
    results = []  
    total_samples = len(dialogues)  
    manipulation_detected_count = 0  
    no_manipulation_count = 0  
      
    print("Running evaluation...")  
    for idx, dialogue in enumerate(tqdm(dialogues)):  
        ground_truth = ground_truth_labels[idx]  
          
        # Step 1: Detect manipulation using RoBERTa  
        predicted_label = generator.detect_manipulation(dialogue)  
          
        if predicted_label == 1:  
            manipulation_detected_count += 1  
              
            # Step 2: Generate baseline response (no control token)  
            baseline_response = generator.generate_baseline_response(dialogue)  
              
            # Step 3: Generate control response (with manipulation token)  
            control_token = generator.create_control_token(predicted_label)  
            control_response = generator.generate_control_response(dialogue, control_token)  
              
            # Step 4: Compute similarity metrics  
            cosine_sim = evaluator.compute_cosine_similarity(baseline_response, control_response)  
            bert_scores = evaluator.compute_bert_score(baseline_response, control_response)  
              
            results.append({  
                'dialogue': dialogue,  
                'ground_truth_label': ground_truth,  
                'predicted_label': predicted_label,  
                'baseline_response': baseline_response,  
                'control_response': control_response,  
                'cosine_similarity': cosine_sim,  
                'bert_precision': bert_scores['precision'],  
                'bert_recall': bert_scores['recall'],  
                'bert_f1': bert_scores['f1']  
            })  
        else:  
            no_manipulation_count += 1  
              
            # Generate baseline response only (no similarity metrics)  
            baseline_response = generator.generate_baseline_response(dialogue)  
              
            results.append({  
                'dialogue': dialogue,  
                'ground_truth_label': ground_truth,  
                'predicted_label': predicted_label,  
                'baseline_response': baseline_response,  
                'control_response': 'N/A (no manipulation detected)',  
                'cosine_similarity': '',  
                'bert_precision': '',  
                'bert_recall': '',  
                'bert_f1': ''  
            })  
      
    # Save results  
    print(f"Saving results to: {output_path}")  
    df = pd.DataFrame(results)  
    df.to_csv(output_path, index=False)  
      
    # Calculate statistics (only for manipulation detected cases)  
    manipulation_results = [r for r in results if r['predicted_label'] == 1]  
      
    if manipulation_results:  
        avg_cosine = sum(r['cosine_similarity'] for r in manipulation_results) / len(manipulation_results)  
        avg_bert_f1 = sum(r['bert_f1'] for r in manipulation_results) / len(manipulation_results)  
          
        # Calculate detection accuracy  
        correct_predictions = sum(1 for r in results if r['predicted_label'] == r['ground_truth_label'])  
        detection_accuracy = correct_predictions / total_samples  
          
        print("\n=== Summary Statistics (Manipulation Detected Cases Only) ===")  
        print(f"Total samples: {total_samples}")  
        print(f"Manipulation detected: {manipulation_detected_count}")  
        print(f"No manipulation detected: {no_manipulation_count}")  
        print(f"Samples with similarity metrics: {len(manipulation_results)}")  
        print(f"Average cosine similarity: {avg_cosine:.4f}")  
        print(f"Average BERT F1: {avg_bert_f1:.4f}")  
        print(f"Detection accuracy: {detection_accuracy:.4f}")  
    else:  
        print("\n=== No manipulation detected in any samples ===")  
      
    print("Evaluation complete!")  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Response evaluation with RoBERTa detection and Llama generation")  
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")  
    parser.add_argument("--output", required=True, help="Path to save results CSV")  
    parser.add_argument("--detection_model", default="roberta_ft/mentalmanip", help="Path to fine-tuned RoBERTa model")  
    parser.add_argument("--generation_model", default="meta-llama/Llama-2-13b-chat-hf", help="Base Llama model for generation")  
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")  
      
    args = parser.parse_args()  
      
    run_evaluation(  
        dataset_path=args.dataset,  
        output_path=args.output,  
        detection_model=args.detection_model,  
        generation_model=args.generation_model,  
        max_samples=args.max_samples  
    )