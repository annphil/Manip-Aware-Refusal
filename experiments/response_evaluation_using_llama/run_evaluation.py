import csv  
import pandas as pd  
import argparse
from tqdm import tqdm  
from control_token_generator import ControlTokenGenerator  
from similarity_evaluator import SimilarityEvaluator  
  
def load_dataset(file_path):  
    """Load MentalManip dataset."""  
    dialogues = []  
    ground_truth_labels = []  # Keep for comparison  
      
    with open(file_path, 'r', encoding='utf-8') as f:  
        reader = csv.reader(f)  
        next(reader)  # Skip header  
        for row in reader:  
            dialogues.append(row[1])  # Dialogue  
            ground_truth_labels.append(int(row[2]))  # Ground truth (for comparison only)  
      
    return dialogues, ground_truth_labels  
  
def run_evaluation(dataset_path, output_path, detection_model, generation_model, max_samples=None):  
    """  
    Run evaluation with manipulation detection and conditional response generation.  
      
    Args:  
        dataset_path: Path to MentalManip CSV file  
        output_path: Path to save results CSV  
        max_samples: Maximum number of samples to evaluate (None for all)  
    """  
    # Load dataset  
    print(f"Loading dataset from: {dataset_path}")  
    dialogues, ground_truth_labels = load_dataset(dataset_path)  
      
    if max_samples:  
        dialogues = dialogues[:max_samples]  
        ground_truth_labels = ground_truth_labels[:max_samples]  
      
    print(f"Loaded {len(dialogues)} dialogues")  
      
    # Initialize models  
    print("Initializing models...")  
    generator = ControlTokenGenerator()  
    evaluator = SimilarityEvaluator()  
      
    # Results storage  
    results = []  
      
    print("Running evaluation...")  
    for idx, dialogue in enumerate(tqdm(dialogues)):  
        result = {  
            'dialogue': dialogue,  
            'ground_truth_label': ground_truth_labels[idx],  
            'predicted_label': None,  
            'baseline_response': None,  
            'control_response': None,  
            'cosine_similarity': None,  
            'bert_precision': None,  
            'bert_recall': None,  
            'bert_f1': None  
        }  
          
        # Step 1: Detect manipulation  
        predicted_label = generator.detect_manipulation(dialogue)  
        result['predicted_label'] = predicted_label  
          
        # Skip if detection failed  
        if predicted_label == -1:  
            print(f"Warning: Detection failed for sample {idx}")  
            results.append(result)  
            continue  
          
        # Step 2: Generate responses only if manipulation detected  
        if predicted_label == 1:  
            # Generate baseline response (no manipulation awareness)  
            baseline_response = generator.generate_baseline_response(dialogue)  
            result['baseline_response'] = baseline_response  
              
            # Generate control response (with manipulation awareness)  
            control_token = generator.create_control_token(predicted_label)  
            control_response = generator.generate_response(dialogue, control_token=control_token)
            #control_response = generator.generate_control_response(dialogue, predicted_label)  
            result['control_response'] = control_response  
              
            # Step 3: Compute similarity metrics  
            cosine_sim = evaluator.compute_cosine_similarity(baseline_response, control_response)  
            result['cosine_similarity'] = cosine_sim  
              
            bert_scores = evaluator.compute_bert_score(baseline_response, control_response)  
            result['bert_precision'] = bert_scores['precision']  
            result['bert_recall'] = bert_scores['recall']  
            result['bert_f1'] = bert_scores['f1']  
        else:  
            # No manipulation detected - generate response but don't compute metrics  
            baseline_response = generator.generate_baseline_response(dialogue)  
            result['baseline_response'] = baseline_response  
            result['control_response'] = "N/A (no manipulation detected)"  
          
        results.append(result)  
      
    # Save results  
    print(f"Saving results to: {output_path}")  
    df = pd.DataFrame(results)  
    df.to_csv(output_path, index=False)  
      
    # Compute summary statistics (only for manipulation-detected cases)  
    manip_detected_results = [r for r in results if r['predicted_label'] == 1 and r['cosine_similarity'] is not None]  
      
    if manip_detected_results:  
        avg_cosine = sum(r['cosine_similarity'] for r in manip_detected_results) / len(manip_detected_results)  
        avg_bert_f1 = sum(r['bert_f1'] for r in manip_detected_results) / len(manip_detected_results)  
          
        print("\n=== Summary Statistics (Manipulation Detected Cases Only) ===")  
        print(f"Total samples: {len(results)}")  
        print(f"Manipulation detected: {sum(1 for r in results if r['predicted_label'] == 1)}")  
        print(f"No manipulation detected: {sum(1 for r in results if r['predicted_label'] == 0)}")  
        print(f"Detection failed: {sum(1 for r in results if r['predicted_label'] == -1)}")  
        print(f"Samples with similarity metrics: {len(manip_detected_results)}")  
        print(f"Average cosine similarity: {avg_cosine:.4f}")  
        print(f"Average BERT F1: {avg_bert_f1:.4f}")  
          
        # Detection accuracy (compared to ground truth)  
        correct_detections = sum(1 for r in results if r['predicted_label'] == r['ground_truth_label'])  
        detection_accuracy = correct_detections / len(results)  
        print(f"Detection accuracy: {detection_accuracy:.4f}")  
    else:  
        print("\nNo manipulation detected in any samples - no similarity metrics computed")  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Response evaluation with manipulation detection")  
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")  
    parser.add_argument("--output", required=True, help="Path to save results CSV")  
    parser.add_argument("--detection_model", default="llama_ft/mentalmanip", help="Path to fine-tuned detection model")  
    parser.add_argument("--generation_model", default="meta-llama/Llama-2-13b-chat-hf", help="Base model for generation")  
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")  
      
    args = parser.parse_args()  
      
    run_evaluation(  
        dataset_path=args.dataset,  
        output_path=args.output,  
        detection_model=args.detection_model,  
        generation_model=args.generation_model,  
        max_samples=args.max_samples  
    )

