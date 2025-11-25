import csv  
import pandas as pd  
import argparse
from tqdm import tqdm  
from control_token_generator import ControlTokenGenerator  
from similarity_evaluator import SimilarityEvaluator  

import sys  
import os  
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
from manipulation_detection.load_data import LoadManipDataset
from llm_judge import MultiLLMJudge

def load_dataset(file_path):  
        """Load MentalManip dataset and return ONLY test split."""  
        manip_dataset = LoadManipDataset(  
            file_name=file_path,  
            train_ratio=0.6,  
            valid_ratio=0.2,  
            test_ratio=0.2  
        )  
 
        test_data = manip_dataset.df_test  
        
        dialogues = test_data['Dialogue'].tolist()  
        ground_truth_labels = [int(v) for v in test_data['Manipulative'].values]  
        
        return dialogues, ground_truth_labels

def run_evaluation(dataset_path, output_path, detection_model, generation_model, max_samples=None):    
    """    
    Run evaluation with manipulation detection and conditional response generation.    
    Writes results incrementally after each sample.  
        
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
    llm_judge = MultiLLMJudge()  
      
    # Initialize CSV file with headers  
    with open(output_path, 'w', newline='', encoding='utf-8') as f:  
        writer = csv.DictWriter(f, fieldnames=[  
            'dialogue', 'ground_truth_label', 'predicted_label',  
            'baseline_response', 'control_response', 'cosine_similarity',  
            'bert_precision', 'bert_recall', 'bert_f1',  
            'baseline_refusal_score', 'control_refusal_score',   
            'baseline_reasoning', 'control_reasoning'  
        ])  
        writer.writeheader()  
      
    # Tracking variables for summary statistics  
    total_samples = len(dialogues)  
    manipulation_detected_count = 0  
    no_manipulation_count = 0  
    detection_failed_count = 0  
    cosine_sims = []  
    bert_f1s = []  
    correct_predictions = 0  
    baseline_refusal_scores = []  
    control_refusal_scores = []
        
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
            'bert_f1': None,
            'baseline_refusal_score': None,  
            'control_refusal_score': None,  
            'baseline_reasoning': None,  
            'control_reasoning': None     
        }    
            
        # Step 1: Detect manipulation    
        predicted_label = generator.detect_manipulation(dialogue)    
        result['predicted_label'] = predicted_label    
            
        # Skip if detection failed    
        if predicted_label == -1:    
            print(f"Warning: Detection failed for sample {idx}")  
            detection_failed_count += 1  
        elif predicted_label == ground_truth_labels[idx]:  
            correct_predictions += 1  
            
        # Step 2: Generate responses only if manipulation detected    
        if predicted_label == 1:    
            manipulation_detected_count += 1  
              
            # Generate baseline response (no manipulation awareness)    
            baseline_response = generator.generate_baseline_response(dialogue)    
            result['baseline_response'] = baseline_response    
                
            # Generate control response (with manipulation awareness)    
            control_token = generator.create_control_token(predicted_label)    
            control_response = generator.generate_response(dialogue, control_token=control_token)  
            result['control_response'] = control_response    
                
            # Step 3: Compute similarity metrics    
            cosine_sim = evaluator.compute_cosine_similarity(baseline_response, control_response)    
            result['cosine_similarity'] = cosine_sim    
            cosine_sims.append(cosine_sim)  
                
            bert_scores = evaluator.compute_bert_score(baseline_response, control_response)    
            result['bert_precision'] = bert_scores['precision']    
            result['bert_recall'] = bert_scores['recall']    
            result['bert_f1'] = bert_scores['f1']  
            bert_f1s.append(bert_scores['f1'])  

            # LLM-as-judge evaluation  
            baseline_judgment = llm_judge.evaluate_refusal_multi(dialogue, baseline_response, predicted_label)  
            baseline_refusal_scores.append(baseline_judgment['refusal_score']) 
            result['baseline_refusal_score'] = baseline_judgment['mean_score']  
            result['baseline_individual_scores'] = baseline_judgment['individual_scores']  
            result['baseline_agreement'] = baseline_judgment['agreement']  
            
            # Evaluate control response  
            control_judgment = llm_judge.evaluate_refusal_multi(dialogue, control_response, predicted_label)  
            control_refusal_scores.append(control_judgment['refusal_score'])
            result['control_refusal_score'] = control_judgment['mean_score']  
            result['control_individual_scores'] = control_judgment['individual_scores']  
            result['control_agreement'] = control_judgment['agreement'] 

        else:    
            no_manipulation_count += 1  
              
            # No manipulation detected - generate response but don't compute metrics    
            baseline_response = generator.generate_baseline_response(dialogue)    
            result['baseline_response'] = baseline_response    
            result['control_response'] = "N/A (no manipulation detected)"    
          
        # Write result immediately to CSV  
        with open(output_path, 'a', newline='', encoding='utf-8') as f:  
            writer = csv.DictWriter(f, fieldnames=[  
                'dialogue', 'ground_truth_label', 'predicted_label',  
                'baseline_response', 'control_response', 'cosine_similarity',  
                'bert_precision', 'bert_recall', 'bert_f1' ,
                'baseline_refusal_score', 'control_refusal_score',   
                'baseline_reasoning', 'control_reasoning'
            ])  
            writer.writerow(result)  
            f.flush()
        
    # Print summary statistics  
    print(f"\nResults saved to: {output_path}")  
      
    if cosine_sims:    
        avg_cosine = sum(cosine_sims) / len(cosine_sims)    
        avg_bert_f1 = sum(bert_f1s) / len(bert_f1s)    
        detection_accuracy = correct_predictions / total_samples    
            
        print("\n=== Summary Statistics (Manipulation Detected Cases Only) ===")    
        print(f"Total samples: {total_samples}")    
        print(f"Manipulation detected: {manipulation_detected_count}")    
        print(f"No manipulation detected: {no_manipulation_count}")    
        print(f"Detection failed: {detection_failed_count}")    
        print(f"Samples with similarity metrics: {len(cosine_sims)}")    
        print(f"Average cosine similarity: {avg_cosine:.4f}")    
        print(f"Average BERT F1: {avg_bert_f1:.4f}")    
        print(f"Detection accuracy: {detection_accuracy:.4f}") 

        # LLM judge statistics  
        if baseline_refusal_scores:  
            avg_baseline_refusal = sum(baseline_refusal_scores) / len(baseline_refusal_scores)  
            avg_control_refusal = sum(control_refusal_scores) / len(control_refusal_scores)  
            refusal_improvement = avg_control_refusal - avg_baseline_refusal  
            
            print("\n=== LLM Judge Refusal Scores (Manipulation Detected Cases) ===")  
            print(f"Average baseline refusal score: {avg_baseline_refusal:.2f}/5.0")  
            print(f"Average control refusal score: {avg_control_refusal:.2f}/5.0")  
            print(f"Refusal improvement (control - baseline): {refusal_improvement:+.2f}")  
            print(f"Samples evaluated by LLM judge: {len(baseline_refusal_scores)}")   
            
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

