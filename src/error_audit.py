import json
import csv

def generate_error_audit_csv(predictions_path, truth_path, output_csv):
    """
    Compares model predictions against ground truth and exports misses to a CSV.
    Assumes JSONL format, which is standard for most LLM/ML hackathon evaluators.
    """
    # Load ground truth into a dictionary keyed by ID for safe matching
    ground_truth = {}
    with open(truth_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Adjust the key 'id' to match whatever the organizers use
            ground_truth[data['id']] = data

    errors = []
    
    with open(predictions_path, 'r') as f:
        for line in f:
            pred_data = json.loads(line)
            q_id = pred_data.get('id')
            
            if q_id not in ground_truth:
                continue
                
            truth_data = ground_truth[q_id]
            expected = str(truth_data.get('answer', '')).strip()
            predicted = str(pred_data.get('prediction', '')).strip()
            raw_output = str(pred_data.get('raw_generation', ''))
            
            # Simple string comparison - swap this for the official evaluator's logic once it drops
            if expected.lower() != predicted.lower():
                errors.append({
                    'ID': q_id,
                    'Question': truth_data.get('question', 'Question text missing'),
                    'Expected_Answer': expected,
                    'Extracted_Prediction': predicted,
                    'Raw_Model_Output': raw_output,
                    'Error_Category': '',  # Blank column for Teammate 2 to fill out
                    'Notes': ''            # Blank column for quick observations
                })

    if not errors:
        print("No errors found! (Check your parsing logic to ensure it isn't broken)")
        return

    # Dump to CSV for easy spreadsheet viewing
    headers = ['ID', 'Question', 'Expected_Answer', 'Extracted_Prediction', 'Raw_Model_Output', 'Error_Category', 'Notes']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(errors)
        
    print(f"Exported {len(errors)} errors to {output_csv}. Ready for Teammate 2's audit.")

# Example usage ready for kickoff:
# generate_error_audit_csv('baseline_preds.jsonl', 'val_ground_truth.jsonl', 'baseline_audit_sheet.csv')