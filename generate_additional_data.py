import time
import pandas as pd
import concurrent.futures
from datetime import datetime
from prompts import MIXED_PROMPT, NEUTRAL_PROMPT
import ollama

# Configuration
ADDITIONAL_NEUTRAL = 2970
ADDITIONAL_MIXED = 3483
MAX_WORKERS = 20  # Adjust based on hardware
LOG_INTERVAL = 10
FINAL_CSV = 'data/additional_feedback.csv'

feedbacks = []
labels = []

def generate_single_sample(prompt, sentiment):
    """
    Function called once per sample: 
    - prompt: The text prompt for the model
    - sentiment: 'mixed' or 'neutral'
    Returns a tuple (feedback_content, sentiment_label)
    """
    # Example call to ollama
    response = ollama.chat(
        model='llama3.1', 
        messages=[{'role': 'user', 'content': prompt}]
    )
    content = response['message']['content']
    return content, sentiment

def parallel_generation(prompt, sentiment, num_samples):
    """
    Generate `num_samples` items in parallel using the specified `prompt` and sentiment label.
    Includes logging for the average time per sample and estimated time remaining.
    """
    start_time = time.time()

    local_feedbacks = []
    local_labels = []

    # Create a list of futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(generate_single_sample, prompt, sentiment)
            for _ in range(num_samples)
        ]

        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                content, label = f.result()
                local_feedbacks.append(content)
                local_labels.append(label)
            except Exception as e:
                print(f"Error generating {sentiment} feedback: {e}")
                continue
            
            # Logging progress every LOG_INTERVAL samples
            if i % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                avg_time_per_item = elapsed / i
                estimated_remaining = (num_samples - i) * avg_time_per_item
                estimated_remaining_min = estimated_remaining / 60.0

                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                print(f"Generated {i}/{num_samples} {sentiment} samples "
                      f"({(i / num_samples * 100):.2f}%)")
                print(f"Average time per sample: {avg_time_per_item:.2f}s")
                print(f"Estimated time remaining: {estimated_remaining_min:.2f} minutes")
                print("-" * 50)

    return local_feedbacks, local_labels

if __name__ == "__main__":
    overall_start = time.time()

    # Generate Mixed
    mixed_fb, mixed_lbl = parallel_generation(MIXED_PROMPT, 'mixed', ADDITIONAL_MIXED)
    feedbacks.extend(mixed_fb)
    labels.extend(mixed_lbl)

    # Generate Neutral
    neutral_fb, neutral_lbl = parallel_generation(NEUTRAL_PROMPT, 'neutral', ADDITIONAL_NEUTRAL)
    feedbacks.extend(neutral_fb)
    labels.extend(neutral_lbl)

    # Save results
    df_new = pd.DataFrame({
        'feedback': feedbacks,
        'label_name': labels
    })
    df_new.to_csv(FINAL_CSV, index=False)

    total_time = (time.time() - overall_start) / 60.0
    print("\nGeneration Complete!")
    print("Final Distribution:")
    print(df_new['label_name'].value_counts())
    print(f"\nTotal time taken: {total_time:.2f} minutes")