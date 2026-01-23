
import json
import os
import random

# Fixed seed for reproducibility
random.seed(42)

GS_TEMPLATES = [
    {
        "topic": "History",
        "context": "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
        "correct": "The Eiffel Tower was finished in 1889 for a fair in Paris.",
        "hallucination": "The Eiffel Tower was completed in 1901 for a French wedding."
    },
    {
        "topic": "Science",
        "context": "Photosynthesis is the process where plants convert sunlight into chemical energy.",
        "correct": "Plants use photosynthesis to turn light from the sun into energy.",
        "hallucination": "Photosynthesis is the process where plants turn oxygen into sunlight."
    },
    {
        "topic": "Finance",
        "context": "The company reported a 15% increase in revenue, totaling $2.5 million this quarter.",
        "correct": "Revenue grew by 15% to reach 2.5 million dollars.",
        "hallucination": "The company reported a 50% decrease in revenue this quarter."
    },
    {
        "topic": "Geography",
        "context": "Mount Everest is located in the Himalayas at 8,849 meters above sea level.",
        "correct": "Everest stands at 8,849m in the Himalayan mountain range.",
        "hallucination": "Mount Everest is in the Alps at 6,500 meters elevation."
    },
     {
        "topic": "Technology",
        "context": "The new processor has 8 cores and runs at 3.5 GHz base frequency.",
        "correct": "An 8-core processor with 3.5 GHz base clock was released.",
        "hallucination": "The processor has 16 cores and runs at 5.0 GHz."
    }
]

def create_golden_dataset(output_path="data/golden_dataset.json"):
    dataset = []
    
    # Generate 1 faithful and 1 hallucinated example for each template
    for t in GS_TEMPLATES:
        # Faithful
        dataset.append({
            "text": f"Context: {t['context']} Answer: {t['correct']}",
            "label": 1,
            "topic": t['topic'],
            "type": "faithful"
        })
        # Hallucinated
        dataset.append({
            "text": f"Context: {t['context']} Answer: {t['hallucination']}",
            "label": 0,
            "topic": t['topic'],
            "type": "hallucination"
        })
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Created golden dataset with {len(dataset)} examples at {output_path}")

if __name__ == "__main__":
    create_golden_dataset()
