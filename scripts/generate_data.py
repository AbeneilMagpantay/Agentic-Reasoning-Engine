"""
generate_data.py - Hallucination Detection Training Data Generator

Optimized script for generating synthetic training data for ModernBERT.
Focuses on: entity swaps, negation errors, and numerical inconsistencies.
"""

import json
import random
import os
from typing import List, Dict

# Diverse templates covering different domains and hallucination types
TEMPLATES = [
    # History
    {
        "topic": "History",
        "context": "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
        "correct": "The Eiffel Tower was finished in 1889 for a fair in Paris.",
        "hallucination": "The Eiffel Tower was completed in 1901 for a French wedding."
    },
    {
        "topic": "History", 
        "context": "World War II ended in 1945 after the surrender of Japan.",
        "correct": "The war ended in 1945 when Japan surrendered.",
        "hallucination": "World War II ended in 1942 after Germany surrendered."
    },
    # Science
    {
        "topic": "Science",
        "context": "Photosynthesis is the process where plants convert sunlight into chemical energy.",
        "correct": "Plants use photosynthesis to turn light from the sun into energy.",
        "hallucination": "Photosynthesis is the process where plants turn oxygen into sunlight."
    },
    {
        "topic": "Science",
        "context": "Water molecules consist of two hydrogen atoms and one oxygen atom.",
        "correct": "H2O contains two hydrogen atoms bonded to one oxygen atom.",
        "hallucination": "Water is made of three oxygen atoms and one hydrogen atom."
    },
    # Finance
    {
        "topic": "Finance",
        "context": "The company reported a 15% increase in revenue, totaling $2.5 million this quarter.",
        "correct": "Revenue grew by 15% to reach 2.5 million dollars.",
        "hallucination": "The company reported a 50% decrease in revenue this quarter."
    },
    {
        "topic": "Finance",
        "context": "Bitcoin reached an all-time high of $69,000 in November 2021.",
        "correct": "BTC hit its peak price of $69,000 in late 2021.",
        "hallucination": "Bitcoin reached $100,000 in January 2020."
    },
    # Legal
    {
        "topic": "Legal",
        "context": "The defendant was found guilty of fraud and sentenced to 5 years in prison.",
        "correct": "A 5-year prison sentence was given to the defendant for fraud.",
        "hallucination": "The defendant was acquitted of all charges and released."
    },
    # Medical
    {
        "topic": "Medical",
        "context": "The patient was diagnosed with Type 2 diabetes and prescribed metformin.",
        "correct": "Metformin was prescribed after a Type 2 diabetes diagnosis.",
        "hallucination": "The patient was diagnosed with Type 1 diabetes and given insulin."
    },
    # Technology
    {
        "topic": "Technology",
        "context": "The new processor has 8 cores and runs at 3.5 GHz base frequency.",
        "correct": "An 8-core processor with 3.5 GHz base clock was released.",
        "hallucination": "The processor has 16 cores and runs at 5.0 GHz."
    },
    # Geography
    {
        "topic": "Geography",
        "context": "Mount Everest is located in the Himalayas at 8,849 meters above sea level.",
        "correct": "Everest stands at 8,849m in the Himalayan mountain range.",
        "hallucination": "Mount Everest is in the Alps at 6,500 meters elevation."
    },
]


def apply_variation(text: str) -> str:
    """Apply random linguistic variations to avoid overfitting."""
    variations = [
        lambda t: t,
        lambda t: f"It is confirmed that {t.lower()}",
        lambda t: f"According to the context, {t}",
        lambda t: f"Based on the information provided, {t.lower()}",
        lambda t: t.replace("The ", "Based on the records, the "),
        lambda t: f"The document states that {t.lower()}",
    ]
    return random.choice(variations)(text)


def generate_hallucination(base: Dict) -> str:
    """Generate diverse hallucination patterns."""
    hallucinations = [
        base["hallucination"],
        f"Actually, {base['hallucination'].lower()}",
        f"The text indicates that {base['hallucination'].lower()}",
        # Random noise injection
        "The provided text does not contain enough information to answer.",
        "This information is not available in the given context.",
    ]
    return random.choice(hallucinations)


def generate_hallucination_dataset(
    output_path: str = "data/hallucination_dataset.jsonl",
    num_samples: int = 1000
) -> List[Dict]:
    """
    Generates a synthetic dataset for training a hallucination grader.
    Labels: 1 (Faithful/Correct), 0 (Hallucinated)
    
    Returns the dataset and saves to JSONL format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dataset = []
    
    for i in range(num_samples):
        # Select random template
        base = random.choice(TEMPLATES)
        
        # Decide label: 1=faithful, 0=hallucinated
        label = random.choice([0, 1])
        
        if label == 1:
            # Faithful: apply variations to correct answer
            answer = apply_variation(base["correct"])
        else:
            # Hallucinated: generate diverse hallucination
            answer = generate_hallucination(base)
        
        # Format for ModernBERT training
        entry = {
            "text": f"Context: {base['context']} Answer: {answer}",
            "label": label,
            "topic": base["topic"]
        }
        dataset.append(entry)
    
    # Shuffle for i.i.d. distribution
    random.shuffle(dataset)
    
    # Save as JSONL (one JSON object per line)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    # Also save as regular JSON for easy viewing
    json_path = output_path.replace(".jsonl", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ Generated {len(dataset)} samples")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Label distribution: {sum(d['label'] for d in dataset)} faithful, {len(dataset) - sum(d['label'] for d in dataset)} hallucinated")
    
    return dataset


if __name__ == "__main__":
    # Generate 1000+ samples
    dataset = generate_hallucination_dataset(
        output_path="data/hallucination_dataset.jsonl",
        num_samples=1200  # Generate extra for train/val split
    )
