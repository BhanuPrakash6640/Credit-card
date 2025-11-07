"""
Sample Data Generator for Demo Purposes
Generates synthetic transaction data for testing without the full dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_transactions(
    n_samples: int = 1000,
    fraud_ratio: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate sample credit card transaction data
    
    Args:
        n_samples: Total number of transactions
        fraud_ratio: Proportion of fraudulent transactions
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic transaction data
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Generate time values (0-172800 seconds = 48 hours)
    times = np.random.randint(0, 172800, n_samples)
    times = np.sort(times)
    
    data = []
    
    # Generate normal transactions
    for i in range(n_normal):
        transaction = {
            'Time': times[i],
            'Amount': np.random.exponential(50) if np.random.random() > 0.1 else np.random.exponential(100),
        }
        
        # Generate V1-V28 features (normally distributed)
        for j in range(1, 29):
            transaction[f'V{j}'] = np.random.randn()
        
        transaction['Class'] = 0
        data.append(transaction)
    
    # Generate fraud transactions (different patterns)
    for i in range(n_fraud):
        transaction = {
            'Time': times[n_normal + i],
            # Fraudulent transactions tend to have higher amounts
            'Amount': np.random.exponential(150) if np.random.random() > 0.3 else np.random.exponential(300),
        }
        
        # Generate V1-V28 features (different distribution for fraud)
        for j in range(1, 29):
            # Some features have different means/stds for fraud
            if j in [1, 3, 4, 9, 10, 12, 14, 16, 17]:
                transaction[f'V{j}'] = np.random.randn() * 2 + np.random.choice([-2, 2])
            else:
                transaction[f'V{j}'] = np.random.randn()
        
        transaction['Class'] = 1
        data.append(transaction)
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Reorder columns
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = df[cols]
    
    return df


def save_sample_data(output_path: str = 'assets/sample_data.csv', n_samples: int = 1000):
    """
    Generate and save sample data to CSV
    
    Args:
        output_path: Where to save the CSV file
        n_samples: Number of transactions to generate
    """
    print(f"Generating {n_samples} sample transactions...")
    df = generate_sample_transactions(n_samples)
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    fraud_count = df['Class'].sum()
    normal_count = len(df) - fraud_count
    
    print(f"✓ Sample data saved to: {output_path}")
    print(f"  Total transactions: {len(df)}")
    print(f"  Normal: {normal_count} ({normal_count/len(df)*100:.1f}%)")
    print(f"  Fraud: {fraud_count} ({fraud_count/len(df)*100:.1f}%)")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    save_sample_data('assets/sample_data.csv', n_samples=2000)
    
    # Also create a smaller test set
    save_sample_data('assets/sample_test.csv', n_samples=500)
    
    print("\n✅ Sample data generation complete!")
