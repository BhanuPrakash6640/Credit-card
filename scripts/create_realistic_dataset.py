"""
Create a realistic credit card fraud dataset for training
"""

import pandas as pd
import numpy as np

def create_realistic_fraud_dataset(n_samples=50000, fraud_ratio=0.002):
    """
    Create a realistic credit card transaction dataset
    
    Args:
        n_samples: Total number of transactions
        fraud_ratio: Ratio of fraudulent transactions (default 0.2%)
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    print(f"Generating {n_samples} realistic transactions...")
    print(f"  Normal: {n_normal}")
    print(f"  Fraud: {n_fraud}")
    
    data = []
    
    # Generate normal transactions
    for i in range(n_normal):
        # Time distributed throughout 2 days
        time = np.random.randint(0, 172800)
        
        # Normal transaction amounts (mostly small purchases)
        if np.random.random() < 0.7:
            amount = np.random.gamma(2, 15)  # Smaller purchases
        else:
            amount = np.random.gamma(3, 30)  # Larger purchases
        
        # Generate V1-V28 features (PCA components)
        # Normal transactions cluster around 0
        v_features = {}
        for j in range(1, 29):
            if j in [1, 2, 3, 4, 5]:  # Some features more variable
                v_features[f'V{j}'] = np.random.randn() * 1.5
            else:
                v_features[f'V{j}'] = np.random.randn() * 0.8
        
        transaction = {
            'Time': time,
            **v_features,
            'Amount': max(0.01, amount),
            'Class': 0
        }
        data.append(transaction)
    
    # Generate fraudulent transactions
    for i in range(n_fraud):
        # Fraud more common at night
        if np.random.random() < 0.6:
            time = np.random.randint(0, 28800) + np.random.randint(79200, 172800)  # Night hours
        else:
            time = np.random.randint(0, 172800)
        
        # Fraud typically higher amounts
        if np.random.random() < 0.5:
            amount = np.random.gamma(5, 50)  # Moderate fraud
        else:
            amount = np.random.gamma(8, 100)  # Large fraud
        
        # Generate V1-V28 features
        # Fraud transactions have different patterns
        v_features = {}
        for j in range(1, 29):
            if j in [1, 3, 4, 9, 10, 12, 14, 16, 17, 18]:  # Key fraud indicators
                # Significantly different from normal
                v_features[f'V{j}'] = np.random.randn() * 3 + np.random.choice([-3, 3])
            elif j in [2, 5, 6, 7, 8, 11, 13, 15]:
                # Moderately different
                v_features[f'V{j}'] = np.random.randn() * 2 + np.random.choice([-1.5, 1.5])
            else:
                # Similar to normal
                v_features[f'V{j}'] = np.random.randn() * 0.9
        
        transaction = {
            'Time': time,
            **v_features,
            'Amount': max(0.01, amount),
            'Class': 1
        }
        data.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reorder columns to match real dataset
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = df[cols]
    
    return df


if __name__ == "__main__":
    # Create realistic dataset
    df = create_realistic_fraud_dataset(n_samples=100000, fraud_ratio=0.002)
    
    # Save as creditcard.csv
    output_path = 'creditcard.csv'
    df.to_csv(output_path, index=False)
    
    fraud_count = df['Class'].sum()
    normal_count = len(df) - fraud_count
    
    print(f"\n✅ Dataset created: {output_path}")
    print(f"   Total: {len(df)} transactions")
    print(f"   Normal: {normal_count} ({normal_count/len(df)*100:.2f}%)")
    print(f"   Fraud: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
    print(f"\n✅ Ready for training!")
