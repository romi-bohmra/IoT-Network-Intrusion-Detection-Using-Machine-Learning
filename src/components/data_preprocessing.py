import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
BENIGN_RATIO = 0.975   # 97.5% benign
ATTACK_RATIO = 0.025   # 2.5% attacks (combined)
N_TOTAL = 206_000      # ~200k benign + 6k attacks

# Attack classes of interest (make sure these match your actual filenames!)
ATTACK_CLASSES = [
    "DoS-HTTP_Flood",
    "DDoS-HTTP_Flood",
    "DNS_Spoofing",
    "XSS",
    "DictionaryBruteForce"
]


def load_packet_data(packet_folder="packet_data"):
    benign_files = glob.glob(os.path.join(packet_folder, "BenignTraffic*.csv"))
    benign_dfs = []
    for f in benign_files:
        df = pd.read_csv(f, low_memory=False)
        df['Label'] = 'Benign'
        benign_dfs.append(df)
    benign_df = pd.concat(benign_dfs, axis=0).reset_index(drop=True)

    attack_dfs = []
    for attack in ATTACK_CLASSES:
        f = os.path.join(packet_folder, f"{attack}.csv")
        if os.path.exists(f):
            df = pd.read_csv(f, low_memory=False)
            df['Label'] = attack
            attack_dfs.append(df)
        else:
            print(f"⚠️ Warning: file not found -> {f}")
    attack_df = pd.concat(attack_dfs, axis=0).reset_index(drop=True)

    return benign_df, attack_df


def generate_dataset(benign_df, attack_df, total_samples=N_TOTAL, random_state=RANDOM_SEED):
    np.random.seed(random_state)
    n_benign = int(total_samples * BENIGN_RATIO)
    n_attack = total_samples - n_benign

    benign_sample = benign_df.sample(n=n_benign, random_state=random_state)

    # random division across attack classes
    attack_subsets = {a: attack_df[attack_df['Label']==a] for a in ATTACK_CLASSES}
    proportions = np.random.dirichlet(np.ones(len(ATTACK_CLASSES)), size=1)[0]
    attack_counts = (proportions * n_attack).astype(int)

    attack_samples = []
    for attack, count in zip(ATTACK_CLASSES, attack_counts):
        available = len(attack_subsets[attack])
        if available > 0:
            count = min(count, available)
            sampled = attack_subsets[attack].sample(n=count, random_state=random_state)
            attack_samples.append(sampled)

    attack_sample = pd.concat(attack_samples, axis=0).reset_index(drop=True)
    dataset = pd.concat([benign_sample, attack_sample], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return dataset


def preprocess_dataset(df, drop_columns=['Flow ID', 'Timestamp']):
    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')
    df = df.fillna(0)
    X = df.drop(columns=['Label'])
    y = df['Label']

    X_numeric = X.select_dtypes(include=[np.number])
    X_categorical = pd.get_dummies(X.select_dtypes(exclude=[np.number]), drop_first=True)
    X_combined = pd.concat([X_numeric, X_categorical], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    return X_scaled, y, scaler, X_combined.columns, X_combined


# Main execution
if __name__ == "__main__":
    packet_folder = "packet_data"  # adjust folder name
    # Load packet-level data
    benign_df, attack_df = load_packet_data(packet_folder)
    print("Benign shape:", benign_df.shape)
    print("Attack shape:", attack_df.shape)
    
    # Generate sampled dataset
    dataset = generate_dataset(benign_df, attack_df)
     # Preprocess dataset
    X_scaled, y, scaler, used_features, X_combined = preprocess_dataset(dataset)
    print("Final dataset shape:", dataset.shape)
    print("Label distribution:\n", dataset['Label'].value_counts()) 
    print("✅ Data ready for Phase 2!")
   
    # Train/test split
     # Split train/test at packet level
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_scaled, y, dataset, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
     # Save to CSV
    pd.DataFrame(X_train, columns=used_features).to_csv("X_train.csv", index=False)
    pd.DataFrame(X_test, columns=used_features).to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    df_train.to_csv("df_train.csv", index=False)
    df_test.to_csv("df_test.csv", index=False)

    print("All datasets saved to CSV successfully.")