import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

X_train = pd.read_csv("X_train.csv").values  # load as NumPy array
X_test = pd.read_csv("X_test.csv").values

y_train = pd.read_csv("y_train.csv").squeeze().values  # Series → NumPy array
y_test = pd.read_csv("y_test.csv").squeeze().values

df_train = pd.read_csv("df_train.csv")
df_test = pd.read_csv("df_test.csv")

packet_df_test = pd.read_csv("packet_test_predictions.csv") 

with open("preds_data.pkl", "rb") as f:
    data = pickle.load(f)

# Now you can access the individual variables
preds = data["preds"]

def preprocess_flow_dataset(df, flow_id_col=None):
    if flow_id_col:
        df = df.drop(columns=[flow_id_col], errors="ignore")
    
    # Replace inf/-inf with NaN and fill NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Normalize label column name
    if "label" in df.columns:
        label_col = "label"
    elif "Label" in df.columns:
        label_col = "Label"
    else:
        raise KeyError("No label column found in dataframe")

    # Split features/labels
    X = df.drop(columns=[label_col], errors="ignore")
    y = df[label_col]

    # Numeric & categorical separation
    X_numeric = X.select_dtypes(include=[np.number])
    X_categorical = X.select_dtypes(exclude=[np.number])
    if not X_categorical.empty:
        X_categorical = pd.get_dummies(X_categorical, drop_first=True)

    X_combined = pd.concat([X_numeric, X_categorical], axis=1) if not X_categorical.empty else X_numeric

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    return X_scaled, y, scaler, X_combined.columns


def train_signature_classifier(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test, y_test):
    preds = clf.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Flow-Level Classifier Confusion Matrix")
    plt.show()
    print("Overall Accuracy:", accuracy_score(y_test, preds))


def get_flagged_flows(packet_df, preds, flow_df):
    packet_df = packet_df.copy()
    packet_df["flagged"] = preds

    # Create base Flow ID first
    packet_df['flow_base'] = (
        packet_df['src_ip'].astype(str) + '_' +
        packet_df['dst_ip'].astype(str) + '_' +
        packet_df['src_port'].astype(str) + '_' +
        packet_df['dst_port'].astype(str)
    )

    # Now filter suspicious packets
    suspicious_packets = packet_df[packet_df["flagged"] == 1]

    if suspicious_packets.empty:
        return pd.DataFrame()  # No flagged packets

    # Align flow_base in flow_df
    flow_df['flow_base'] = (
        flow_df['Flow ID'].str.replace('-', '_').str.rsplit('_', n=1).str[0]
    )

    suspicious_flows = flow_df.merge(
        suspicious_packets[['flow_base']].drop_duplicates(),
        on='flow_base',
        how='inner'
    )
    return suspicious_flows

if __name__ == "__main__":
    # flow_dataset: flow-level dataset loaded from Phase 1/2
    # packet_df_test: packet-level test data
    # preds: anomaly predictions from Phase 2 (1=suspicious, 0=benign)

     # Flow-level datasets (merge all attack categories + benign traffic)
    flow_files = [
        "flow_data/BenignTraffic.csv",
        "flow_data/DDoS-HTTP_Flood.csv",
        "flow_data/DictionaryBruteForce.csv",
        "flow_data/DNS_Spoofing.csv",
        "flow_data/DoS-HTTP_Flood.csv",
        "flow_data/DoS-HTTP_Flood1.csv",
        "flow_data/XSS.csv"
    ]

    # Read and concatenate the flow-level data
    flow_dfs = [pd.read_csv(f) for f in flow_files]
    flow_dataset = pd.concat(flow_dfs, ignore_index=True)


    # Map suspicious packets to flow-level
    suspicious_flows = get_flagged_flows(packet_df_test, preds, flow_dataset)

    if suspicious_flows.empty:
        print("No suspicious flows found. Exiting Phase 3.")
    else:
        print("Flagged suspicious flows shape:", suspicious_flows.shape)

        # Preprocess flows
        X_flow, y_flow, scaler, flow_features = preprocess_flow_dataset(suspicious_flows, flow_id_col='Flow ID')

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_flow, y_flow, test_size=0.2, random_state=42, stratify=y_flow
        )

        # Train classifier
        clf = train_signature_classifier(X_train, y_train)

        # Evaluate classifier
        evaluate_classifier(clf, X_test, y_test)

        print("✅ Phase 3 complete: Signature-based flow-level refinement done!")
