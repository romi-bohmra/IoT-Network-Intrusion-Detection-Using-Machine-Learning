# Iot Network Intrusion Detection using Machine Learning

## Project Overview

This project implements a hybrid intrusion detection system (IDS) for IoT networks, combining unsupervised anomaly detection at the packet level and supervised classification at the flow level. The system detects and classifies malicious network traffic from multiple attack types (DoS, DDoS, DNS spoofing, XSS, brute force).

The IDS pipeline is divided into three phases:

-> Phase 1 – Data Preprocessing: Clean, preprocess, and balance packet/flow datasets.

-> Phase 2 – Anomaly Detection (Unsupervised): Train a regularized autoencoder on benign traffic to detect anomalies.

-> Phase 3 – Flow-Level Classification (Supervised): Map flagged packets to flows and classify them using Random Forests.

## Phase 1 – Data Preparation

-> Load benign + attack packet data

-> Balance benign vs attack samples (97.5% benign, 2.5% attack)

-> Preprocess features (scaling + one-hot encoding)

-> Save train/test splits

## Phase 2 – Anomaly Detection (Autoencoder)

-> Train a regularized autoencoder on benign traffic

-> Detect suspicious packets using reconstruction error thresholds

-> Evaluate with accuracy, precision, recall, F1, ROC-AUC

-> Save flagged packets for flow-level mapping

## Phase 3 – Flow-Level Supervised Classification

-> Map suspicious packets to corresponding flows

-> Preprocess flow-level features

-> Train Random Forest classifier

-> Evaluate performance with confusion matrix & classification report

## Tech Stack

-> Programming: Python

-> Libraries: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn

-> Environment: Jupyter Notebook / Python 3.10+

## Results

-> Anomaly Detection (Phase 2): ~94% accuracy using regularized autoencoders

-> Flow-Level Classification (Phase 3): Strong supervised performance with Random Forests

-> Effective detection of diverse attack types with balanced precision and recall

## Learning Outcomes

-> Built a hybrid IDS combining anomaly detection + supervised learning

-> Learned feature engineering, scaling, threshold tuning, and autoencoder regularization

-> Mapped anomalies from packet-level to flow-level for refined attack classification
