# ICB-BERT-Inference-Auditor & Benchmarker

## 📋 Overview
**ICB-BERT-Inference-Auditor** is a production-grade benchmarking suite designed to validite the performance, latency, and accuracy of AWS SageMaker NLP endpoints. 

It is specifically engineered for extraction models (e.g., BERT) that identify entities like **Order Numbers** and **Subtotals** from raw text. The tool operates as a "Test Harness," pulling ground-truth labeled data from S3, running inference against a live endpoint, and generating an executive-level dashboard with statistical visualizations.

## 🧠 Core Components
The architecture is modularized into three distinct "Personas":

1.  **👁️ The Eye (Parser):** Handles API response parsing. Isolated logic ensures that changes to the model's JSON schema only require updates in one location.
2.  **🧠 The Brain (Logic):** The judgment engine. 
    * **Data Quality Assessment:** Tags samples as "Gold" (Exact Match in input) or "Silver" (Fuzzy Match in input) before testing.
    * **Grading:** Uses fuzzy logic and regex to grade predictions (handles `1200` vs `1200.00` equality).
3.  **🎨 The Reporter (Viz):** * **Live Audit:** Real-time console logging with color-coded pass/fail indicators.
    * **Executive Dashboard:** A final ASCII summary of latency (P50/P95) and accuracy metrics.
    * **Visualizations:** Generates Seaborn charts for Latency vs. Length, Merchant distribution, and Accuracy binning.

## ⚙️ Prerequisites
* **Python 3.8+**
* **AWS Credentials** configured (usually via `~/.aws/credentials` or environment variables).
* **Access Permissions** to the S3 bucket and `sagemaker:InvokeEndpoint`.

## 📦 Installation

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/sagemaker-inference-auditor.git](https://github.com/your-username/sagemaker-inference-auditor.git)
cd sagemaker-inference-auditor

# 2. Install dependencies
pip install boto3 pandas matplotlib seaborn colorama numpy
