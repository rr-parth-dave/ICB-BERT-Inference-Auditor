import boto3
import pandas as pd
import time
import json
import random
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from botocore.exceptions import ClientError
from colorama import init, Fore, Back, Style

# --- SETUP & SILENCE ---
# Initialize Colorama
init(autoreset=True)
# Silence specific warnings for a clean output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# 1. ⚙️ CONFIGURATION CENTER
# ==============================================================================
class Config:
    # --- Toggles ---
    VERBOSE_LOGGING = False       # Set False for "Silent Mode" (Summary only)
    SHOW_CHARTS     = True       # Pop up the Seaborn chart at the end?

    # --- Benchmark Settings ---
    NUM_FILES       = 384         # Total files to analyze
    SAMPLES_PER_FILE= 5          # Rows per file to test
    MAX_CHARS       = 150_000    # Truncation limit (Speed & Cost safety)
    RANDOM_SEED     = 42         # Reproducibility

    # --- AWS Settings ---
    ENDPOINT_NAME   = 'icb-bert'
    REGION          = 'us-west-2'
    BUCKET          = 'datasciences-common-us-west-2-567141159380'
    PREFIX          = 'dev/ICB-Engines/Final_Eng_2_V_3_Training_Data/small test/all stores/'

# Initialize AWS Clients
s3 = boto3.client('s3', region_name=Config.REGION)
sm = boto3.client('sagemaker-runtime', region_name=Config.REGION)
random.seed(Config.RANDOM_SEED)

# ==============================================================================
# 2. 👁️ THE EYE: API RESPONSE PARSER (MODULAR)
# ==============================================================================
class TheEye:
    """
    🚨 EDIT THIS CLASS IF YOUR API JSON SCHEMA CHANGES 🚨
    This is the only place that touches the raw API response.
    """
    @staticmethod
    def parse(json_string):
        try:
            data = json.loads(json_string)
            
            # --- [CONFIGURATION START] ---
            # Update these keys to match your current API model
            target_block = data.get("Cleaned Predictions", {})
            
            order = target_block.get("Order_number", "")
            sub   = target_block.get("Subtotal", "")
            conf  = target_block.get("Order_number_Confidance", 0.0)
            # --- [CONFIGURATION END] ---
            
            return str(order).strip(), str(sub).strip(), float(conf)
            
        except Exception:
            return "", "", 0.0

# ==============================================================================
# 3. 🧠 THE BRAIN: LOGIC & VALIDATION LAYER
# ==============================================================================
class TheBrain:
    """
    The 'Judge'. Optimized for speed using pre-compiled Regex.
    Determines Data Quality (Gold/Silver) and Grades Predictions.
    """
    # 🚀 SPEED OPTIMIZATION: Compile regex once
    _clean_pattern = re.compile(r'[$,\s]')
    _float_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

    @staticmethod
    def normalize(text):
        """Standardizes text for 'Apple-to-Apple' comparison."""
        if not isinstance(text, str): return ""
        return TheBrain._clean_pattern.sub('', text).lower()

    @staticmethod
    def assess_quality(truncated_text, ground_truth):
        """
        Determines if the answer actually exists in the input.
        Returns: (Quality_Tag, Is_Valid_Bool)
        """
        if not ground_truth or str(ground_truth).lower() == 'nan':
            return f"{Fore.LIGHTBLACK_EX}VOID{Style.RESET_ALL}", False
            
        # 1. GOLD TIER: Exact Match (Fastest check)
        if str(ground_truth) in truncated_text:
            return f"{Fore.YELLOW}🥇 GOLD{Style.RESET_ALL}", True
            
        # 2. SILVER TIER: Fuzzy Match (Slower, normalized check)
        text_norm = TheBrain.normalize(truncated_text)
        gt_norm = TheBrain.normalize(ground_truth)
        if gt_norm and gt_norm in text_norm:
            return f"{Fore.CYAN}🥈 SLVR{Style.RESET_ALL}", True
            
        return f"{Fore.LIGHTBLACK_EX}🌑 VOID{Style.RESET_ALL}", False

    @staticmethod
    def grade(gt, pred):
        """Grades the prediction against the ground truth."""
        if not gt or not pred: return False
        
        norm_gt = TheBrain.normalize(str(gt))
        norm_pred = TheBrain.normalize(str(pred))
        
        if not norm_gt or not norm_pred: return False

        # Rule 1: Exact Match
        if norm_gt == norm_pred: return True
        # Rule 2: Containment
        if norm_gt in norm_pred or norm_pred in norm_gt: return True
        # Rule 3: Numeric Equality (Handle 1200 vs 1200.00)
        try:
            val_gt = float(TheBrain._float_pattern.findall(norm_gt)[0])
            val_pred = float(TheBrain._float_pattern.findall(norm_pred)[0])
            if abs(val_gt - val_pred) < 0.01:
                return True
        except:
            pass
        return False

# ==============================================================================
# 4. 🎨 THE REPORTER: VISUALIZATION & LOGGING
# ==============================================================================
class TheReporter:
    @staticmethod
    def log_live_audit(status, quality_tag, label, gt, pred, conf):
        """Prints a beautiful, aligned audit log for a single sample."""
        if not Config.VERBOSE_LOGGING: return
        icon = f"{Fore.GREEN}✅{Style.RESET_ALL}" if status else f"{Fore.RED}❌{Style.RESET_ALL}"
        print(f"   {icon} {Fore.MAGENTA}{label}{Style.RESET_ALL}: [{quality_tag}] GT='{gt}' -> PRED='{pred}' {Fore.LIGHTBLACK_EX}(Conf: {conf:.0f}%){Style.RESET_ALL}")

    @staticmethod
    def log_file_summary(idx, total, merchant, lat, ord_acc, sub_acc):
        """Prints a high-level summary line for the file."""
        if not Config.VERBOSE_LOGGING: return
        o_col = Fore.GREEN if ord_acc == 100 else (Fore.YELLOW if ord_acc > 50 else Fore.RED)
        s_col = Fore.GREEN if sub_acc == 100 else (Fore.YELLOW if sub_acc > 50 else Fore.RED)
        print(f"{Fore.BLUE}📂 [{idx}/{total}]{Style.RESET_ALL} Store {merchant} | ⚡ {lat:.0f}ms | 🎯 Ord:{o_col}{ord_acc:.0f}%{Style.RESET_ALL} Sub:{s_col}{sub_acc:.0f}%{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLACK_EX}{'-'*85}{Style.RESET_ALL}")

    @staticmethod
    def print_executive_dashboard(df):
        """Prints the final ASCII summary to the terminal."""
        if df.empty: return print("❌ No data collected.")

        # --- PRE-CALCULATIONS ---
        valid_orders = df[df["Valid_Order"] == True]
        valid_subs   = df[df["Valid_Sub"] == True]
        
        ord_acc = valid_orders["Match_Order"].mean() * 100 if not valid_orders.empty else 0
        sub_acc = valid_subs["Match_Sub"].mean() * 100 if not valid_subs.empty else 0
        
        valid_both = df[(df["Valid_Order"] == True) & (df["Valid_Sub"] == True)]
        perf_acc = 0
        if not valid_both.empty:
            perf_acc = valid_both[(valid_both["Match_Order"]) & (valid_both["Match_Sub"])].shape[0] / len(valid_both) * 100
        
        corr = df['Length'].corr(df['Latency'])
        p95 = df['Latency'].quantile(0.95)
        
        # --- ASCII DASHBOARD ---
        print("\n" + "#"*80)
        print(f"{Fore.WHITE}{Back.BLUE} 📊 BENCHMARK EXECUTIVE SUMMARY | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {Style.RESET_ALL}".center(90))
        print("#"*80)
        
        print(f"""
1. OPERATIONAL SCOPE
   - Endpoint:       {Fore.CYAN}{Config.ENDPOINT_NAME}{Style.RESET_ALL}
   - Files Read:     {df['File'].nunique()}
   - Total Samples:  {len(df)}
   - Max Chars Sent: {Config.MAX_CHARS}

2. LATENCY PERFORMANCE (Response Time)
   - Average:        {Fore.YELLOW}{df['Latency'].mean():.2f} ms{Style.RESET_ALL}
   - Median (P50):   {df['Latency'].median():.2f} ms
   - P95 (Tail):     {Fore.RED}{p95:.2f} ms{Style.RESET_ALL}  <-- 95% faster than this
   - Fastest:        {df['Latency'].min():.2f} ms
   - Slowest:        {df['Latency'].max():.2f} ms

3. SCALABILITY ANALYSIS
   - Correlation:    {Fore.MAGENTA}{corr:.4f}{Style.RESET_ALL} (Length vs Latency)
   - Insight:        {'Strong relationship.' if corr > 0.7 else 'Weak relationship.'}

4. ACCURACY METRICS (Valid Samples Only)
   - Order ID:       {Fore.GREEN if ord_acc > 90 else Fore.YELLOW}{ord_acc:.1f}%{Style.RESET_ALL}  (n={len(valid_orders)})
   - Subtotal:       {Fore.GREEN if sub_acc > 90 else Fore.YELLOW}{sub_acc:.1f}%{Style.RESET_ALL}  (n={len(valid_subs)})
   - Perfect Rows:   {perf_acc:.1f}%  (Both correct)
        """)

    @staticmethod
    def generate_all_charts(df):
        """Orchestrates the creation of all three world-class charts."""
        if df.empty or not Config.SHOW_CHARTS: return
        print(f"\n{Fore.CYAN}🎨 Generating World-Class Charts (Interactive Mode)...{Style.RESET_ALL}")
        sns.set_theme(style="whitegrid", context="talk") 

        TheReporter._chart_latency_vs_length(df)
        TheReporter._chart_latency_by_merchant(df)
        TheReporter._chart_accuracy_by_length(df)
        
        print(f"{Fore.GREEN}✅ All charts displayed!{Style.RESET_ALL}")

    @staticmethod
    def _chart_latency_vs_length(df):
        """Chart 1: Latency vs. Input Length with Regression Line."""
        plt.figure(figsize=(14, 8))
        
        # Fix: Added 'hue' to match usage or removed palette if hue isn't needed. 
        # Using hue="Merchant" enables the palette without warning.
        sns.scatterplot(data=df, x="Length", y="Latency", hue="Merchant", palette="viridis", 
                        alpha=0.7, s=100, edgecolor="w", linewidth=0.5, legend=False)
        
        sns.regplot(data=df, x="Length", y="Latency", scatter=False, color="#2c3e50", 
                    line_kws={"linestyle": "--", "linewidth": 3, "alpha": 0.8})

        plt.title(f"API Latency vs Input Text Size\n(n={len(df)} samples)", fontsize=20, fontweight='bold', pad=20)
        plt.xlabel("Input Characters Sent", fontsize=14, labelpad=10)
        plt.ylabel("Latency (ms)", fontsize=14, labelpad=10)
        
        # Stats box
        stats_txt = f"Avg: {df['Latency'].mean():.0f}ms\nP95: {df['Latency'].quantile(0.95):.0f}ms"
        plt.gca().text(0.02, 0.95, stats_txt, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

        plt.tight_layout()
        plt.show() # <--- Just show, do not save

    @staticmethod
    def _chart_latency_by_merchant(df):
        """Chart 2: Latency Distribution by Merchant (Box Plot)."""
        plt.figure(figsize=(14, 8))
        
        sorted_merchants = df.groupby('Merchant')['Latency'].median().sort_values().index
        
        # Fix: Added hue="Merchant" and legend=False to silence palette warning
        sns.boxplot(data=df, x="Merchant", y="Latency", hue="Merchant", order=sorted_merchants, palette="mako", legend=False)
        
        # Fix: Added hue="Merchant"
        sns.stripplot(data=df, x="Merchant", y="Latency", hue="Merchant", order=sorted_merchants, 
                      color='black', alpha=0.3, jitter=True, size=4, legend=False)

        plt.title("Latency Distribution by Merchant Store", fontsize=20, fontweight='bold', pad=20)
        plt.xlabel("Merchant ID", fontsize=14, labelpad=10)
        plt.ylabel("Latency (ms)", fontsize=14, labelpad=10)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show() # <--- Just show, do not save

    @staticmethod
    def _chart_accuracy_by_length(df):
        """Chart 3: Accuracy vs. Input Length (Dual-Axis Line Chart)."""
        bins = [0, 50000, 100000, 150000, 200000, float('inf')]
        labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k+']
        df['Bin'] = pd.cut(df['Length'], bins=bins, labels=labels)

        # Aggregate Data (using observed=False to silence categorical warning)
        bin_stats = df.groupby('Bin', observed=False).agg({
            'Match_Order': lambda x: x[df.loc[x.index, 'Valid_Order']].mean() * 100,
            'Match_Sub': lambda x: x[df.loc[x.index, 'Valid_Sub']].mean() * 100
        }).reset_index()
        
        melted_df = bin_stats.melt('Bin', var_name='Metric', value_name='Accuracy')
        metric_map = {'Match_Order': 'Order Accuracy', 'Match_Sub': 'Subtotal Accuracy'}
        melted_df['Metric'] = melted_df['Metric'].map(metric_map)

        plt.figure(figsize=(14, 8))
        
        # Fix: Replaced deprecated 'scale' with 'markersize' and 'linewidth'
        sns.pointplot(data=melted_df, x='Bin', y='Accuracy', hue='Metric', 
                      palette=['#2ecc71', '#3498db'], markers=['o', 's'], 
                      markersize=10, linewidth=2)

        plt.title("Model Accuracy vs. Input Text Length", fontsize=20, fontweight='bold', pad=20)
        plt.xlabel("Input Character Count (Binned)", fontsize=14, labelpad=10)
        plt.ylabel("Accuracy (%) on Valid Samples", fontsize=14, labelpad=10)
        plt.ylim(0, 105)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend(title=None, loc='lower left', frameon=True)
        
        plt.tight_layout()
        plt.show() # <--- Just show, do not save

# ==============================================================================
# 5. ⚡ THE ENGINE: MAIN LOOP
# ==============================================================================
def get_files():
    print(f"{Fore.CYAN}📡 Connecting to S3 ({Config.BUCKET})...{Style.RESET_ALL}")
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=Config.BUCKET, Prefix=Config.PREFIX):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.csv'): files.append(obj['Key'])
    
    selected = random.sample(files, min(len(files), Config.NUM_FILES))
    print(f"{Fore.GREEN}✅ Selected {len(selected)} random files.{Style.RESET_ALL}\n")
    return selected


def run_benchmark():
    files = get_files()
    global_results = []
    
    print(f"{Fore.WHITE}{Back.BLUE} 🚀 ENGINE STARTED | Max Length: {Config.MAX_CHARS} chars {Style.RESET_ALL}")
    print("-" * 85)

    start_time_total = time.time() # Track total time for ETA

    for i, file_key in enumerate(files):
        file_name = file_key.split('/')[-1]
        f_stats = [] 
        
        # --- PROGRESS & ETA CALCULATION ---
        elapsed = time.time() - start_time_total
        avg_time_per_file = elapsed / (i + 1)
        remaining_files = len(files) - (i + 1)
        eta_seconds = avg_time_per_file * remaining_files
        eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))

        # --- FIX: PADDING WITH SPACES ---
        # We construct the message, then add spaces (.ljust(120)) to wipe previous long lines
        msg = f"{Fore.BLUE}📂 Processing [{i+1}/{len(files)}] {file_name}... {Fore.LIGHTBLACK_EX}(ETA: {eta_str}){Style.RESET_ALL}"
        print(msg + " " * 20, end='\r') # Add explicit padding spaces
        sys.stdout.flush() # Force update immediately
        
        try:
            # 1. Load & Sample
            obj = s3.get_object(Bucket=Config.BUCKET, Key=file_key)
            df = pd.read_csv(obj['Body'], dtype=str)
            if len(df) > Config.SAMPLES_PER_FILE:
                df = df.sample(n=Config.SAMPLES_PER_FILE, random_state=Config.RANDOM_SEED)

            for _, row in df.iterrows():
                # --- A. PREPARE ---
                text = str(row.get('sanitized_text', ''))[:Config.MAX_CHARS]
                if len(text) < 10: continue 

                gt_ord = str(row.get('ground_truth_order_id', '')).strip()
                gt_sub = str(row.get('ground_truth_subtotal', '')).strip()
                merch  = str(row.get('merchant_id', 'Unknown'))

                # --- B. ASSESS QUALITY ---
                q_tag_o, valid_o = TheBrain.assess_quality(text, gt_ord)
                q_tag_s, valid_s = TheBrain.assess_quality(text, gt_sub)

                # --- C. INFERENCE ---
                payload = json.dumps({"text": text, "max_chars": Config.MAX_CHARS})
                start = time.time()
                try:
                    resp = sm.invoke_endpoint(EndpointName=Config.ENDPOINT_NAME, ContentType='application/json', Body=payload)
                    lat = (time.time() - start) * 1000
                    p_ord, p_sub, conf = TheEye.parse(resp['Body'].read().decode('utf-8'))
                except Exception:
                    lat, p_ord, p_sub, conf = 0, "", "", 0.0

                # --- D. GRADE ---
                match_o = TheBrain.grade(gt_ord, p_ord)
                match_s = TheBrain.grade(gt_sub, p_sub)

                # --- E. LOGGING ---
                if Config.VERBOSE_LOGGING:
                    # Clear line fully before printing log
                    print(" " * 120, end='\r') 
                    if valid_o or p_ord: TheReporter.log_live_audit(match_o, q_tag_o, "ORD", gt_ord, p_ord, conf)
                    if valid_s or p_sub: TheReporter.log_live_audit(match_s, q_tag_s, "SUB", gt_sub, p_sub, 0.0)

                # --- F. RECORD ---
                record = {"Merchant": merch, "File": file_name, "Length": len(text), "Latency": lat,
                          "Valid_Order": valid_o, "Valid_Sub": valid_s, "Match_Order": match_o, "Match_Sub": match_s}
                f_stats.append(record)
                global_results.append(record)

            # --- G. FILE SUMMARY ---
            if f_stats and Config.VERBOSE_LOGGING:
                f_df = pd.DataFrame(f_stats)
                acc_o = f_df[f_df['Valid_Order']==True]['Match_Order'].mean() * 100 if f_df['Valid_Order'].any() else 0.0
                acc_s = f_df[f_df['Valid_Sub']==True]['Match_Sub'].mean() * 100 if f_df['Valid_Sub'].any() else 0.0
                TheReporter.log_file_summary(i+1, len(files), merch, f_df['Latency'].mean(), acc_o, acc_s)

        except Exception as e:
            # Clear line before printing error
            print(" " * 120, end='\r')
            print(f"{Fore.RED}⚠️ Error on {file_name}: {e}{Style.RESET_ALL}")

    # Final cleanup
    print(" " * 120, end='\r')
    print(f"{Fore.GREEN}✅ Benchmarking Complete!{Style.RESET_ALL}")
    
    return pd.DataFrame(global_results)
# ==============================================================================
# 6. 🏆 RUN
# ==============================================================================
if __name__ == "__main__":
    df_results = run_benchmark()
    
    if not df_results.empty:
        TheReporter.print_executive_dashboard(df_results)
        TheReporter.generate_all_charts(df_results)
    else:
        print(f"{Fore.RED}❌ No data collected.{Style.RESET_ALL}")
