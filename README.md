# Stock Market Anomaly Detection
> **Capstone Project**: Automated Financial Anomaly Detection System

## 1. Research Overview
**Objective**: To build an automated, unsupervised machine learning system capable of detecting financial market anomalies—such as flash crashes, liquidity shocks, and extreme volatility events—without relying on labeled historical data.

The project focuses on a "leakage-free" approach, ensuring that all detection features are computed using only past data (rolling windows) to simulate a real-world trading environment. The analysis covers high-profile tech tickers (**QQQ, AAPL, MSFT, NVDA, AMZN, AMD**) during the volatile period from **Jan 2018 to Apr 2020**.

![Market Returns 2020](plots/market_returns_2020.png)
*Figure 1: Market returns context during the analysis period, highlighting the extreme volatility in early 2020.*

---

## 2. Feature Engineering
The core of this project lies in robust feature engineering that captures market stress signals independent of price direction. 

### Core Features
| Feature | Name | Definition & Rationale |
| :--- | :--- | :--- |
| **`ret_z`** | Returns Z-Score | Measures the deviation of today's return against the volatility of the trailing **63 days**. Identifying outliers relative to recent history. |
| **`vol_z`** | Volume Z-Score | Measures if trading volume is unusually high, using a **21-day rolling window**. Log-transformed to handle skewness. |
| **`range_pct`** | Range Percentile | Ranks today's intraday volatility (High - Low) against the distribution of the last **63 days**. |

![Feature Distribution](plots/feature_distribution.png)
*Figure 2: Distribution of engineered features, showing the statistical properties used for anomaly detection.*

---

## 3. Methodology & Models
I implemented and compared three distinct detection methodologies, ranging from simple heuristics to advanced density-based clustering.

### A. Rule-Based Detection (Baseline)
This method utilizes the 'Z_score' index as a hard threshold filter.
- **Return Shock**: |ret_z| > 2.5
- **Volume Shock**: |vol_z| > 2.5
- **Range Shock**: |range_pct| > 0.95 (95th percentile)

**Findings**: Not very Effective since it marks most of the market events as anomalies especially during march 2020 when the stock market was very unstable due to the COVID-19 pandemic.

![Threshold Anomalies](/plots/threshold_anomalies.png)
*Figure 3: Anomalies detected using simple severity thresholds.*

### B. Unsupervised Learning: K-Means Clustering
**Hypothesis**: Market data points form clusters representing different "regimes" (e.g., Calm, Bullish, Volatile). Anomalies are points that are structurally distant from these standard regime centroids.

- **Configuration**: `K=3` clusters (Calm, Moderate, Volatile) selected via Elbow Method.
- **Anomaly Definition**: Points where the distance to the nearest cluster center is in the **95th percentile**.

![Elbow and Silhouette](/plots/elbow_and_silhouette_kmeans.jpg)
*Figure 4: Elbow Method and Silhouette Score analysis used to determine optimal K=3.*

![Cluster Stability](/plots/kmeans_cluster_stability.png)
*Figure 5: Stability of K-Means clusters over time.*

### C. Density-Based: DBSCAN (Walk-Forward)
**Hypothesis**: Anomalies occur in low-density regions of the feature space. Unlike K-Means, DBSCAN does not force every point into a cluster; it explicitly labels outliers as "Noise" (-1).

- **Training**: Walk-Forward validation (retrained monthly) to prevent look-ahead bias and adapt to changing market conditions.
- **Parameters**: `eps=0.6`, `min_samples=20` (Tuned via k-Distance Graph).

![k-Distance Graph](/plots/k_distance_dbscan.png)
*Figure 6: k-Distance graph used to identify the optimal epsilon (`eps`) parameter.*

![DBSCAN Visualization](/plots/dbscan_cluster_visualization.png)
*Figure 7: Visualization of DBSCAN clusters and noise points (anomalies).*

### D. UNION Aproach (K-Means + DBSCAN)
**Hypothesis**: By combining the strengths of both clustering and density-based methods, I created a more robust anomaly detection system.

- **STRONG SIGNAL**: When both the models agree on an anomaly, it is marked as an anomaly with `STRONG SIGNAL` label.
- **WEAK SIGNAL**: When only one of the models agree on an anomaly, it is marked as an anomaly with `WEAK SIGNAL` label.

**Findings**: UNION approach performed better than both K-Means and DBSCAN. UNION approach correctly identified crash days in late February 2020 as early signs of the big market crash events in March 2020.

---

## 4. Comparative Analysis & Results

The project successfully identified major market stress events, particularly during the COVID-19 crash of February andMarch 2020.

### K-Means vs. Rule-Based
K-Means proved more adaptive than simple rules. While rule-based methods flagged most of the days as anomalies, K-Means could identify days that were anomalous due to unusual combinations of features (e.g., high volume but low price movement) that a linear severity score might miss.

![Rule-Based vs K-Means](/plots/simple_rules_Vs_kmeans.png)
*Figure 8: Comparison of anomalies detected by Rule-Based thresholds vs. K-Means.*

### K-Means vs. DBSCAN
DBSCAN offered the most "pure" anomaly detection. By designating points as noise, it filtered out high-variance days that were still part of a "volatile regime" (captured by K-Means) and focused only on truly unique market events.
- **DBSCAN Flag Rate**: ~3.84% of trading days on 2019 Validation Dataset.
- **K-Means Flag Rate**: ~4.76% of trading days on 2019 Validation Dataset.

**Findings**: DBSCAN and KMeans agreed on most of the anomalies in February 2020. But DBSCAN learned the trends of the market (Walk-Forward Validation) and when the market shifted it learned the new regime and became less sensitive and detected what truly were market events among these anomalies.

![K-Means vs DBSCAN](/plots/QQQ_anomalie_kmeans_vs_dbscan.png)
*Figure 9: Detailed comparison of K-Means vs. DBSCAN behavior on QQQ.*

## 5. Usage Instructions
To replicate this analysis:

1.  **Dependencies**: Ensure you have `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` installed.
2.  **Dataset**: The notebook downloads data using `kagglehub` (`jacksoncrow/stock-market-dataset`).
3.  **Run**: Execute `notebooks/StockAnomalyDetection.ipynb` sequentially.
