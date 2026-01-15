- **Heartbeat Segmentation**: Post-beamforming, a high-pass filter above 50 CPM extracts the heart signal. Residual respiratory interference introduces rotations in the heartbeat signal, complicating segmentation. A noniterative algorithm aligns and rotates segments to match consecutive heartbeats, improving accuracy.

- **Synchronizing Data Streams**: Initial timing offsets between acoustic recordings and ground truth (ECG/PPG) are corrected by a 5-second manual offset. Fine-tuning involves manually aligning the first heartbeat across streams. This ensures accurate heart rate and R–R interval comparisons.

- **Handling Missed Heartbeats**: To maintain synchronization, each R–R interval is matched across data streams. If beats do not mutually match, they are excluded. For unmatched intervals, interpolation increases median error from 28 to 32 ms and the 90th percentile error from 75 to 89 ms.

- **Statistical Analysis**: Heart rate and R–R interval errors were analyzed using Python. Metrics include bias (mean error), precision (standard deviation), MAE, and 90th percentile error. Scatter plots show limits of agreement. ICC and CCC are calculated for reliability measures.

- **Clinical Reporting**: Continuous clinical variables are reported as mean ± standard deviation, while categorical variables are reported as number (percentage). Statistical analyses were conducted using SAS University Edition to ensure robustness and reproducibility of results.