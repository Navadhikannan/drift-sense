# DRIFT-SENSE

## AI-Assisted Visual Drift Localization

Drift-Sense is a computer-vision-based visual localization system designed to identify the position of a reference pattern inside a search image.

The project combines classical computer vision techniques with an AI-based candidate reranking approach to improve localization accuracy, particularly in visually ambiguous and noisy conditions.

---

## Project Overview

Traditional template matching generally selects the candidate location with the highest similarity score.

However, visually similar regions may produce high matching scores even when they are not the correct target. This can result in significant localization errors.

Drift-Sense addresses this problem by generating multiple candidate locations and evaluating them using both visual features and relative candidate information.

The proposed AI-V2 system uses a Random Forest model to rerank candidate locations based on their relative characteristics.

### Core Concept

Instead of asking:

> Which candidate has the highest absolute similarity?

Drift-Sense AI-V2 asks:

> Which candidate is stronger relative to the other candidates?

---

# Problem Statement

Visual localization is the process of determining the position of a target or reference pattern within a larger search image.

Traditional template matching can fail when:

- Multiple regions have similar visual characteristics
- Image noise affects similarity scores
- The strongest template match is not the true target
- Similar structures occur at multiple locations
- Candidate scores are close to each other

Therefore, a more robust candidate-selection mechanism is required.

---

# Objectives

The main objectives of Drift-Sense are:

1. Generate candidate locations from a search image.
2. Extract multiple visual similarity features.
3. Evaluate candidates using classical computer vision techniques.
4. Introduce relative candidate-ranking features.
5. Train an AI model to distinguish stronger candidates.
6. Estimate the final target coordinates.
7. Evaluate localization accuracy under different noise conditions.
8. Compare AI-assisted localization against the classical baseline.

---

# System Architecture

```text
                  ┌──────────────────┐
                  │  Reference Image │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Template         │
                  │ Extraction       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Candidate        │
                  │ Generation       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Feature          │
                  │ Extraction       │
                  └────────┬─────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ Classical CV Verification│
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ AI-V2 Candidate          │
              │ Reranking                │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ Confidence Analysis      │
              └────────────┬─────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Final X,Y        │
                  │ Localization     │
                  └──────────────────┘

---

# Classical Computer Vision Pipeline

The baseline localization system evaluates candidates using multiple visual measurements.

Visual Features
Template similarity
Gray-level similarity
Edge similarity
Gradient similarity
Structural similarity
Spatial candidate information

The classical pipeline provides the initial candidate ranking.

The AI-V2 system then uses additional relative information to determine whether another candidate may be more reliable.

# AI-V2 Candidate Reranking

AI-V2 introduces a relative candidate-ranking approach.

Rather than relying only on the absolute score of a candidate, the model considers how the candidate compares with other candidates within the same candidate group.

AI-V2 Features
Feature	Description
template_gap	Difference between candidate and best template score
gray_gap	Relative gray-level similarity
edge_gap	Relative edge similarity
gradient_gap	Relative gradient similarity
structural_gap	Relative structural similarity
combined_score	Combined candidate similarity
combined_gap	Relative combined-score difference
normalized_rank	Candidate rank normalized within the group
distance_from_template_best	Spatial distance from the template-best candidate
nearest_candidate_distance	Distance to the nearest competing candidate
neighborhood_density_50	Candidate density within 50 pixels
neighborhood_density_100	Candidate density within 100 pixels
Machine Learning Model

The AI-V2 candidate reranker uses a:

Random Forest Classifier

The model receives the relative candidate features and predicts the probability that a candidate represents the correct target.

The candidate with the strongest AI ranking is selected for final localization.

# Dataset

The AI candidate dataset contains:

30 image samples
4 noise conditions
30 candidates per image
Noise Conditions
Clean
Low
Medium
High
Dataset Statistics
Category	Count
Candidate rows	3,600
Positive candidates	112
Negative candidates	3,488
Candidate groups	120

The final localization evaluation contains:

120 evaluated cases

Training and Testing

The candidate groups were divided into separate training and testing groups.

Parameter	Value
Training groups	90
Testing groups	30
Training rows	2,700
Testing rows	900
Model	Random Forest

The group-based split prevents candidates from the same image/noise group from being mixed between training and testing.

AI-V2 Classification Performance

The trained AI-V2 candidate classifier achieved:

Metric	Result
Accuracy	99.67%
ROC-AUC	0.9994
Positive Precision	90.00%
Positive Recall	100.00%
Positive F1-score	94.74%

These results indicate that the model can effectively distinguish positive candidates from negative candidates within the evaluation dataset.

# Feature Importance

The most important AI-V2 features were:

Feature	Importance
Gradient gap	23.49%
Structural gap	22.48%
Combined gap	15.67%
Gray gap	11.69%
Template gap	10.34%
Normalized rank	5.36%
Distance from template best	5.35%
Edge gap	3.73%

Gradient and structural differences were the most influential features in the trained Random Forest model.

Final Localization Benchmark

The final benchmark compares the classical V5.1 baseline with AI-V2 across 120 evaluated cases.

Metric	Baseline V5.1	AI-V2
Mean error	13.691 px	3.983 px
Median error	1.077 px	1.053 px
Worst error	629.048 px	71.732 px
@1 px	41.67%	43.33%
@2 px	88.33%	93.33%
@5 px	88.33%	93.33%
@20 px	88.33%	93.33%
@50 px	93.33%	96.67%
Overall Improvement
Mean Localization Error
Baseline V5.1 : 13.691 px
AI-V2         :  3.983 px
Mean Error Reduction

70.91%

Worst-Case Error
Baseline V5.1 : 629.048 px
AI-V2         :  71.732 px
Worst-Case Error Reduction

88.60%

Major Improvement Cases
Sample 013 — High Noise
Baseline error : 629.048 px
AI-V2 error    :   1.726 px


Improvement    : 627.322 px

This corresponds to approximately:

99.73% error reduction

Sample 026
Baseline error : 43.000 px
AI-V2 error    :  0.200 px


Improvement    : 42.800 px

This corresponds to approximately:

99.53% error reduction

Pass Rate Comparison

The percentage of cases within different localization error thresholds is shown below.

Threshold	Baseline	AI-V2
≤ 1 px	41.67%	43.33%
≤ 2 px	88.33%	93.33%
≤ 5 px	88.33%	93.33%
≤ 20 px	88.33%	93.33%
≤ 50 px	93.33%	96.67%

AI-V2 improves the pass rate at every evaluated threshold.

Confidence Analysis

AI-V2 also produces candidate probabilities that can be used to estimate localization confidence.

The confidence analysis classified the evaluated test groups into three categories:

Confidence	Cases	Mean Error
HIGH	28	1.633 px
MEDIUM	1	21.900 px
LOW	1	21.100 px

Confidence is determined using the difference between the strongest and second-best candidate probabilities.

A larger probability gap indicates a more decisive candidate ranking.

Failure Cases and Limitations

AI-V2 does not improve every individual localization case.

Across the 120 evaluated cases:

Improved : 6
Worse    : 1
Equal    : 113
Difficult Case — Sample 009

The AI-V2 system retains an error of approximately:

71.73 px

This represents a visually ambiguous case where the correct target is difficult to distinguish from competing regions.

Important Observation

A high model probability does not always guarantee low localization error.

This indicates that confidence calibration and improved visual representations are potential areas for future development.

# Project Structure

drift-sense/
│
├── data/
│   ├── sample_001/
│   ├── sample_002/
│   └── ...
│
├── localization/
│   ├── baseline_v5_1.py
│   ├── baseline_v5_2.py
│   └── baseline_v5_3.py
│
├── evaluation/
│   ├── build_ai_dataset.py
│   ├── build_ai_v2_dataset.py
│   ├── train_ai_reranker.py
│   ├── train_ai_v2_reranker.py
│   ├── evaluate_ai_reranker.py
│   ├── evaluate_ai_v2_reranker.py
│   ├── analyze_ai_confidence.py
│   ├── final_benchmark.py
│   └── generate_final_plots.py
│
├── results/
│   ├── ai/
│   ├── ai_v2/
│   └── final_benchmark/
│
├── requirements.txt
└── README.md
Installation

Clone the repository:

git clone https://github.com/Navadhikannan/drift-sense.git

Move into the project directory:

cd drift-sense

Install the required Python packages:

pip install -r requirements.txt
Running the Final Benchmark

Run the final benchmark using:

python evaluation/final_benchmark.py

The benchmark generates:

results/final_benchmark/model_comparison.csv
results/final_benchmark/pass_rates.csv
results/final_benchmark/sample_improvements.csv
Generating Final Visualizations

Run:

python evaluation/generate_final_plots.py

The plots are generated inside:

results/final_benchmark/plots/

The visualization stage produces:

01_mean_error_comparison.png
02_worst_case_error.png
03_pass_rate_comparison.png
04_per_sample_improvement.png
05_top_improvements.png
Reproducing the AI-V2 Pipeline
Step 1 — Build the candidate dataset
python evaluation/build_ai_dataset.py
Step 2 — Build the AI-V2 relative ranking dataset
python evaluation/build_ai_v2_dataset.py
Step 3 — Train the original AI reranker
python evaluation/train_ai_reranker.py
Step 4 — Evaluate the original AI reranker
python evaluation/evaluate_ai_reranker.py
Step 5 — Train AI-V2
python evaluation/train_ai_v2_reranker.py
Step 6 — Evaluate AI-V2 localization
python evaluation/evaluate_ai_v2_reranker.py
Step 7 — Analyze confidence
python evaluation/analyze_ai_confidence.py
Step 8 — Run the final benchmark
python evaluation/final_benchmark.py
Step 9 — Generate plots
python evaluation/generate_final_plots.py

# Technologies Used

Python
OpenCV
NumPy
Pandas
Scikit-learn
Matplotlib
Random Forest
Classical Computer Vision
Machine Learning

# Future Work

Future development can focus on:

Hard-negative mining
Learned visual embeddings
Siamese or contrastive neural networks
Transformer-based visual localization
Sub-pixel coordinate refinement
Confidence calibration
Larger and more diverse datasets
GPU acceleration
Real-time localization
Improved handling of visually ambiguous regions
Conclusion

Drift-Sense demonstrates an AI-assisted approach to visual localization by combining classical computer vision with relative candidate ranking.

The AI-V2 system reduced the mean localization error from:

13.691 px → 3.983 px

representing a:

70.91% reduction in mean localization error

The worst-case error was reduced from:

629.048 px → 71.732 px

representing an:

88.60% reduction in worst-case error

The results demonstrate that relative candidate ranking can improve localization robustness compared with selecting candidates using the classical baseline alone.

# Authors

Navadhikannan N
Mohammad Abdul Rahmam F

# Acknowledgement

This project was developed as part of an experimental study into robust visual localization using classical computer vision and machine learning-based candidate ranking.

# License

This project is intended for academic and research purposes.

