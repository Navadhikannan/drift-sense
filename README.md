# \# Drift-Sense

# 

# \## AI-Assisted Visual Drift Localization

# 

# Drift-Sense is a computer-vision and machine-learning pipeline for locating a reference visual pattern inside a larger search image.

# 

# The project starts with classical template matching and progressively improves localization using structural verification, spatial candidate analysis, and an AI-based relative candidate reranker.

# 

# \---

# 

# \## Project Objective

# 

# The objective of Drift-Sense is to estimate the `(X, Y)` position of a reference object or visual pattern inside a search image while remaining robust to image noise and ambiguous template matches.

# 

# The system is designed to address a key limitation of conventional template matching:

# 

# > A high template-matching score does not always correspond to the correct physical location.

# 

# Drift-Sense therefore generates multiple candidate locations and uses additional structural and relative features to select the most reliable candidate.

# 

# \---

# 

# \## System Pipeline

# 

# Reference Image

# &#x20;       |

# &#x20;       v

# Template Matching

# &#x20;       |

# &#x20;       v

# Candidate Generation

# &#x20;       |

# &#x20;       v

# Feature Extraction

# &#x20;       |

# &#x20;       +----------------------+

# &#x20;       |                      |

# &#x20;       v                      v

# Classical Scores       Spatial Features

# &#x20;       |                      |

# &#x20;       +----------+-----------+

# &#x20;                  |

# &#x20;                  v

# &#x20;         AI-V2 Candidate

# &#x20;            Reranker

# &#x20;                  |

# &#x20;                  v

# &#x20;         Confidence Analysis

# &#x20;                  |

# &#x20;                  v

# &#x20;         Final Localization

# &#x20;                  |

# &#x20;                  v

# &#x20;             (X, Y)

# 

# 

# \---

# 

# \## Development Stages

# 

# \### V5.1 - Confidence-Aware Matching

# 

# The first improved classical localization stage combines:

# 

# \- Template matching

# \- Edge representation

# \- Gradient representation

# \- Structural verification

# \- Candidate confidence

# \- Conservative candidate selection

# 

# \---

# 

# \### V5.2 - Spatially-Aware Reranking

# 

# V5.2 introduces spatial neighborhood information.

# 

# Candidate locations are evaluated using:

# 

# \- Original template score

# \- Neighboring candidate response

# \- Spatial distribution of candidate matches

# 

# \---

# 

# \### V5.3 - Multi-Scale Contextual Verification

# 

# V5.3 introduces additional contextual verification using:

# 

# \- Local similarity

# \- Context similarity

# \- Regional similarity

# \- Multi-scale information

# 

# This stage demonstrated the limitations of purely handcrafted verification and motivated the transition toward machine learning.

# 

# \---

# 

# \# AI Localization

# 

# \## AI-V1

# 

# The first AI system uses a Random Forest classifier to evaluate candidate locations.

# 

# Candidate features include:

# 

# \- Template score

# \- Gray score

# \- Edge score

# \- Gradient score

# \- Structural score

# \- Candidate rank

# \- Candidate coordinates

# \- Center distance

# 

# The model learns whether an individual candidate is likely to correspond to the ground-truth location.

# 

# \---

# 

# \# AI-V2

# 

# The final AI approach uses \*\*relative candidate ranking\*\*.

# 

# Instead of evaluating a candidate independently, AI-V2 compares each candidate with the other candidates generated for the same image.

# 

# \### AI-V2 features

# 

# \- `template\_gap`

# \- `gray\_gap`

# \- `edge\_gap`

# \- `gradient\_gap`

# \- `structural\_gap`

# \- `combined\_score`

# \- `combined\_gap`

# \- `normalized\_rank`

# \- `distance\_from\_template\_best`

# \- `nearest\_candidate\_distance`

# \- `neighborhood\_density\_50`

# \- `neighborhood\_density\_100`

# 

# This allows the model to learn relationships between competing candidates rather than relying only on absolute similarity scores.

# 

# \---

# 

# \# Dataset

# 

# The robustness evaluation contains:

# 

# \- 30 image samples

# \- 4 noise conditions

# \- Clean

# \- Low noise

# \- Medium noise

# \- High noise

# 

# Total evaluation cases:

# 

# \*\*120\*\*

# 

# For AI candidate training:

# 

# \- 3,600 candidate rows

# \- 112 positive candidates

# \- 3,488 negative candidates

# 

# The large robustness image dataset is intentionally excluded from Git using:

# 

# ```text

# data/robustness/

