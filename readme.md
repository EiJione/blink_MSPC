# Video Analysis and Multivariate Statistical Process Control (MSPC) Project

This project involves processing video data to extract features related to eye aspect ratio (EAR), which are then used to perform MSPC analysis. The objective is to identify patterns and anomalies in video data, particularly focusing on behaviors like blinking which may indicate fatigue or other physiological states.

## Overview

The project is divided into two main parts:

1. **Video Analysis**: Process video files to extract EAR-related features.
2. **MSPC Calculation**: Use the extracted features from the videos to compute MSPC metrics such as the \( T^2 \) and \( Q \) statistics.

## Requirements

- Python 3.8+
- Libraries: `numpy`, `pandas`, `sklearn`, `os`
- Video files in MP4 format.

## Setup

To set up the project, clone the repository and install the required Python packages:

```bash
git clone <repository-url>
cd <project-directory>
pip install numpy pandas scikit-learn
