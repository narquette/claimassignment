# Claims Unpaid Prediction
This is the project code used for fake healthcare claims data.  The problem statement being addressed in the project is, "What is correlated with unpaid procedure claims?"

# Data Used for Analysis
1. Healthcare Claims Data

# Pre-requisites

WSL (Windows Sub-Linux)

1. Enable [WSL](https://winaero.com/blog/enable-wsl-windows-10-fall-creators-update/) in windows 
2. Install Ubuntu App from Windows Store
3. Create Login and sudo password for Linux

# Getting Started 

1. Open Windows Sub Linux (Ubuntu App)

2. Run the following command

```sh
git clone https://github.com/narquette/claimsassignment
```

3. Change install script to executable and run install file (you may be prompted to enter the sudo password for installing java)

```sh
cd ~/claimsassignment
chmod +x prereq_install.sh
./prereq_install.sh
conda activate claims
```

4. Open Jupyter Notebook

```sh
jupyter notebook --no-browser
```

5. [Copy URL from command line](https://www.screencast.com/t/JgVmAL6wC)

6. Run Claim_EDA.ipynb in the Code folder

# Folder Overview

Code 
- Claim_EDA.ipynb (all of the code required to produce a final model)
- Final_Model_Predictions.ipynb (perform a prediction using original data to compare results)
- helperFile.py (contains the machine learning class needed to run in the Claim EDA Notebook

Data
- original (original claims data, original procedure data (sub set of claims data)
- cleaned (cleaned procedured data)
- train_test_splite (split data needed to run h2o)
- prediction (storing prediction results from Final Model Prediction notebook)

Logs
- Previous Model Logs and Where New Logs Information will be placed

Models
- The final models produced from running the notebook

Visualizations 
- Visualizations produced in the EDA (exploratory data analysis phase)
- Pandas Profile HTML file for the original data set

Project_Documentation
- DS Challenge Document (DS_Challenge_Questions.docx)
- Assignment Response (Claim Assignment.pptx)
