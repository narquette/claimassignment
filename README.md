# Claims Unpaid Prediction
This is the project code used for fake healthcare claims data.  The problem statement being addressed in the project is, "What is correlated with unpaid procedure claims?"

# Data Used for Analysis
1. Healthcare Claims Data

# Summary Report 

Please review the following [pdf](https://github.com/narquette/claimassignment/blob/master/project_documentation/Claims%20Assignment.pdf) to see reponses to the project questions and overall modeling results.

# Pre-requisites

Option 1 - WSL (Windows Sub-Linux)

1. Enable [WSL](https://winaero.com/blog/enable-wsl-windows-10-fall-creators-update/) in windows 
2. Install Ubuntu App from Windows Store
3. Create Login and sudo password for Linux

Option 2 - Docker Desktop

1. Install [docker desktop - windows](https://docs.docker.com/docker-for-windows/install/)

# Getting Started (WSL)

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

6. Run Claim_EDA.ipynb in the Code folder to perform an Exploratory Data Analysis

7. Run Model_Tuning_Evaluation.ipynb to perform model evaluation and tuning

# Getting Started (Docker)

1. Ensure Docker Desktop is running
2. Pull docker image
```cmd
   docker pull narquette/claims
```
3. Start up docker image
```cmd
  docker run -it --rm -p 8888:8888 narquette/claims 
```
4. Run Jupyter Notebook
```sh
  ./run_notebook.sh
```
5. Copy and paste url with tokens into browser
6. Navigate to Code / Claims EDA.ipynb
7. Run all cells

# Folder Overview

Code 
- Claim_EDA.ipynb (all of the code required to produce a final model)
- Model_Tuning_Evaluation.ipynb (perform tuning and evaluation on the models)
- machine_learning.py (contains the machine learning class needed to run in the Claim EDA Notebook)
- config.py (contains the information to write and get the best model parameters
- eda_functions.py (contains the information needed to run function in the EDA notebook)
- feature_importance.py (contains the code to be able to return the feature importance information for the model_tuning_evaluation notebook

Config
- feature_info.json (contain numerica features, categorical feature and dropped columns used in the pipeline steps)
- model json files (contains tuning information, and best result for each model)

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
- Assignment Response (Claim Assignment.pdf
