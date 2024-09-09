# Welcome to TFDA!
# This the the repo for Transient Fault Detection algorithm. <br />
This repo is currently used to share and update codes for TFDA. <br />
New algorithms developed in the future will also be uploaded and updated here. <br />
# Installing packages
NOTE: The installation instruction below assume that you have python installed on your machine and are using conda as your package/environment manager.
1. Create a new environment: conda create -n oedi python=3.8  
2. Activate environment: conda activate oedi  
3. Install packages listed in requirements.txt by running the following lines: <br />
   conda install --yes --file requirements.txt <br />
   pip install pip install comtrade <br />
   pip install pip install -U scikit-learn <br />
   pip install scikit-plot <br />
# Code scipts
Auxiliary functions are in functions.py <br />
fft_3faults.py loads the data in folder Faults and train the model to detect non-fault condidtion or three different fault conditions. <br />
Example of training results -- confusion matrix.<br>
   1. 100% accuracy on non-fault conditions.
   2. 99% accuracy on type 1 fault condition.
   3. 94% accuracy on type 2 fault condition.
   4. 100% accuracy on type 3 fault condition.
   <img src="/images/example1.png" width="300" height="300" alt="Alt text">

Example of training results -- ROC curve.<br>
   <img src="/images/example2.png" width="400" height="300" alt="Alt text">
