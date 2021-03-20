# wegl-gbt
Stanford CS 224W (Winter 2021) Project

Authors: Sharan Ramjee (sramjee@stanford.edu) and Michael Wornow (mwornow@stanford.edu)

## Steps to reproduce results
- Clone the repository
- Download the data from [here](https://drive.google.com/drive/folders/11tNGqTrKg4hg966IcLpYnpyqp1rbTkDo?usp=sharing) and extract it in the data folder
- Run the following command
'''
$ python models.py
'''

The command uses the saved embeddings obtained using the Linear WEGL model, applied SMOTE on it, trains a HGBT model, and runs an OGB evaluator to evaluate the model
