## Project description
This implementation of algorithm CBA (Classification Based on Associations) is based on the paper Integrating Classification and Association Rule Mining written by Bing Liu et al. The code was adapted from online references and we have improvised certain functions and reduced the time taken to generate rules. 

## Datasets description
We have chosen 5 datasets from UCI Machine Learning Repository and 2 more datasets from online resources. 
The datasets we have experimented on include: 
- iris
- glass
- pima 
- tic-tac-toe
- german
- social 
- spine 

Each dataset consists of 2 files: 
1. `*.data` contains many instances. Each line represents a sample. The attributes of the instance are divided by comma mark (,) without the space ( ). The last attribute is the class label, both number or literal string are vaild.
2. `*.names` contains two lines. The first line is title line, describing the name of each attribute. The last word must be **`class`**, representing the class label. The second line is the type of each attribute, only **`numerical`** and **`categorical`** are acceptable. The last word must be **`label`**. The words in both lines are all divided by comma mark (,) without space ( ).

Moreover, the names of two files must be the **same**.

You can open `iris.data` and `iris.names` under `datasets` directory to understand the rules above.

## How to run the code?
The entry of code is in the file `main.py`. You can use flags in the terminal to choose the dataset that you want to run.All datasets can be found in `datasets` directory. We have several options available to run the CBA-CB M2 code: 
1. With or without pruning
2. Single or multiple minsups used 

Without using flags, 
The default run would be using the iris dataset without pruning and a single minsup value. 

Using flags, 
Find the main.py file you want to run and type in terminal, for example "../GitHub/CZ4032_DAM_1/main.py", then add flags behind the code to run the different datasets and modes: 
    to change dataset: --filename dataset_name 
    to run with pruning: --prune True 
    to use multiple minsups: --multiple True 
    
For example, "../GitHub/CZ4032_DAM_1/main.py" --filename social --prune True --multiple True 
*note: take note of the formatting and spaces for the commands in terminal
This should run the program successfully with your chosen options 

## Python files 
- car.py: define Class Association Rules (CAR) 
- cba_cb_m2.py: CBA classifier builder M2 
- cba_rg.py: CBA rule generator 
- classifier_m2.py: definition of classifier formed in CBA-CB M2
- CrossValM2.py: cross validation for classifier
- dataset.py: define dataset class 
- frequentRuleItems.py: define frequentRuleItems class 
- main.py: **** run this file ****
- other_classifiers.py: other classifiers to compare with the CBA-CB 
- pre_processing.py: pre-processing for the data
- prune.py: to find the pruning rules for the classifier
- read.py: to read the dataset 
- rule.py: define a rule 
- ruleitem.py: define a ruleitem


## Reference
Liu, Bing, W. Hsu, and Y. Ma. "Integrating Classification and Association Rule Mining." Proc of Kdd (1998):80--86.

## Project Group Members
Chua Ziheng
Mun Kei Wuai 
Pooja Srinivas Nag 
Tan Wen Xiu 