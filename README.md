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
- gender

Datasets can be found in `datasets/`

Each dataset consists of 2 files:
1. `*.data` contains many instances. Each line represents a sample. The attributes of the instance are divided by comma mark (,) without the space ( ). The last attribute is the class label, both number or literal string are vaild.
2. `*.names` contains two lines. The first line is title line, describing the name of each attribute. The last word must be **`class`**, representing the class label. The second line is the type of each attribute, only **`numerical`** and **`categorical`** are acceptable. The last word must be **`label`**. The words in both lines are all divided by comma mark (,) without space ( ).

Moreover, the names of two files must be the **same**.

You can open `iris.data` and `iris.names` under `datasets` directory to understand the rules above.

## How to run the code?

### Driver Code
The entry of code is in the file `main.py`. You can use flags in the terminal to choose the dataset that you want to run.All datasets can be found in `datasets` directory. We have several options available to run the CBA-CB M2 code:
1. With or without pruning
2. Single or multiple minsups used

Without using flags,
The default run would be using the iris dataset without pruning and a single minsup value.

Using flags,
Run the `./main.py` from the project root directory, then add flags behind the code to run the different datasets and modes:
- to change dataset: `--filename DATASET_NAME`
- to run with pruning: `--prune`
- to use multiple minsups: `--multiple`
- Set initial minsup `--minsup 0.01`
- Set minconf `--minconf 0.5`

E.g. Running iris dataset with pruning and multiple minsups

`./main.py --filename iris --prune --multiple`

### Other classifications

We handpicked 4 other classifiers, namely: Decision Tree, Random Forest, SVM and Naive Bayes and benchmarked their performances using accuracy and F1 Score.

`./other_classifiers.py --filename iris `


## Python files
### Driver files
- main.py: **** run this file ****
- other_classifiers.py: other classifiers to compare with the CBA-CB

### CBA logic
- cba_rg.py: CBA rule generator
- cba_cb_m2.py: CBA classifier builder M2


### Utility files
Inside `utils/`
- pre_processing.py: pre-processing for the data
- car.py: define Class Association Rules (CAR)
- dataset.py: define dataset class
- frequentRuleItems.py: define frequentRuleItems class
- prune.py: to find the pruning rules for the classifier
- read.py: to read the dataset
- rule.py: define a rule
- ruleitem.py: define a ruleitem
- classifier_m2.py: definition of classifier formed in CBA-CB M2
- CrossValM2.py: cross validation for classifier


## Reference
Liu, Bing, W. Hsu, and Y. Ma. "Integrating Classification and Association Rule Mining." Proc of Kdd (1998):80--86.

## Project Group Members
Chua Ziheng
Mun Kei Wuai
Pooja Srinivas Nag
Tan Wen Xiu
