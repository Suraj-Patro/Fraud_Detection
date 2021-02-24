*This page is available as an executable or viewable **Jupyter Notebook**:* 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](href="https://colab.research.google.com/github/Suraj-Patro/Fraud_Detection/blob/main/Fraud_Detection.ipynb)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Suraj-Patro/Fraud_Detection/blob/main/Fraud_Detection.ipynb)
[![mybinder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Suraj-Patro/Fraud_Detection/main?filepath=Fraud_Detection.ipynb)

# **Fraud Detection**

A typical organization loses an estimated 5% of its yearly revenue to fraud. In this course, learn to fight fraud by using data. Apply supervised learning algorithms to detect fraudulent behavior based upon past fraud, and use unsupervised learning methods to discover new types of fraud activities. 

Fraudulent transactions are rare compared to the norm.  As such, learn to properly classify imbalanced datasets.

The Project provides technical and theoretical insights and demonstrates how to implement fraud detection models. Finally, get tips and advice from real-life experience to help prevent common mistakes in fraud analytics.

**Imports**


```python
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
```


```python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pprint import pprint as pp
import csv
from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import homogeneity_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN
import seaborn as sns
from itertools import product
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

!pip install pyldavis
import pyLDAvis.gensim
```

    Collecting pyldavis
    [?25l  Downloading https://files.pythonhosted.org/packages/a5/3a/af82e070a8a96e13217c8f362f9a73e82d61ac8fff3a2561946a97f96266/pyLDAvis-2.1.2.tar.gz (1.6MB)
    [K     |████████████████████████████████| 1.6MB 6.4MB/s 
    [?25hRequirement already satisfied: wheel>=0.23.0 in /usr/local/lib/python3.6/dist-packages (from pyldavis) (0.36.2)
    Requirement already satisfied: numpy>=1.9.2 in /usr/local/lib/python3.6/dist-packages (from pyldavis) (1.19.5)
    Requirement already satisfied: scipy>=0.18.0 in /usr/local/lib/python3.6/dist-packages (from pyldavis) (1.4.1)
    Requirement already satisfied: pandas>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from pyldavis) (1.1.5)
    Requirement already satisfied: joblib>=0.8.4 in /usr/local/lib/python3.6/dist-packages (from pyldavis) (1.0.0)
    Requirement already satisfied: jinja2>=2.7.2 in /usr/local/lib/python3.6/dist-packages (from pyldavis) (2.11.2)
    Requirement already satisfied: numexpr in /usr/local/lib/python3.6/dist-packages (from pyldavis) (2.7.2)
    Requirement already satisfied: pytest in /usr/local/lib/python3.6/dist-packages (from pyldavis) (3.6.4)
    Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from pyldavis) (0.16.0)
    Collecting funcy
      Downloading https://files.pythonhosted.org/packages/66/89/479de0afbbfb98d1c4b887936808764627300208bb771fcd823403645a36/funcy-1.15-py2.py3-none-any.whl
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.17.0->pyldavis) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.17.0->pyldavis) (2018.9)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from jinja2>=2.7.2->pyldavis) (1.1.1)
    Requirement already satisfied: more-itertools>=4.0.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyldavis) (8.6.0)
    Requirement already satisfied: atomicwrites>=1.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyldavis) (1.4.0)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyldavis) (1.15.0)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyldavis) (20.3.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from pytest->pyldavis) (51.1.1)
    Requirement already satisfied: pluggy<0.8,>=0.5 in /usr/local/lib/python3.6/dist-packages (from pytest->pyldavis) (0.7.1)
    Requirement already satisfied: py>=1.5.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyldavis) (1.10.0)
    Building wheels for collected packages: pyldavis
      Building wheel for pyldavis (setup.py) ... [?25l[?25hdone
      Created wheel for pyldavis: filename=pyLDAvis-2.1.2-py2.py3-none-any.whl size=97712 sha256=a9bf9c1c8d8ccc3ff08e3fe826648095a26b6f718e16871953a163707c56b15a
      Stored in directory: /root/.cache/pip/wheels/98/71/24/513a99e58bb6b8465bae4d2d5e9dba8f0bef8179e3051ac414
    Successfully built pyldavis
    Installing collected packages: funcy, pyldavis
    Successfully installed funcy-1.15 pyldavis-2.1.2


**Pandas Configuration Options**


```python
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 300)
pd.set_option('display.expand_frame_repr', True)
```

**Data File Objects**


```python
!wget https://assets.datacamp.com/production/repositories/2162/datasets/cc3a36b722c0806e4a7df2634e345975a0724958/chapter_1.zip
!wget https://assets.datacamp.com/production/repositories/2162/datasets/4fb6199be9b89626dcd6b36c235cbf60cf4c1631/chapter_2.zip
!wget https://assets.datacamp.com/production/repositories/2162/datasets/08cfcd4158b3a758e72e9bd077a9e44fec9f773b/chapter_3.zip
!wget https://assets.datacamp.com/production/repositories/2162/datasets/94f2356652dc9ea8f0654b5e9c29645115b6e77f/chapter_4.zip

!unzip chapter_1.zip
!unzip chapter_2.zip
!unzip chapter_3.zip
!unzip chapter_4.zip
```

    --2021-01-18 09:13:41--  https://assets.datacamp.com/production/repositories/2162/datasets/cc3a36b722c0806e4a7df2634e345975a0724958/chapter_1.zip
    Resolving assets.datacamp.com (assets.datacamp.com)... 104.18.17.147, 104.18.16.147
    Connecting to assets.datacamp.com (assets.datacamp.com)|104.18.17.147|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3301209 (3.1M) [application/zip]
    Saving to: ‘chapter_1.zip’
    
    chapter_1.zip       100%[===================>]   3.15M  17.1MB/s    in 0.2s    
    
    2021-01-18 09:13:41 (17.1 MB/s) - ‘chapter_1.zip’ saved [3301209/3301209]
    
    --2021-01-18 09:13:41--  https://assets.datacamp.com/production/repositories/2162/datasets/4fb6199be9b89626dcd6b36c235cbf60cf4c1631/chapter_2.zip
    Resolving assets.datacamp.com (assets.datacamp.com)... 104.18.17.147, 104.18.16.147
    Connecting to assets.datacamp.com (assets.datacamp.com)|104.18.17.147|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1839230 (1.8M) [application/zip]
    Saving to: ‘chapter_2.zip’
    
    chapter_2.zip       100%[===================>]   1.75M  10.5MB/s    in 0.2s    
    
    2021-01-18 09:13:42 (10.5 MB/s) - ‘chapter_2.zip’ saved [1839230/1839230]
    
    --2021-01-18 09:13:42--  https://assets.datacamp.com/production/repositories/2162/datasets/08cfcd4158b3a758e72e9bd077a9e44fec9f773b/chapter_3.zip
    Resolving assets.datacamp.com (assets.datacamp.com)... 104.18.17.147, 104.18.16.147
    Connecting to assets.datacamp.com (assets.datacamp.com)|104.18.17.147|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 300387 (293K) [application/zip]
    Saving to: ‘chapter_3.zip’
    
    chapter_3.zip       100%[===================>] 293.35K  --.-KB/s    in 0.1s    
    
    2021-01-18 09:13:42 (2.89 MB/s) - ‘chapter_3.zip’ saved [300387/300387]
    
    --2021-01-18 09:13:42--  https://assets.datacamp.com/production/repositories/2162/datasets/94f2356652dc9ea8f0654b5e9c29645115b6e77f/chapter_4.zip
    Resolving assets.datacamp.com (assets.datacamp.com)... 104.18.17.147, 104.18.16.147
    Connecting to assets.datacamp.com (assets.datacamp.com)|104.18.17.147|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2825830 (2.7M) [application/zip]
    Saving to: ‘chapter_4.zip’
    
    chapter_4.zip       100%[===================>]   2.69M  15.1MB/s    in 0.2s    
    
    2021-01-18 09:13:43 (15.1 MB/s) - ‘chapter_4.zip’ saved [2825830/2825830]
    
    Archive:  chapter_1.zip
       creating: chapter_1/
      inflating: chapter_1/creditcard_sampledata_3.csv  
      inflating: chapter_1/creditcard_sampledata.csv  
    Archive:  chapter_2.zip
       creating: chapter_2/
      inflating: chapter_2/creditcard_sampledata_2.csv  
    Archive:  chapter_3.zip
       creating: chapter_3/
      inflating: chapter_3/x_scawed_full.pickle  
      inflating: chapter_3/db_full.pickle  
      inflating: chapter_3/banksim.csv   
      inflating: chapter_3/x_scaled.pickle  
      inflating: chapter_3/banksim_adj.csv  
      inflating: chapter_3/labels_full.pickle  
      inflating: chapter_3/labels.pickle  
    Archive:  chapter_4.zip
       creating: chapter_4/
     extracting: chapter_4/cleantext.pickle  
     extracting: chapter_4/ldamodel.pickle  
      inflating: chapter_4/dict.pickle   
      inflating: chapter_4/corpus.pickle  
      inflating: chapter_4/enron_emails_clean.csv  



```python
#data = Path.cwd() / 'data' / 'fraud_detection'
data = Path.cwd()

ch1 = data / 'chapter_1'
cc1_file = ch1 / 'creditcard_sampledata.csv'
cc3_file = ch1 / 'creditcard_sampledata_3.csv'

ch2 = data / 'chapter_2'
cc2_file = ch2 / 'creditcard_sampledata_2.csv'

ch3 = data / 'chapter_3'
banksim_file = ch3 / 'banksim.csv'
banksim_adj_file = ch3 / 'banksim_adj.csv'
db_full_file = ch3 / 'db_full.pickle'
labels_file = ch3 / 'labels.pickle'
labels_full_file = ch3 / 'labels_full.pickle'
x_scaled_file = ch3 / 'x_scaled.pickle'
x_scaled_full_file = ch3 / 'x_scaled_full.pickle'

ch4 = data / 'chapter_4'
enron_emails_clean_file = ch4 / 'enron_emails_clean.csv'
cleantext_file = ch4 / 'cleantext.pickle'
corpus_file = ch4 / 'corpus.pickle'
dict_file = ch4 / 'dict.pickle'
ldamodel_file = ch4 / 'ldamodel.pickle'
```

# Introduction and preparing your data

Describe about the typical challenges associated with fraud detection. Describe how to resample data in a smart way, and tackle problems with imbalanced data.

## Introduction to fraud detection

* Types:
    * Insurance
    * Credit card
    * Identity theft
    * Money laundering
    * Tax evasion
    * Healthcare
    * Product warranty
* e-commerce businesses must continuously assess the legitimacy of client transactions
* Detecting fraud is challenging:
    * Uncommon; < 0.01% of transactions
    * Attempts are made to conceal fraud
    * Behavior evolves
    * Fraudulent activities perpetrated by networks - organized crime
* Fraud detection requires training an algorithm to identify concealed observations from any normal observations
* Fraud analytics teams:
    * Often use rules based systems, based on manually set thresholds and experience
    * Check the news
    * Receive external lists of fraudulent accounts and names
        * suspicious names or track an external hit list from police to reference check against the client base
    * Sometimes use machine learning algorithms to detect fraud or suspicious behavior
        * Existing sources can be used as inputs into the ML model
        * Verify the veracity of rules based labels

### Checking the fraud to non-fraud ratio

In this chapter, you will work on `creditcard_sampledata.csv`, a dataset containing credit card transactions data. Fraud occurrences are fortunately an **extreme minority** in these transactions.

However, Machine Learning algorithms usually work best when the different classes contained in the dataset are more or less equally present. If there are few cases of fraud, then there's little data to learn how to identify them. This is known as **class imbalance**, and it's one of the main challenges of fraud detection.

Let's explore this dataset, and observe this class imbalance problem.

**Explanation**

* `import pandas as pd`, read the credit card data in and assign it to `df`. This has been done for you.
* Use `.info()` to print information about `df`.
* Use `.value_counts()` to get the count of fraudulent and non-fraudulent transactions in the `'Class'` column. Assign the result to `occ`.
* Get the ratio of fraudulent transactions over the total number of transactions in the dataset.


```python
df = pd.read_csv(cc3_file)
```

#### Explore the features available in your dataframe


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5050 entries, 0 to 5049
    Data columns (total 31 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Unnamed: 0  5050 non-null   int64  
     1   V1          5050 non-null   float64
     2   V2          5050 non-null   float64
     3   V3          5050 non-null   float64
     4   V4          5050 non-null   float64
     5   V5          5050 non-null   float64
     6   V6          5050 non-null   float64
     7   V7          5050 non-null   float64
     8   V8          5050 non-null   float64
     9   V9          5050 non-null   float64
     10  V10         5050 non-null   float64
     11  V11         5050 non-null   float64
     12  V12         5050 non-null   float64
     13  V13         5050 non-null   float64
     14  V14         5050 non-null   float64
     15  V15         5050 non-null   float64
     16  V16         5050 non-null   float64
     17  V17         5050 non-null   float64
     18  V18         5050 non-null   float64
     19  V19         5050 non-null   float64
     20  V20         5050 non-null   float64
     21  V21         5050 non-null   float64
     22  V22         5050 non-null   float64
     23  V23         5050 non-null   float64
     24  V24         5050 non-null   float64
     25  V25         5050 non-null   float64
     26  V26         5050 non-null   float64
     27  V27         5050 non-null   float64
     28  V28         5050 non-null   float64
     29  Amount      5050 non-null   float64
     30  Class       5050 non-null   int64  
    dtypes: float64(29), int64(2)
    memory usage: 1.2 MB



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>258647</td>
      <td>1.725265</td>
      <td>-1.337256</td>
      <td>-1.012687</td>
      <td>-0.361656</td>
      <td>-1.431611</td>
      <td>-1.098681</td>
      <td>-0.842274</td>
      <td>-0.026594</td>
      <td>-0.032409</td>
      <td>0.215113</td>
      <td>1.618952</td>
      <td>-0.654046</td>
      <td>-1.442665</td>
      <td>-1.546538</td>
      <td>-0.230008</td>
      <td>1.785539</td>
      <td>1.419793</td>
      <td>0.071666</td>
      <td>0.233031</td>
      <td>0.275911</td>
      <td>0.414524</td>
      <td>0.793434</td>
      <td>0.028887</td>
      <td>0.419421</td>
      <td>-0.367529</td>
      <td>-0.155634</td>
      <td>-0.015768</td>
      <td>0.010790</td>
      <td>189.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>69263</td>
      <td>0.683254</td>
      <td>-1.681875</td>
      <td>0.533349</td>
      <td>-0.326064</td>
      <td>-1.455603</td>
      <td>0.101832</td>
      <td>-0.520590</td>
      <td>0.114036</td>
      <td>-0.601760</td>
      <td>0.444011</td>
      <td>1.521570</td>
      <td>0.499202</td>
      <td>-0.127849</td>
      <td>-0.237253</td>
      <td>-0.752351</td>
      <td>0.667190</td>
      <td>0.724785</td>
      <td>-1.736615</td>
      <td>0.702088</td>
      <td>0.638186</td>
      <td>0.116898</td>
      <td>-0.304605</td>
      <td>-0.125547</td>
      <td>0.244848</td>
      <td>0.069163</td>
      <td>-0.460712</td>
      <td>-0.017068</td>
      <td>0.063542</td>
      <td>315.17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>96552</td>
      <td>1.067973</td>
      <td>-0.656667</td>
      <td>1.029738</td>
      <td>0.253899</td>
      <td>-1.172715</td>
      <td>0.073232</td>
      <td>-0.745771</td>
      <td>0.249803</td>
      <td>1.383057</td>
      <td>-0.483771</td>
      <td>-0.782780</td>
      <td>0.005242</td>
      <td>-1.273288</td>
      <td>-0.269260</td>
      <td>0.091287</td>
      <td>-0.347973</td>
      <td>0.495328</td>
      <td>-0.925949</td>
      <td>0.099138</td>
      <td>-0.083859</td>
      <td>-0.189315</td>
      <td>-0.426743</td>
      <td>0.079539</td>
      <td>0.129692</td>
      <td>0.002778</td>
      <td>0.970498</td>
      <td>-0.035056</td>
      <td>0.017313</td>
      <td>59.98</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>281898</td>
      <td>0.119513</td>
      <td>0.729275</td>
      <td>-1.678879</td>
      <td>-1.551408</td>
      <td>3.128914</td>
      <td>3.210632</td>
      <td>0.356276</td>
      <td>0.920374</td>
      <td>-0.160589</td>
      <td>-0.801748</td>
      <td>0.137341</td>
      <td>-0.156740</td>
      <td>-0.429388</td>
      <td>-0.752392</td>
      <td>0.155272</td>
      <td>0.215068</td>
      <td>0.352222</td>
      <td>-0.376168</td>
      <td>-0.398920</td>
      <td>0.043715</td>
      <td>-0.335825</td>
      <td>-0.906171</td>
      <td>0.108350</td>
      <td>0.593062</td>
      <td>-0.424303</td>
      <td>0.164201</td>
      <td>0.245881</td>
      <td>0.071029</td>
      <td>0.89</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>86917</td>
      <td>1.271253</td>
      <td>0.275694</td>
      <td>0.159568</td>
      <td>1.003096</td>
      <td>-0.128535</td>
      <td>-0.608730</td>
      <td>0.088777</td>
      <td>-0.145336</td>
      <td>0.156047</td>
      <td>0.022707</td>
      <td>-0.963306</td>
      <td>-0.228074</td>
      <td>-0.324933</td>
      <td>0.390609</td>
      <td>1.065923</td>
      <td>0.285930</td>
      <td>-0.627072</td>
      <td>0.170175</td>
      <td>-0.215912</td>
      <td>-0.147394</td>
      <td>0.031958</td>
      <td>0.123503</td>
      <td>-0.174528</td>
      <td>-0.147535</td>
      <td>0.735909</td>
      <td>-0.262270</td>
      <td>0.015577</td>
      <td>0.015955</td>
      <td>6.53</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Count the occurrences of fraud and no fraud and print them
occ = df['Class'].value_counts()
occ
```




    0    5000
    1      50
    Name: Class, dtype: int64




```python
# Print the ratio of fraud cases
ratio_cases = occ/len(df.index)
print(f'Ratio of fraudulent cases: {ratio_cases[1]}\nRatio of non-fraudulent cases: {ratio_cases[0]}')
```

    Ratio of fraudulent cases: 0.009900990099009901
    Ratio of non-fraudulent cases: 0.9900990099009901


**The ratio of fraudulent transactions is very low. This is a case of class imbalance problem, and we're going to describe how to deal with this in the next code.**

### Data visualization

From the previous exercise we know that the ratio of fraud to non-fraud observations is very low. You can do something about that, for example by **re-sampling** our data, which is explained in the next video.

In this exercise, you'll look at the data and **visualize the fraud to non-fraud ratio**. It is always a good starting point in your fraud analysis, to look at your data first, before you make any changes to it.

Moreover, when talking to your colleagues, a picture often makes it very clear that we're dealing with heavily imbalanced data. Let's create a plot to visualize the ratio fraud to non-fraud data points on the dataset `df`.

The function `prep_data()` is already loaded in your workspace, as well as `matplotlib.pyplot as plt`.

**Explanation**

* Define the `plot_data(X, y)` function, that will nicely plot the given feature set `X` with labels `y` in a scatter plot. This has been done for you.
* Use the function `prep_data()` on your dataset `df` to create feature set `X` and labels `y`.
* Run the function `plot_data()` on your newly obtained `X` and `y` to visualize your results.

#### def prep_data


```python
def prep_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Convert the DataFrame into two variable
    X: data columns (V1 - V28)
    y: lable column
    """
    X = df.iloc[:, 2:30].values
    y = df.Class.values
    return X, y
```

#### def plot_data


```python
# Define a function to create a scatter plot of our data and labels
def plot_data(X: np.ndarray, y: np.ndarray):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()
```


```python
# Create X and y from the prep_data function 
X, y = prep_data(df)
```


```python
# Plot our data by running our plot data function on X and y
plot_data(X, y)
```


![png](output_26_0.png)


**By visualizing the data, we can immediately see how our fraud cases are scattered over our data, and how few cases we have. A picture often makes the imbalance problem clear. In the next code snippets we'll visually explore how to improve our fraud to non-fraud balance.**

#### Reproduced using the DataFrame


```python
plt.scatter(df.V2[df.Class == 0], df.V3[df.Class == 0], label="Class #0", alpha=0.5, linewidth=0.15)
plt.scatter(df.V2[df.Class == 1], df.V3[df.Class == 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
plt.legend()
plt.show()
```


![png](output_29_0.png)


## Increase successful detections with data resampling

* resampling can help model performance in cases of imbalanced data sets

#### Undersampling

* ![undersampling](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/undersampling.JPG)
* Undersampling the majority class (non-fraud cases)
    * Straightforward method to adjust imbalanced data
    * Take random draws from the non-fraud observations, to match the occurences of fraud observations (as shown in the picture)

#### Oversampling

* ![oversampling](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/oversampling.JPG)
* Oversampling the minority class (fraud cases)
    * Take random draws from the fraud cases and copy those observations to increase the amount of fraud samples
* Both methods lead to having a balance between fraud and non-fraud cases
* Drawbacks
    * with random undersampling, a lot of information is thrown away
    * with oversampling, the model will be trained on a lot of duplicates

#### Implement resampling methods using Python imblean module

* compatible with scikit-learn

```python
from imblearn.over_sampling import RandomOverSampler

method = RandomOverSampler()
X_resampled, y_resampled =  method.fit_sample(X, y)

compare_plots(X_resampled, y_resampled, X, y)
```

![oversampling plot](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/oversampling_plot.JPG)
* The darker blue points reflect there are more identical data

#### SMOTE

* ![smote](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/smote.JPG)
* Synthetic minority Oversampling Technique (SMOTE)
    * [Resampling strategies for Imbalanced Data Sets](https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets)
    * Another way of adjusting the imbalance by oversampling minority observations
    * SMOTE uses characteristics of nearest neighbors of fraud cases to create new synthetic fraud cases
        * avoids duplicating observations

#### Determining the best resampling method is situational

* Random Undersampling (RUS):
    * If there is a lot of data and many minority cases, then undersampling may be computationally more convenient
        * In most cases, throwing away data is not desirable
* Random Oversampling (ROS):
    * Straightforward
    * Training the model on many duplicates
* SMOTE:
    * more sophisticated
    * realistic data set
    * training on synthetic data
    * only works well if the minority case features are similar
        * **if fraud is spread through the data and not distinct, using nearest neighbors to create more fraud cases, introduces noise into the data, as the nearest neighbors might not be fraud cases**

#### When to use resmapling methods

* Use resampling methods on the training set, not on the test set
* The goal is to produce a better model by providing balanced data
    * The goal is not to predict the synthetic samples
* Test data should be free of duplicates and synthetic data
* Only test the model on real data
    * First, spit the data into train and test sets
    
```python
# Define resampling method and split into train and test
method = SMOTE(kind='borderline1')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# Apply resampling to the training data only
X_resampled, y_resampled = method.fit_sample(X_train, y_train)

# Continue fitting the model and obtain predictions
model = LogisticRegression()
model.fit(X_resampled, y_resampled)

# Get model performance metrics
predicted = model.predict(X_test)
print(classification_report(y_test, predicted))
```

### Resampling methods for imbalanced data

Which of these methods takes a random subsample of your majority class to account for class "imbalancedness"?

**Possible Answers**

* ~~Random Over Sampling (ROS)~~
* **Random Under Sampling (RUS)**
* ~~Synthetic Minority Over-sampling Technique (SMOTE)~~
* ~~None of the above~~

**By using ROS and SMOTE you add more examples to the minority class. RUS adjusts the balance of your data by reducing the majority class.**

### Applying Synthetic Minority Oversampling Technique (SMOTE)

In this exercise, you're going to re-balance our data using the **Synthetic Minority Over-sampling Technique** (SMOTE). Unlike ROS, SMOTE does not create exact copies of observations, but **creates new, synthetic, samples** that are quite similar to the existing observations in the minority class. SMOTE is therefore slightly more sophisticated than just copying observations, so let's apply SMOTE to our credit card data. The dataset `df` is available and the packages you need for SMOTE are imported. In the following exercise, you'll visualize the result and compare it to the original data, such that you can see the effect of applying SMOTE very clearly.

**Explanation**

* Use the `prep_data` function on `df` to create features `X` and labels `y`.
* Define the resampling method as SMOTE of the regular kind, under the variable `method`.
* Use `.fit_sample()` on the original `X` and `y` to obtain newly resampled data.
* Plot the resampled data using the `plot_data()` function.


```python
# Run the prep_data function
X, y = prep_data(df)
```


```python
print(f'X shape: {X.shape}\ny shape: {y.shape}')
```

    X shape: (5050, 28)
    y shape: (5050,)



```python
# Define the resampling method
method = SMOTE()
```


```python
# Create the resampled feature set
X_resampled, y_resampled = method.fit_sample(X, y)
```


```python
# Plot the resampled data
plot_data(X_resampled, y_resampled)
```


![png](output_43_0.png)


**The minority class is now much more prominently visible in our data. To see the results of SMOTE even better, we'll compare it to the original data in the next exercise.**

### Compare SMOTE to original data

In the last exercise, you saw that using SMOTE suddenly gives us more observations of the minority class. Let's compare those results to our original data, to get a good feeling for what has actually happened. Let's have a look at the value counts again of our old and new data, and let's plot the two scatter plots of the data side by side. You'll use the function compare_plot() for that that, which takes the following arguments: `X`, `y`, `X_resampled`, `y_resampled`, `method=''`. The function plots your original data in a scatter plot, along with the resampled side by side.

**Explanation**

* Print the value counts of our original labels, `y`. Be mindful that `y` is currently a Numpy array, so in order to use value counts, we'll assign `y` back as a pandas Series object.
* Repeat the step and print the value counts on `y_resampled`. This shows you how the balance between the two classes has changed with SMOTE.
* Use the `compare_plot()` function called on our original data as well our resampled data to see the scatterplots side by side.


```python
pd.value_counts(pd.Series(y))
```




    0    5000
    1      50
    dtype: int64




```python
pd.value_counts(pd.Series(y_resampled))
```




    1    5000
    0    5000
    dtype: int64



#### def compare_plot


```python
def compare_plot(X: np.ndarray, y: np.ndarray, X_resampled: np.ndarray, y_resampled: np.ndarray, method: str):
    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title('Original Set')
    plt.subplot(1, 2, 2)
    plt.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title(method)
    plt.legend()
    plt.show()
```


```python
compare_plot(X, y, X_resampled, y_resampled, method='SMOTE')
```


![png](output_50_0.png)


**It should by now be clear that SMOTE has balanced our data completely, and that the minority class is now equal in size to the majority class. Visualizing the data shows the effect on the data very clearly. The next exercise will demonstrate multiple ways to implement SMOTE and that each method will have a slightly different effect.**

## Fraud detection algorithms in action

#### Rules Based Systems

* ![rules based](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/rules_based.JPG)
* Might block transactions from risky zip codes
* Block transactions from cards used too frequently (e.g. last 30 minutes)
* Can catch fraud, but also generates false alarms (false positive)
* Limitations:
    * Fixed threshold per rule and it's difficult to determine the threshold; they don't adapt over time
    * Limited to yes / no outcomes, whereas ML yields a probability
        * probability allows for fine-tuning the outcomes (i.e. rate of occurences of false positives and false negatives)
    * Fails to capture interaction between features
        * Ex. Size of the transaction only matters in combination to the frequency

#### ML Based Systems

* Adapt to the data, thus can change over time
* Uses all the data combined, rather than a threshold per feature
* Produces a probability, rather than a binary score
* Typically have better performance and can be combined with rules


```python
# Step 1: split the features and labels into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```


```python
# Step 2: Define which model to use
model = LinearRegression()
```


```python
# Step 3: Fit the model to the training data
model.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# Step 4: Obtain model predictions from the test data
y_predicted = model.predict(X_test)
```


```python
# Step 5: Compare y_test to predictions and obtain performance metrics (r^2 score)
r2_score(y_test, y_predicted)
```




    0.6185374148907309



### Exploring the traditional method of fraud detection

In this exercise you're going to try finding fraud cases in our credit card dataset the *"old way"*. First you'll define threshold values using common statistics, to split fraud and non-fraud. Then, use those thresholds on your features to detect fraud. This is common practice within fraud analytics teams.

Statistical thresholds are often determined by looking at the **mean** values of observations. Let's start this exercise by checking whether feature **means differ between fraud and non-fraud cases**. Then, you'll use that information to create common sense thresholds. Finally, you'll check how well this performs in fraud detection.

`pandas` has already been imported as `pd`.

**Explanation**

* Use `groupby()` to group `df` on `Class` and obtain the mean of the features.
* Create the condition `V1` smaller than -3, and `V3` smaller than -5 as a condition to flag fraud cases.
* As a measure of performance, use the `crosstab` function from `pandas` to compare our flagged fraud cases to actual fraud cases.


```python
df.drop(['Unnamed: 0'], axis=1, inplace=True)
```


```python
df.groupby('Class').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
    </tr>
    <tr>
      <th>Class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.035030</td>
      <td>0.011553</td>
      <td>0.037444</td>
      <td>-0.045760</td>
      <td>-0.013825</td>
      <td>-0.030885</td>
      <td>0.014315</td>
      <td>-0.022432</td>
      <td>-0.002227</td>
      <td>0.001667</td>
      <td>-0.004511</td>
      <td>0.017434</td>
      <td>0.004204</td>
      <td>0.006542</td>
      <td>-0.026640</td>
      <td>0.001190</td>
      <td>0.004481</td>
      <td>-0.010892</td>
      <td>-0.016554</td>
      <td>-0.002896</td>
      <td>-0.010583</td>
      <td>-0.010206</td>
      <td>-0.003305</td>
      <td>-0.000918</td>
      <td>-0.002613</td>
      <td>-0.004651</td>
      <td>-0.009584</td>
      <td>0.002414</td>
      <td>85.843714</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.985211</td>
      <td>3.321539</td>
      <td>-7.293909</td>
      <td>4.827952</td>
      <td>-3.326587</td>
      <td>-1.591882</td>
      <td>-5.776541</td>
      <td>1.395058</td>
      <td>-2.537728</td>
      <td>-5.917934</td>
      <td>4.020563</td>
      <td>-7.032865</td>
      <td>-0.104179</td>
      <td>-7.100399</td>
      <td>-0.120265</td>
      <td>-4.658854</td>
      <td>-7.589219</td>
      <td>-2.650436</td>
      <td>0.894255</td>
      <td>0.194580</td>
      <td>0.703182</td>
      <td>0.069065</td>
      <td>-0.088374</td>
      <td>-0.029425</td>
      <td>-0.073336</td>
      <td>-0.023377</td>
      <td>0.380072</td>
      <td>0.009304</td>
      <td>113.469000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['flag_as_fraud'] = np.where(np.logical_and(df.V1 < -3, df.V3 < -5), 1, 0)
```


```python
pd.crosstab(df.Class, df.flag_as_fraud, rownames=['Actual Fraud'], colnames=['Flagged Fraud'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Flagged Fraud</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual Fraud</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4984</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>



**With this rule, 22 out of 50 fraud cases are detected, 28 are not detected, and 16 false positives are identified.**

### Using ML classification to catch fraud

In this exercise you'll see what happens when you use a simple machine learning model on our credit card data instead.

Do you think you can beat those results? Remember, you've predicted *22 out of 50* fraud cases, and had *16 false positives*.

So with that in mind, let's implement a **Logistic Regression** model. If you have taken the class on supervised learning in Python, you should be familiar with this model. If not, you might want to refresh that at this point. But don't worry, you'll be guided through the structure of the machine learning model.

The `X` and `y` variables are available in your workspace.

**Explanation**

* Split `X` and `y` into training and test data, keeping 30% of the data for testing.
* Fit your model to your training data.
* Obtain the model predicted labels by running `model.predict` on `X_test`.
* Obtain a classification comparing `y_test` with `predicted`, and use the given confusion matrix to check your results.


```python
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```


```python
# Fit a logistic regression model to our data
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# Obtain model predictions
predicted = model.predict(X_test)
```


```python
# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)
```

    Classification report:
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1505
               1       0.89      0.80      0.84        10
    
        accuracy                           1.00      1515
       macro avg       0.94      0.90      0.92      1515
    weighted avg       1.00      1.00      1.00      1515
    
    Confusion matrix:
     [[1504    1]
     [   2    8]]


**Do you think these results are better than the rules based model? We are getting far fewer false positives, so that's an improvement. Also, we're catching a higher percentage of fraud cases, so that is also better than before. Do you understand why we have fewer observations to look at in the confusion matrix? Remember we are using only our test data to calculate the model results on. We're comparing the crosstab on the full dataset from the last exercise, with a confusion matrix of only 30% of the total dataset, so that's where that difference comes from. In the next chapter, we'll dive deeper into understanding these model performance metrics. Let's now explore whether we can improve the prediction results even further with resampling methods.**

### Logistic regression with SMOTE

In this exercise, you're going to take the Logistic Regression model from the previous exercise, and combine that with a **SMOTE resampling method**. We'll show you how to do that efficiently by using a pipeline that combines the resampling method with the model in one go. First, you need to define the pipeline that you're going to use.

**Explanation**

* Import the `Pipeline` module from `imblearn`, this has been done for you.
* Then define what you want to put into the pipeline, assign the `SMOTE` method with `borderline2` to `resampling`, and assign `LogisticRegression()` to the `model`.
* Combine two steps in the `Pipeline()` function. You need to state you want to combine `resampling` with the `model` in the respective place in the argument. I show you how to do this.


```python
# Define which resampling method and which ML model to use in the pipeline
resampling = SMOTE(kind='borderline2')
model = LogisticRegression(solver='liblinear')
```


```python
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])
```

### Pipelining

Now that you have our pipeline defined, aka **combining a logistic regression with a SMOTE method**, let's run it on the data. You can treat the pipeline as if it were a **single machine learning model**. Our data X and y are already defined, and the pipeline is defined in the previous exercise. Are you curious to find out what the model results are? Let's give it a try!

**Explanation**

* Split the data 'X'and 'y' into the training and test set. Set aside 30% of the data for a test set, and set the `random_state` to zero.
* Fit your pipeline onto your training data and obtain the predictions by running the `pipeline.predict()` function on our `X_test` dataset.


```python
# Split your data X and y, into a training and a test set and fit the pipeline onto the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```


```python
pipeline.fit(X_train, y_train) 
predicted = pipeline.predict(X_test)
```

    /usr/local/lib/python3.6/dist-packages/imblearn/over_sampling/_smote.py:749: DeprecationWarning: "kind" is deprecated in 0.4 and will be removed in 0.6. Use SMOTE, BorderlineSMOTE or SVMSMOTE instead.
      'SVMSMOTE instead.', DeprecationWarning)



```python
# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)
```

    Classifcation report:
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1505
               1       0.62      1.00      0.77        10
    
        accuracy                           1.00      1515
       macro avg       0.81      1.00      0.88      1515
    weighted avg       1.00      1.00      1.00      1515
    
    Confusion matrix:
     [[1499    6]
     [   0   10]]


**As you can see, the SMOTE slightly improves our results. We now manage to find all cases of fraud, but we have a slightly higher number of false positives, albeit only 7 cases. Remember, resampling doesn't necessarily lead to better results. When the fraud cases are very spread and scattered over the data, using SMOTE can introduce a bit of bias. Nearest neighbors aren't necessarily also fraud cases, so the synthetic samples might 'confuse' the model slightly. In the next chapters, we'll learn how to also adjust our machine learning models to better detect the minority fraud cases.**

# Fraud detection using labeled data

Learn how to flag fraudulent transactions with supervised learning. Use classifiers, adjust and compare them to find the most efficient fraud detection model.

## Review classification methods

* Classification:
    * The problem of identifying to which class a new observation belongs, on the basis of a training set of data containing observations whose class is known
    * Goal: use known fraud cases to train a model to recognize new cases
    * Classes are sometimes called targets, labels or categories
    * Spam detection in email service providers can be identified as a classification problem
        * Binary classification since there are only 2 classes, spam and not spam
    * Fraud detection is also a binary classification prpoblem
    * Patient diagnosis
    * Classification problems normall have categorical output like yes/no, 1/0 or True/False
    * Variable to predict: $$y\in0,1$$
        * 0: negative calss ('majority' normal cases)
        * 1: positive class ('minority' fraud cases)

#### Logistic Regression

* Logistic Regression is one of the most used ML algorithms in binary classification
* ![logistic regression](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/logistic_regression.JPG)
* Can be adjusted reasonably well to work on imbalanced data...useful for fraud detection

#### Neural Network

* ![neural network](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/neural_network.JPG)
* Can be used as classifiers for fraud detection
* Capable of fitting highly non-linear models to the data
* More complex to implement than other classifiers - not demonstrated here

#### Decision Trees

* ![decision tree](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/decision_tree.JPG)
* Commonly used for fraud detection
* Transparent results, easily interpreted by analysts
* Decision trees are prone to overfit the data

#### Random Forests

* ![random forest](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/random_forest.JPG)
* **Random Forests are a more robust option than a single decision tree**
    * Construct a multitude of decision trees when training the model and outputting the class that is the mode or mean predicted class of the individual trees
    * A random forest consists of a collection of trees on a random subset of features
    * Final predictions are the combined results of those trees
    * Random forests can handle complex data and are not prone to overfit
    * They are interpretable by looking at feature importance, and can be adjusted to work well on highly imbalanced data
    * Their drawback is they're computationally complex
    * Very popular for fraud detection
    * A Random Forest model will be optimized in the exercises
    
**Implementation:**

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print(f'Accuracy Score:\n{accuracy_score(y_test, predicted)}')
```

### Natural hit rate

In this exercise, you'll again use credit card transaction data. The features and labels are similar to the data in the previous chapter, and the **data is heavily imbalanced**. We've given you features `X` and labels `y` to work with already, which are both numpy arrays.

First you need to explore how prevalent fraud is in the dataset, to understand what the **"natural accuracy"** is, if we were to predict everything as non-fraud. It's is important to understand which level of "accuracy" you need to "beat" in order to get a **better prediction than by doing nothing**. In the following exercises, you'll create our first random forest classifier for fraud detection. That will serve as the **"baseline"** model that you're going to try to improve in the upcoming exercises.

**Explanation**

* Count the total number of observations by taking the length of your labels `y`.
* Count the non-fraud cases in our data by using list comprehension on `y`; remember `y` is a NumPy array so `.value_counts()` cannot be used in this case.
* Calculate the natural accuracy by dividing the non-fraud cases over the total observations.
* Print the percentage.


```python
df2 = pd.read_csv(cc2_file)
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221547</td>
      <td>-1.191668</td>
      <td>0.428409</td>
      <td>1.640028</td>
      <td>-1.848859</td>
      <td>-0.870903</td>
      <td>-0.204849</td>
      <td>-0.385675</td>
      <td>0.352793</td>
      <td>-1.098301</td>
      <td>-0.334597</td>
      <td>-0.679089</td>
      <td>-0.039671</td>
      <td>1.372661</td>
      <td>-0.732001</td>
      <td>-0.344528</td>
      <td>1.024751</td>
      <td>0.380209</td>
      <td>-1.087349</td>
      <td>0.364507</td>
      <td>0.051924</td>
      <td>0.507173</td>
      <td>1.292565</td>
      <td>-0.467752</td>
      <td>1.244887</td>
      <td>0.697707</td>
      <td>0.059375</td>
      <td>-0.319964</td>
      <td>-0.017444</td>
      <td>27.44</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>184524</td>
      <td>1.966614</td>
      <td>-0.450087</td>
      <td>-1.228586</td>
      <td>0.142873</td>
      <td>-0.150627</td>
      <td>-0.543590</td>
      <td>-0.076217</td>
      <td>-0.108390</td>
      <td>0.973310</td>
      <td>-0.029903</td>
      <td>0.279973</td>
      <td>0.885685</td>
      <td>-0.583912</td>
      <td>0.322019</td>
      <td>-1.065335</td>
      <td>-0.340285</td>
      <td>-0.385399</td>
      <td>0.216554</td>
      <td>0.675646</td>
      <td>-0.190851</td>
      <td>0.124055</td>
      <td>0.564916</td>
      <td>-0.039331</td>
      <td>-0.283904</td>
      <td>0.186400</td>
      <td>0.192932</td>
      <td>-0.039155</td>
      <td>-0.071314</td>
      <td>35.95</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>91201</td>
      <td>1.528452</td>
      <td>-1.296191</td>
      <td>-0.890677</td>
      <td>-2.504028</td>
      <td>0.803202</td>
      <td>3.350793</td>
      <td>-1.633016</td>
      <td>0.815350</td>
      <td>-1.884692</td>
      <td>1.465259</td>
      <td>-0.188235</td>
      <td>-0.976779</td>
      <td>0.560550</td>
      <td>-0.250847</td>
      <td>0.936115</td>
      <td>0.136409</td>
      <td>-0.078251</td>
      <td>0.355086</td>
      <td>0.127756</td>
      <td>-0.163982</td>
      <td>-0.412088</td>
      <td>-1.017485</td>
      <td>0.129566</td>
      <td>0.948048</td>
      <td>0.287826</td>
      <td>-0.396592</td>
      <td>0.042997</td>
      <td>0.025853</td>
      <td>28.40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26115</td>
      <td>-0.774614</td>
      <td>1.100916</td>
      <td>0.679080</td>
      <td>1.034016</td>
      <td>0.168633</td>
      <td>0.874582</td>
      <td>0.209454</td>
      <td>0.770550</td>
      <td>-0.558106</td>
      <td>-0.165442</td>
      <td>0.017562</td>
      <td>0.285377</td>
      <td>-0.818739</td>
      <td>0.637991</td>
      <td>-0.370124</td>
      <td>-0.605148</td>
      <td>0.275686</td>
      <td>0.246362</td>
      <td>1.331927</td>
      <td>0.080978</td>
      <td>0.011158</td>
      <td>0.146017</td>
      <td>-0.130401</td>
      <td>-0.848815</td>
      <td>0.005698</td>
      <td>-0.183295</td>
      <td>0.282940</td>
      <td>0.123856</td>
      <td>43.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>201292</td>
      <td>-1.075860</td>
      <td>1.361160</td>
      <td>1.496972</td>
      <td>2.242604</td>
      <td>1.314751</td>
      <td>0.272787</td>
      <td>1.005246</td>
      <td>0.132932</td>
      <td>-1.558317</td>
      <td>0.484216</td>
      <td>-1.967998</td>
      <td>-1.818338</td>
      <td>-2.036184</td>
      <td>0.346962</td>
      <td>-1.161316</td>
      <td>1.017093</td>
      <td>-0.926787</td>
      <td>0.183965</td>
      <td>-2.102868</td>
      <td>-0.354008</td>
      <td>0.254485</td>
      <td>0.530692</td>
      <td>-0.651119</td>
      <td>0.626389</td>
      <td>1.040212</td>
      <td>0.249501</td>
      <td>-0.146745</td>
      <td>0.029714</td>
      <td>10.59</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X, y = prep_data(df2)
print(f'X shape: {X.shape}\ny shape: {y.shape}')
```

    X shape: (7300, 28)
    y shape: (7300,)



```python
X[0, :]
```




    array([ 4.28408570e-01,  1.64002800e+00, -1.84885886e+00, -8.70902974e-01,
           -2.04848888e-01, -3.85675453e-01,  3.52792552e-01, -1.09830131e+00,
           -3.34596757e-01, -6.79088729e-01, -3.96709268e-02,  1.37266082e+00,
           -7.32000706e-01, -3.44528134e-01,  1.02475103e+00,  3.80208554e-01,
           -1.08734881e+00,  3.64507163e-01,  5.19236276e-02,  5.07173439e-01,
            1.29256539e+00, -4.67752261e-01,  1.24488683e+00,  6.97706854e-01,
            5.93750372e-02, -3.19964326e-01, -1.74444289e-02,  2.74400000e+01])




```python
df2.Class.value_counts()
```




    0    7000
    1     300
    Name: Class, dtype: int64




```python
# Count the total number of observations from the length of y
total_obs = len(y)
total_obs
```




    7300




```python
# Count the total number of non-fraudulent observations 
non_fraud = [i for i in y if i == 0]
count_non_fraud = non_fraud.count(0)
count_non_fraud
```




    7000




```python
percentage = count_non_fraud/total_obs * 100
print(f'{percentage:0.2f}%')
```

    95.89%


**This tells us that by doing nothing, we would be correct in 95.9% of the cases. So now you understand, that if we get an accuracy of less than this number, our model does not actually add any value in predicting how many cases are correct. Let's see how a random forest does in predicting fraud in our data.**

### Random Forest Classifier - part 1

Let's now create a first **random forest classifier** for fraud detection. Hopefully you can do better than the baseline accuracy you've just calculated, which was roughly **96%**. This model will serve as the **"baseline" model** that you're going to try to improve in the upcoming exercises. Let's start first with **splitting the data into a test and training set**, and **defining the Random Forest model**. The data available are features `X` and labels `y`.

**Explanation**

* Import the random forest classifier from `sklearn`.
* Split your features `X` and labels `y` into a training and test set. Set aside a test set of 30%.
* Assign the random forest classifier to `model` and keep `random_state` at 5. We need to set a random state here in order to be able to compare results across different models.

#### X_train, X_test, y_train, y_test


```python
# Split your data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```


```python
# Define the model as the random forest
model = RandomForestClassifier(random_state=5, n_estimators=20)
```

### Random Forest Classifier - part 2

Let's see how our Random Forest model performs **without doing anything special to it**. The `model` from the previous exercise is available, and you've already split your data in `X_train, y_train, X_test, y_test`.

**Explanation 1/3**

* Fit the earlier defined `model` to our training data and obtain predictions by getting the model predictions on `X_test`.


```python
# Fit the model to our training set
model.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=20,
                           n_jobs=None, oob_score=False, random_state=5, verbose=0,
                           warm_start=False)




```python
# Obtain predictions from the test data 
predicted = model.predict(X_test)
```

**Explanation 2/3**

* Obtain and print the accuracy score by comparing the actual labels `y_test` with our predicted labels `predicted`.


```python
print(f'Accuracy Score:\n{accuracy_score(y_test, predicted):0.3f}')
```

    Accuracy Score:
    0.991


**Explanation 3/3**

What is a benefit of using Random Forests versus Decision Trees?

**Possible Answers**

* ~~Random Forests always have a higher accuracy than Decision Trees.~~
* **Random Forests do not tend to overfit, whereas Decision Trees do.**
* ~~Random Forests are computationally more efficient than Decision Trees.~~
* ~~You can obtain "feature importance" from Random Forest, which makes it more transparent.~~

**Random Forest prevents overfitting most of the time, by creating random subsets of the features and building smaller trees using these subsets. Afterwards, it combines the subtrees of subsamples of features, so it does not tend to overfit to your entire feature set the way "deep" Decisions Trees do.**

## Perfomance evaluation

* Performance metrics for fraud detection models
* There are other performace metrics that are more informative and reliable than accuracy

#### Accuracy

![accuracy](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/accuracy.JPG)
* Accuracy isn't a reliable performance metric when working with highly imbalanced data (such as fraud detection)
* By doing nothing, aka predicting everything is the majority class (right image), a higher accuracy is obtained than by trying to build a predictive model (left image)

#### Confusion Matrix

![advanced confusion matrix](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/confusion_matrix_advanced.JPG)
![confusion matrix](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/confusion_matrix.JPG)
* [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
* False Positives (FP) / False Negatives (FN)
    * FN: predicts the person is not pregnant, but actually is
        * Cases of fraud not caught by the model
    * FP: predicts the person is pregnant, but actually is not
        * Cases of 'false alarm'
    * the business case determines whether FN or FP cases are more important
        * a credit card company might want to catch as much fraud as possible and reduce false negatives, as fraudulent transactions can be incredibly costly
            * a false alarm just means a transaction is blocked
        * an insurance company can't handle many false alarms, as it means getting a team of investigators involved for each positive prediction
        
* True Positives / True Negatives are the cases predicted correctly (e.g. fraud / non-fraud)

#### Precision Recall

* Credit card company wants to optimize for recall
* Insurance company wants to optimize for precision
* Precision:
    * $$Precision=\frac{\#\space True\space Positives}{\#\space True\space Positives+\#\space False\space Positives}$$
    * Fraction of actual fraud cases out of all predicted fraud cases
        * true positives relative to the sum of true positives and false positives
* Recall:
    * $$Recall=\frac{\#\space True\space Positives}{\#\space True\space Positives+\#\space False\space Negatives}$$
    * Fraction of predicted fraud cases out of all actual fraud cases
        * true positives relative to the sum of true positives and false negative
* Precision and recall are typically inversely related
    * As precision increases, recall falls and vice-versa
    * ![precision recall inverse relation](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/precision_recall_inverse.JPG)

#### F-Score

* Weighs both precision and recall into on measure

\begin{align}
F-measure = \frac{2\times{Precision}\times{Recall}}{Precision\times{Recall}} \\ 
\\
= \frac{2\times{TP}}{2\times{TP}+FP+FN}
\end{align}

* is a performance metric that takes into account a balance between Precision and Recall

#### Obtaining performance metrics from sklean

```python
# import the methods
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate average precision and the PR curve
average_precision = average_precision_score(y_test, predicted)

# Obtain precision and recall
precision, recall = precision_recall_curve(y_test, predicted)
```

#### Receiver Operating Characteristic (ROC) curve to compare algorithms

* Created by plotting the true positive rate against the false positive rate at various threshold settings
* ![roc curve](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/roc_curve.JPG)
* Useful for comparing performance of different algorithms

```python
# Obtain model probabilities
probs = model.predict_proba(X_test)

# Print ROC_AUC score using probabilities
print(metrics.roc_auc_score(y_test, probs[:, 1]))
```

#### Confusion matrix and classification report

```python
from sklearn.metrics import classification_report, confusion_matrix

# Obtain predictions
predicted = model.predict(X_test)

# Print classification report using predictions
print(classification_report(y_test, predicted))

# Print confusion matrix using predictions
print(confusion_matrix(y_test, predicted))
```

### Performance metrics for the RF model

In the previous exercises you obtained an accuracy score for your random forest model. This time, we know **accuracy can be misleading** in the case of fraud detection. With highly imbalanced fraud data, the AUROC curve is a more reliable performance metric, used to compare different classifiers. Moreover, the *classification report* tells you about the precision and recall of your model, whilst the *confusion matrix* actually shows how many fraud cases you can predict correctly. So let's get these performance metrics.

You'll continue working on the same random forest model from the previous exercise. Your model, defined as `model = RandomForestClassifier(random_state=5)` has been fitted to your training data already, and `X_train, y_train, X_test, y_test` are available.

**Explanation**

* Import the classification report, confusion matrix and ROC score from `sklearn.metrics`.
* Get the binary predictions from your trained random forest `model`.
* Get the predicted probabilities by running the `predict_proba()` function.
* Obtain classification report and confusion matrix by comparing `y_test` with `predicted`.


```python
# Obtain the predictions from our random forest model 
predicted = model.predict(X_test)
```


```python
# Predict probabilities
probs = model.predict_proba(X_test)
```


```python
# Print the ROC curve, classification report and confusion matrix
print('ROC Score:')
print(roc_auc_score(y_test, probs[:,1]))
print('\nClassification Report:')
print(classification_report(y_test, predicted))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predicted))
```

    ROC Score:
    0.9419896444670147
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      1.00      1.00      2099
               1       0.97      0.80      0.88        91
    
        accuracy                           0.99      2190
       macro avg       0.98      0.90      0.94      2190
    weighted avg       0.99      0.99      0.99      2190
    
    
    Confusion Matrix:
    [[2097    2]
     [  18   73]]


**You have now obtained more meaningful performance metrics that tell us how well the model performs, given the highly imbalanced data that you're working with. The model predicts 76 cases of fraud, out of which 73 are actual fraud. You have only 3 false positives. This is really good, and as a result you have a very high precision score. You do however, miss 18 cases of actual fraud. Recall is therefore not as good as precision.**

### Plotting the Precision vs. Recall Curve

You can also plot a **Precision-Recall curve**, to investigate the trade-off between the two in your model. In this curve **Precision and Recall are inversely related**; as Precision increases, Recall falls and vice-versa. A balance between these two needs to be achieved in your model, otherwise you might end up with many false positives, or not enough actual fraud cases caught. To achieve this and to compare performance, the precision-recall curves come in handy.

Your Random Forest Classifier is available as `model`, and the predictions as `predicted`. You can simply obtain the average precision score and the PR curve from the sklearn package. The function `plot_pr_curve()` plots the results for you. Let's give it a try.

**Explanation 1/3**

* Calculate the average precision by running the function on the actual labels `y_test` and your predicted labels `predicted`.


```python
# Calculate average precision and the PR curve
average_precision = average_precision_score(y_test, predicted)
average_precision
```




    0.7890250388880526



**Explanation 2/3**

* Run the `precision_recall_curve()` function on the same arguments `y_test` and `predicted` and plot the curve (this last thing has been done for you).


```python
# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, predicted)
print(f'Precision: {precision}\nRecall: {recall}')
```

    Precision: [0.04155251 0.97333333 1.        ]
    Recall: [1.        0.8021978 0.       ]


#### def plot_pr_curve


```python
def plot_pr_curve(recall, precision, average_precision):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    from inspect import signature
    plt.figure()
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title(f'2-class Precision-Recall curve: AP={average_precision:0.2f}')
    return plt.show()
```


```python
# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)
```


![png](output_124_0.png)


**Explanation 3/3**

What's the benefit of the performance metric ROC curve (AUROC) versus Precision and Recall?

**Possible Answers**

* **The AUROC answers the question: "How well can this classifier be expected to perform in general, at a variety of different baseline probabilities?" but precision and recall don't.**
* ~~The AUROC answers the question: "How meaningful is a positive result from my classifier given the baseline probabilities of my problem?" but precision and recall don't.~~
* ~~Precision and Recall are not informative when the data is imbalanced.~~
* ~~The AUROC curve allows you to visualize classifier performance and with Precision and Recall you cannot.~~

**The ROC curve plots the true positives vs. false positives , for a classifier, as its discrimination threshold is varied. Since, a random method describes a horizontal curve through the unit interval, it has an AUC of 0.5. Minimally, classifiers should perform better than this, and the extent to which they score higher than one another (meaning the area under the ROC curve is larger), they have better expected performance.**

## Adjusting the algorithm weights

* Adjust model parameter to optimize for fraud detection.
* When training a model, try different options and settings to get the best recall-precision trade-off
* sklearn has two simple options to tweak the model for heavily imbalanced data
    * `class_weight`:
        * `balanced` mode: `model = RandomForestClassifier(class_weight='balanced')`
            * uses the values of y to automatically adjust weights inversely proportional to class frequencies in the the input data
            * this option is available for other classifiers
                * `model = LogisticRegression(class_weight='balanced')`
                * `model = SVC(kernel='linear', class_weight='balanced', probability=True)`
        * `balanced_subsample` mode: `model = RandomForestClassifier(class_weight='balanced_subsample')`
            * is the same as the `balanced` option, except weights are calculated again at each iteration of growing a tree in a the random forest
            * this option is only applicable for the Random Forest model
        * manual input
            * adjust weights to any ratio, not just value counts relative to sample
            * `class_weight={0:1,1:4}`
            * this is a good option to slightly upsample the minority class

#### Hyperparameter tuning

* Random Forest takes many other options to optimize the model

```python
model = RandomForestClassifier(n_estimators=10, 
                               criterion=’gini’, 
                               max_depth=None, 
                               min_samples_split=2, 
                               min_samples_leaf=1, 
                               max_features=’auto’, 
                               n_jobs=-1, class_weight=None)
```

* the shape and size of the trees in a random forest are adjusted with **leaf size** and **tree depth**
* `n_estimators`: one of the most important setting is the number of trees in the forest
* `max_features`: the number of features considered for splitting at each leaf node
* `criterion`: change the way the data is split at each node (default is `gini` coefficient)

#### GridSearchCV for hyperparameter tuning

* [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
* `from sklearn.model_selection import GridSearchCV`
* `GridSearchCV evaluates all combinations of parameters defined in the parameter grid
* Random Forest Parameter Grid:

```python
# Create the parameter grid 
param_grid = {'max_depth': [80, 90, 100, 110],
              'max_features': [2, 3],
              'min_samples_leaf': [3, 4, 5],
              'min_samples_split': [8, 10, 12],
              'n_estimators': [100, 200, 300, 1000]}

# Define which model to use
model = RandomForestRegressor()

# Instantiate the grid search model
grid_search_model = GridSearchCV(estimator = model, 
                                 param_grid = param_grid, 
                                 cv = 5,
                                 n_jobs = -1, 
                                 scoring='f1')
```

* define the ML model to be used
* put the model into `GridSearchCV`
* pass in `param_grid`
* frequency of cross-validation
* define a scoring metric to evaluate the models
    * the default option is accuracy which isn't optimal for fraud detection
    * use `precision`, `recall` or `f1`

```python
# Fit the grid search to the data
grid_search_model.fit(X_train, y_train)

# Get the optimal parameters 
grid_search_model.best_params_

{'bootstrap': True,
 'max_depth': 80,
 'max_features': 3,
 'min_samples_leaf': 5,
 'min_samples_split': 12,
 'n_estimators': 100}
```

* once `GridSearchCV` and `model` are fit to the data, obtain the parameters belonging to the optimal model by using the `best_params_` attribute
* `GridSearchCV` is computationally heavy
    * Can require many hours, depending on the amount of data and number of parameters in the grid
    * __**Save the Results**__

```python
# Get the best_estimator results
grid_search.best_estimator_
grid_search.best_score_
```

* `best_score_`: mean cross-validated score of the `best_estimator_`, which depends on the `scoring` option

### Model adjustments

A simple way to adjust the random forest model to deal with highly imbalanced fraud data, is to use the **`class_weights` option** when defining the `sklearn` model. However, as you will see, it is a bit of a blunt force mechanism and might not work for your very special case.

In this exercise you'll explore the ``weight = "balanced_subsample"`` mode the Random Forest model from the earlier exercise. You already have split your data in a training and test set, i.e `X_train`, `X_test`, `y_train`, `y_test` are available. The metrics function have already been imported.

**Explanation**

* Set the `class_weight` argument of your classifier to `balanced_subsample`.
* Fit your model to your training set.
* Obtain predictions and probabilities from X_test.
* Obtain the `roc_auc_score`, the classification report and confusion matrix.


```python
# Define the model with balanced subsample
model = RandomForestClassifier(class_weight='balanced_subsample', random_state=5, n_estimators=100)

# Fit your training model to your training set
model.fit(X_train, y_train)

# Obtain the predicted values and probabilities from the model 
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)

# Print the ROC curve, classification report and confusion matrix
print('ROC Score:')
print(roc_auc_score(y_test, probs[:,1]))
print('\nClassification Report:')
print(classification_report(y_test, predicted))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predicted))
```

    ROC Score:
    0.9712788402640714
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      1.00      1.00      2099
               1       0.99      0.80      0.88        91
    
        accuracy                           0.99      2190
       macro avg       0.99      0.90      0.94      2190
    weighted avg       0.99      0.99      0.99      2190
    
    
    Confusion Matrix:
    [[2098    1]
     [  18   73]]


**You can see that the model results don't improve drastically. We now have 3 less false positives, but now 19 in stead of 18 false negatives, i.e. cases of fraud we are not catching. If we mostly care about catching fraud, and not so much about the false positives, this does actually not improve our model at all, albeit a simple option to try. In the next exercises you'll see how to more smartly tweak your model to focus on reducing false negatives and catch more fraud.**

### Adjusting RF for fraud detection

In this exercise you're going to dive into the options for the random forest classifier, as we'll **assign weights** and **tweak the shape** of the decision trees in the forest. You'll **define weights manually**, to be able to off-set that imbalance slightly. In our case we have 300 fraud to 7000 non-fraud cases, so by setting the weight ratio to 1:12, we get to a 1/3 fraud to 2/3 non-fraud ratio, which is good enough for training the model on.

The data in this exercise has already been split into training and test set, so you just need to focus on defining your model. You can then use the function `get_model_results()` as a short cut. This function fits the model to your training data, predicts and obtains performance metrics similar to the steps you did in the previous exercises.

**Explanation**

* Change the `weight` option to set the ratio to 1 to 12 for the non-fraud and fraud cases, and set the split criterion to 'entropy'.
* Set the maximum depth to 10.
* Set the minimal samples in leaf nodes to 10.
* Set the number of trees to use in the model to 20.

#### def get_model_results


```python
def get_model_results(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, model):
    """
    model: sklearn model (e.g. RandomForestClassifier)
    """
    # Fit your training model to your training set
    model.fit(X_train, y_train)

    # Obtain the predicted values and probabilities from the model 
    predicted = model.predict(X_test)
    
    try:
        probs = model.predict_proba(X_test)
        print('ROC Score:')
        print(roc_auc_score(y_test, probs[:,1]))
    except AttributeError:
        pass

    # Print the ROC curve, classification report and confusion matrix
    print('\nClassification Report:')
    print(classification_report(y_test, predicted))
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, predicted))
```


```python
# Change the model options
model = RandomForestClassifier(bootstrap=True,
                               class_weight={0:1, 1:12},
                               criterion='entropy',
                               # Change depth of model
                               max_depth=10,
                               # Change the number of samples in leaf nodes
                               min_samples_leaf=10, 
                               # Change the number of trees to use
                               n_estimators=20,
                               n_jobs=-1,
                               random_state=5)

# Run the function get_model_results
get_model_results(X_train, y_train, X_test, y_test, model)
```

    ROC Score:
    0.9609651901219315
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      1.00      1.00      2099
               1       0.97      0.85      0.91        91
    
        accuracy                           0.99      2190
       macro avg       0.98      0.92      0.95      2190
    weighted avg       0.99      0.99      0.99      2190
    
    
    Confusion Matrix:
    [[2097    2]
     [  14   77]]


**By smartly defining more options in the model, you can obtain better predictions. You have effectively reduced the number of false negatives, i.e. you are catching more cases of fraud, whilst keeping the number of false positives low. In this exercise you've manually changed the options of the model. There is a smarter way of doing it, by using `GridSearchCV`, which you'll see in the next exercise!**

### Parameter optimization with GridSearchCV

In this exercise you're going to **tweak our model in a less "random" way**, but use `GridSearchCV` to do the work for you.

With `GridSearchCV` you can define **which performance metric to score** the options on. Since for fraud detection we are mostly interested in catching as many fraud cases as possible, you can optimize your model settings to get the best possible Recall score. If you also cared about reducing the number of false positives, you could optimize on F1-score, this gives you that nice Precision-Recall trade-off.

`GridSearchCV` has already been imported from `sklearn.model_selection`, so let's give it a try!

**Explanation**

* Define in the parameter grid that you want to try 1 and 30 trees, and that you want to try the `gini` and `entropy` split criterion.
* Define the model to be simple `RandomForestClassifier`, you want to keep the random_state at 5 to be able to compare models.
* Set the `scoring` option such that it optimizes for recall.
* Fit the model to the training data `X_train` and `y_train` and obtain the best parameters for the model.


```python
# Define the parameter sets to test
param_grid = {'n_estimators': [1, 30],
              'max_features': ['auto', 'log2'], 
              'max_depth': [4, 8, 10, 12],
              'criterion': ['gini', 'entropy']}

# Define the model to use
model = RandomForestClassifier(random_state=5)

# Combine the parameter sets with the defined model
CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)

# Fit the model to our training data and obtain best parameters
CV_model.fit(X_train, y_train)
CV_model.best_params_
```




    {'criterion': 'gini',
     'max_depth': 8,
     'max_features': 'log2',
     'n_estimators': 30}



### Model results with GridSearchCV

You discovered that the **best parameters for your model** are that the split criterion should be set to `'gini'`, the number of estimators (trees) should be 30, the maximum depth of the model should be 8 and the maximum features should be set to `"log2"`.

Let's give this a try and see how well our model performs. You can use the `get_model_results()` function again to save time.

**Explanation**

* Input the optimal settings into the model definition.
* Fit the model, obtain predictions and get the performance parameters with `get_model_results()`.


```python
# Input the optimal parameters in the model
model = RandomForestClassifier(class_weight={0:1,1:12},
                               criterion='gini',
                               max_depth=8,
                               max_features='log2', 
                               min_samples_leaf=10,
                               n_estimators=30,
                               n_jobs=-1,
                               random_state=5)

# Get results from your model
get_model_results(X_train, y_train, X_test, y_test, model)
```

    ROC Score:
    0.9749697658225529
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      1.00      1.00      2099
               1       0.95      0.84      0.89        91
    
        accuracy                           0.99      2190
       macro avg       0.97      0.92      0.94      2190
    weighted avg       0.99      0.99      0.99      2190
    
    
    Confusion Matrix:
    [[2095    4]
     [  15   76]]


**The model has been improved even further. The number of false positives has now been slightly reduced even further, which means we are catching more cases of fraud. However, you see that the number of false positives actually went up. That is that Precision-Recall trade-off in action. To decide which final model is best, you need to take into account how bad it is not to catch fraudsters, versus how many false positives the fraud analytics team can deal with. Ultimately, this final decision should be made by you and the fraud team together.**

## Ensemble methods

![ensemble](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/ensemble.JPG)
* Ensemble methods are techniques that create multiple machine learning models and then combine them to produce a final result
* Usually produce more accurate predictions than a single model
* The goal of an ML problem is to find a single model that will best predict our wanted outcome
    * Use ensemble methods rather than making one model and hoping it's best, most accurate predictor
* Ensemble methods take a myriad of models into account and average them to produce one final model
    * Ensures the predictions are robust
    * Less likely to be the result of overfitting
    * Can improve prediction performance
        * Especially by combining models with different recall and precision scores
    * Are a winning formula at Kaggle competitions
* The Random Forest classifier is an ensemble of Decision Trees
    * **Bootstrap Aggregation** or **Bagging Ensemble** method
    * In a Random Forest, models are trained on random subsamples of data and the results are aggregated by taking the average prediction of all the trees

#### Stacking Ensemble Methods

![stacking ensemble](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/ensemble_stacking.JPG)
* Multiple models are combined via a "voting" rule on the model outcome
* The base level models are each trained based on the complete training set
    * Unlike the Bagging method, models are not trained on a subsample of the data
* Algorithms of different types can be combined

#### Voting Classifier

* available in sklearn
    * easy way of implementing an ensemble model

```python
from sklearn.ensemble import VotingClassifier

# Define Models
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

# Combine models into ensemble
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

# Fit and predict as with other models
ensemble_model.fit(X_train, y_train)
ensemble_model.predict(X_test)
```

* the `voting='hard'` option uses the predicted class labels and takes the majority vote
* the `voting='soft'` option takes the average probability by combining the predicted probabilities of the individual models
* Weights can be assigned to the `VotingClassifer` with `weights=[2,1,1]`
    * Useful when one model significantly outperforms the others

#### Reliable Labels

* In real life it's unlikely the data will have truly unbiased, reliable labels for the model
* In credit card fraud you often will have reliable labels, in which case, use the methods learned so far
* Most cases you'll need to rely on unsupervised learning techniques to detect fraud

### Logistic Regression

In this last lesson you'll **combine three algorithms** into one model with the **VotingClassifier**. This allows us to benefit from the different aspects from all models, and hopefully improve overall performance and detect more fraud. The first model, the Logistic Regression, has a slightly higher recall score than our optimal Random Forest model, but gives a lot more false positives. You'll also add a Decision Tree with balanced weights to it. The data is already split into a training and test set, i.e. `X_train`, `y_train`, `X_test`, `y_test` are available.

In order to understand how the Voting Classifier can potentially improve your original model, you should check the standalone results of the Logistic Regression model first.

**Explanation**

* Define a LogisticRegression model with class weights that are 1:15 for the fraud cases.
* Fit the model to the training set, and obtain the model predictions.
* Print the classification report and confusion matrix.


```python
# Define the Logistic Regression model with weights
model = LogisticRegression(class_weight={0:1, 1:15}, random_state=5, solver='liblinear')

# Get the model results
get_model_results(X_train, y_train, X_test, y_test, model)
```

    ROC Score:
    0.9722054981702433
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.98      0.99      2099
               1       0.63      0.88      0.73        91
    
        accuracy                           0.97      2190
       macro avg       0.81      0.93      0.86      2190
    weighted avg       0.98      0.97      0.98      2190
    
    
    Confusion Matrix:
    [[2052   47]
     [  11   80]]


**As you can see the Logistic Regression has quite different performance from the Random Forest. More false positives, but also a better Recall. It will therefore will a useful addition to the Random Forest in an ensemble model.**

### Voting Classifier

Let's now **combine three machine learning models into one**, to improve our Random Forest fraud detection model from before. You'll combine our usual Random Forest model, with the Logistic Regression from the previous exercise, with a simple Decision Tree. You can use the short cut `get_model_results()` to see the immediate result of the ensemble model.

**Explanation**

* Import the Voting Classifier package.
* Define the three models; use the Logistic Regression from before, the Random Forest from previous exercises and a Decision tree with balanced class weights.
* Define the ensemble model by inputting the three classifiers with their respective labels.


```python
# Define the three classifiers to use in the ensemble
clf1 = LogisticRegression(class_weight={0:1, 1:15},
                          random_state=5,
                          solver='liblinear')

clf2 = RandomForestClassifier(class_weight={0:1, 1:12}, 
                              criterion='gini', 
                              max_depth=8, 
                              max_features='log2',
                              min_samples_leaf=10, 
                              n_estimators=30, 
                              n_jobs=-1,
                              random_state=5)

clf3 = DecisionTreeClassifier(random_state=5,
                              class_weight="balanced")

# Combine the classifiers in the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='hard')

# Get the results 
get_model_results(X_train, y_train, X_test, y_test, ensemble_model)
```

    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99      2099
               1       0.90      0.86      0.88        91
    
        accuracy                           0.99      2190
       macro avg       0.95      0.93      0.94      2190
    weighted avg       0.99      0.99      0.99      2190
    
    
    Confusion Matrix:
    [[2090    9]
     [  13   78]]


**By combining the classifiers, you can take the best of multiple models. You've increased the cases of fraud you are catching from 76 to 78, and you only have 5 extra false positives in return. If you do care about catching as many fraud cases as you can, whilst keeping the false positives low, this is a pretty good trade-off. The Logistic Regression as a standalone was quite bad in terms of false positives, and the Random Forest was worse in terms of false negatives. By combining these together you indeed managed to improve performance.**

### Adjusting weights within the Voting Classifier

You've just seen that the Voting Classifier allows you to improve your fraud detection performance, by combining good aspects from multiple models. Now let's try to **adjust the weights** we give to these models. By increasing or decreasing weights you can play with **how much emphasis you give to a particular model** relative to the rest. This comes in handy when a certain model has overall better performance than the rest, but you still want to combine aspects of the others to further improve your results.

For this exercise the data is already split into a training and test set, and `clf1`, `clf2` and `clf3` are available and defined as before, i.e. they are the Logistic Regression, the Random Forest model and the Decision Tree respectively.

**Explanation**

* Define an ensemble method where you over weigh the second classifier (`clf2`) with 4 to 1 to the rest of the classifiers.
* Fit the model to the training and test set, and obtain the predictions `predicted` from the ensemble model.
* Print the performance metrics, this is ready for you to run.


```python
# Define the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[1, 4, 1], flatten_transform=True)

# Get results 
get_model_results(X_train, y_train, X_test, y_test, ensemble_model)
```

    ROC Score:
    0.9739279300975348
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      1.00      1.00      2099
               1       0.94      0.85      0.89        91
    
        accuracy                           0.99      2190
       macro avg       0.97      0.92      0.94      2190
    weighted avg       0.99      0.99      0.99      2190
    
    
    Confusion Matrix:
    [[2094    5]
     [  14   77]]


**The weight option allows you to play with the individual models to get the best final mix for your fraud detection model. Now that you have finalized fraud detection with supervised learning, let's have a look at how fraud detetion can be done when you don't have any labels to train on.**

# Fraud detection using unlabeled data

Use unsupervised learning techniques to detect fraud. Segment customers, use K-means clustering and other clustering algorithms to find suspicious occurrences in your data.

## Normal versus abnormal behavior

* Explore fraud detection without reliable data labels
* Unsupervised learning to detect suspicious behavior
* Abnormal behavior isn't necessarily fraudulent
* Challenging because it's difficult to validate

#### What's normal behavior?

* thoroughly describe the data:
    * plot histograms
    * check for outliers
    * investigate correlations
* Are there any known historic cases of fraud? What typifies those cases?
* Investigate whether the data is homogeneous, or whether different types of clients display different behavior
* Check patterns within subgroups of data: is your data homogeneous?
* Verify data points are the same type:
    * individuals
    * groups
    * companies
    * governmental organizations
* Do the data points differ on:
    * spending patterns
    * age
    * location
    * frequency
* For credit card fraud, location can be an indication of fraud
* This goes for e-commerce sites
    * where's the IP address located and where is the product ordered to ship?
* Create a separate model for each segment
* How to aggregate the many model results back into one final list

### Exploring the data

In the next exercises, you will be looking at bank **payment transaction data**. The financial transactions are categorized by type of expense, as well as the amount spent. Moreover, you have some client characteristics available such as age group and gender. Some of the transactions are labeled as fraud; you'll treat these labels as given and will use those to validate the results.

When using unsupervised learning techniques for fraud detection, you want to **distinguish normal from abnormal** (thus potentially fraudulent) behavior. As a fraud analyst to understand what is "normal", you need to have a good understanding of the data and its characteristics. Let's explore the data in this first exercise.

**Explanation 1/3**

* Obtain the shape of the dataframe `df` to inspect the size of our data and display the first rows to see which features are available.


```python
banksim_df = pd.read_csv(banksim_file)
banksim_df.drop(['Unnamed: 0'], axis=1, inplace=True)
banksim_adj_df = pd.read_csv(banksim_adj_file)
banksim_adj_df.drop(['Unnamed: 0'], axis=1, inplace=True)
```


```python
banksim_df.shape
```




    (7200, 5)




```python
banksim_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>category</th>
      <th>amount</th>
      <th>fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>F</td>
      <td>es_transportation</td>
      <td>49.71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>F</td>
      <td>es_health</td>
      <td>39.29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>F</td>
      <td>es_transportation</td>
      <td>18.76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>M</td>
      <td>es_transportation</td>
      <td>13.95</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>M</td>
      <td>es_transportation</td>
      <td>49.87</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
banksim_adj_df.shape
```




    (7189, 18)




```python
banksim_adj_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>amount</th>
      <th>fraud</th>
      <th>M</th>
      <th>es_barsandrestaurants</th>
      <th>es_contents</th>
      <th>es_fashion</th>
      <th>es_food</th>
      <th>es_health</th>
      <th>es_home</th>
      <th>es_hotelservices</th>
      <th>es_hyper</th>
      <th>es_leisure</th>
      <th>es_otherservices</th>
      <th>es_sportsandtoys</th>
      <th>es_tech</th>
      <th>es_transportation</th>
      <th>es_travel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>49.71</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>39.29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>18.76</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13.95</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>49.87</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**Explanation 2/3**

* Group the data by transaction category and take the mean of the data.


```python
banksim_df.groupby(['category']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>fraud</th>
    </tr>
    <tr>
      <th>category</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>es_barsandrestaurants</th>
      <td>43.841793</td>
      <td>0.022472</td>
    </tr>
    <tr>
      <th>es_contents</th>
      <td>55.170000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>es_fashion</th>
      <td>59.780769</td>
      <td>0.020619</td>
    </tr>
    <tr>
      <th>es_food</th>
      <td>35.216050</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>es_health</th>
      <td>126.604704</td>
      <td>0.242798</td>
    </tr>
    <tr>
      <th>es_home</th>
      <td>120.688317</td>
      <td>0.208333</td>
    </tr>
    <tr>
      <th>es_hotelservices</th>
      <td>172.756245</td>
      <td>0.548387</td>
    </tr>
    <tr>
      <th>es_hyper</th>
      <td>46.788180</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>es_leisure</th>
      <td>229.757600</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>es_otherservices</th>
      <td>149.648960</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>es_sportsandtoys</th>
      <td>157.251737</td>
      <td>0.657895</td>
    </tr>
    <tr>
      <th>es_tech</th>
      <td>132.852862</td>
      <td>0.179487</td>
    </tr>
    <tr>
      <th>es_transportation</th>
      <td>27.422014</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>es_travel</th>
      <td>231.818656</td>
      <td>0.944444</td>
    </tr>
    <tr>
      <th>es_wellnessandbeauty</th>
      <td>66.167078</td>
      <td>0.060606</td>
    </tr>
  </tbody>
</table>
</div>



**Explanation 3/3**

Based on these results, can you already say something about fraud in our data?

**Possible Answers**

* ~~No, I don't have enough information.~~
* **Yes, the majority of fraud is observed in travel, leisure and sports related transactions.**

### Customer segmentation

In this exercise you're going to check whether there are any **obvious patterns** for the clients in this data, thus whether you need to segment your data into groups, or whether the data is rather homogenous.

You unfortunately don't have a lot client information available; you can't for example distinguish between the wealth levels of different clients. However, there is data on **age ** available, so let's see whether there is any significant difference between behavior of age groups.

**Explanation 1/3**

* Group the dataframe `df` by the category `age` and get the means for each age group.


```python
banksim_df.groupby(['age']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>fraud</th>
    </tr>
    <tr>
      <th>age</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49.468935</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.622829</td>
      <td>0.026648</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37.228665</td>
      <td>0.028718</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37.279338</td>
      <td>0.023283</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36.197985</td>
      <td>0.035966</td>
    </tr>
    <tr>
      <th>5</th>
      <td>37.547521</td>
      <td>0.023990</td>
    </tr>
    <tr>
      <th>6</th>
      <td>36.700852</td>
      <td>0.022293</td>
    </tr>
    <tr>
      <th>U</th>
      <td>39.117000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Instructions 2/3**

* Count the values of each age group.


```python
banksim_df.age.value_counts()
```




    2    2333
    3    1718
    4    1279
    5     792
    1     713
    6     314
    0      40
    U      11
    Name: age, dtype: int64



**Instructions 3/3**

Based on the results you see, does it make sense to divide your data into age segments before running a fraud detection algorithm?

**Possible Answers**

* **No, the age groups who are the largest are relatively similar.**
* ~~Yes, the age group "0" is very different and I would split that one out.~~

**The average amount spent as well as fraud occurrence is rather similar across groups. Age group '0' stands out but since there are only 40 cases, it does not make sense to split these out in a separate group and run a separate model on them.**

### Using statistics to define normal behavior

In the previous exercises we saw that fraud is **more prevalent in certain transaction categories**, but that there is no obvious way to segment our data into for example age groups. This time, let's investigate the **average amounts spent** in normal transactions versus fraud transactions. This gives you an idea of how fraudulent transactions **differ structurally** from normal transactions.

**Explanation**

* Create two new dataframes from fraud and non-fraud observations. Locate the data in `df` with `.loc` and assign the condition "where fraud is 1" and "where fraud is 0" for creation of the new dataframes.
* Plot the `amount` column of the newly created dataframes in the histogram plot functions and assign the labels `fraud` and `nonfraud` respectively to the plots.


```python
# Create two dataframes with fraud and non-fraud data 
df_fraud = banksim_df[banksim_df.fraud == 1] 
df_non_fraud = banksim_df[banksim_df.fraud == 0]
```


```python
# Plot histograms of the amounts in fraud and non-fraud data 
plt.hist(df_fraud.amount, alpha=0.5, label='fraud')
plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud')
plt.xlabel('amount')
plt.legend()
plt.show()
```


![png](output_174_0.png)


**As the number fraud observations is much smaller, it is difficult to see the full distribution. Nonetheless, you can see that the fraudulent transactions tend to be on the larger side relative to normal observations. This is good news, as it helps us later in detecting fraud from non-fraud. In the next chapter you're going to implement a clustering model to distinguish between normal and abnormal transactions, when the fraud labels are no longer available.**

## Clustering methods to detect fraud

#### K-means clustering

![k-means](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/k-means.JPG)

* The objective of any clustering model is to detect patterns in the data
* More specifically, to group the data into distinct clusters made of data points that are very similar to each other, but distinct from the points in the other clusters.
* **The objective of k-means is to minimize the sum of all distances between the data samples and their associated cluster centroids**
    * The score is the inverse of that minimization, so the score should be close to 0.
* **Using the distance to cluster centroids**
    * Training samples are shown as dots and cluster centroids are shown as crosses
    * Attempt to cluster the data in image A
        * Start by putting in an initial guess for two cluster centroids, as in B
        * Predefine the number of clusters at the start
        * Then calculate the distances of each sample in the data to the closest centroid
        * Figure C shows the data split into the two clusters
        * Based on the initial clusters, the location of the centroids can be redefined (fig D) to minimize the sum of all distances in the two clusters.
        * Repeat the step of reassigning points that are nearest to the centroid (fig E) until it converges to the point where no sample gets reassigned to another cluster (fig F)
        * ![clustering](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/clustering.JPG)

#### K-means clustering in Python

* It's of utmost importance to scale the data before doing K-means clustering, or any algorithm that uses distances
* Without scaling, features on a larger scale will weight more heavily in the algorithm.  All features should weigh equally at the initial stage
* fix `random_state` so models can be compared

```python
# Import the packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Transform and scale your data
X = np.array(df).astype(np.float)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the k-means model and fit to the data
kmeans = KMeans(n_clusters=6, random_state=42).fit(X_scaled)
```

#### The right amount of clusters

* The drawback of K-means clustering is the need to assign the number of clusters beforehand
* There are multiple ways to check what the right number of clusters should be
    * Silhouette method
    * Elbow curve
* By running a k-means model on clusters varying from 1 to 10 and generate an **elbow curve** by saving the scores for each model under "score".
* Plot the scores against the number of clusters

    
```python
clust = range(1, 10) 
kmeans = [KMeans(n_clusters=i) for i in clust]

score = [kmeans[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans))]

plt.plot(clust,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
```

![elbow curve](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/elbow.JPG)

* The slight elbow at 3 means that 3 clusters could be optimal, but it's not very pronounced

### Scaling the data

For ML algorithms using distance based metrics, it is **crucial to always scale your data**, as features using different scales will distort your results. K-means uses the Euclidean distance to assess distance to cluster centroids, therefore you first need to scale your data before continuing to implement the algorithm. Let's do that first.

Available is the dataframe `df` from the previous exercise, with some minor data preparation done so it is ready for you to use with `sklearn`. The fraud labels are separately stored under labels, you can use those to check the results later.

**Explanation**

* Import the ``MinMaxScaler``.
* Transform your dataframe `df` into a numpy array `X` by taking only the values of `df` and make sure you have all `float` values.
* Apply the defined scaler onto `X` to obtain scaled values of `X_scaled` to force all your features to a 0-1 scale.


```python
labels = banksim_adj_df.fraud
```


```python
cols = ['age', 'amount', 'M', 'es_barsandrestaurants', 'es_contents',
        'es_fashion', 'es_food', 'es_health', 'es_home', 'es_hotelservices',
        'es_hyper', 'es_leisure', 'es_otherservices', 'es_sportsandtoys',
        'es_tech', 'es_transportation', 'es_travel']
```


```python
# Take the float values of df for X
X = banksim_adj_df[cols].values.astype(np.float)
```


```python
X.shape
```




    (7189, 17)




```python
# Define the scaler and apply to the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### K-mean clustering

A very commonly used clustering algorithm is **K-means clustering**. For fraud detection, K-means clustering is straightforward to implement and relatively powerful in predicting suspicious cases. It is a good algorithm to start with when working on fraud detection problems. However, fraud data is oftentimes very large, especially when you are working with transaction data. **MiniBatch K-means** is an **efficient way** to implement K-means on a large dataset, which you will use in this exercise.

The scaled data from the previous exercise, `X_scaled` is available. Let's give it a try.

**Explanation**

* Import `MiniBatchKMeans` from `sklearn`.
* Initialize the minibatch kmeans model with 8 clusters.
* Fit the model to your scaled data.


```python
# Define the model 
kmeans = MiniBatchKMeans(n_clusters=8, random_state=0)

# Fit the model to the scaled data
kmeans.fit(X_scaled)
```




    MiniBatchKMeans(batch_size=100, compute_labels=True, init='k-means++',
                    init_size=None, max_iter=100, max_no_improvement=10,
                    n_clusters=8, n_init=3, random_state=0, reassignment_ratio=0.01,
                    tol=0.0, verbose=0)



**You have now fitted your MiniBatch K-means model to the data. In the upcoming exercises you're going to explore whether this model is any good at flagging fraud. But before doing that, you still need to figure our what the right number of clusters to use is. Let's do that in the next exercise.**

### Elbow method

In the previous exercise you've implemented MiniBatch K-means with 8 clusters, without actually checking what the right amount of clusters should be. For our first fraud detection approach, it is important to **get the number of clusters right**, especially when you want to use the outliers of those clusters as fraud predictions. To decide which amount of clusters you're going to use, let's apply the **Elbow method** and see what the optimal number of clusters should be based on this method.

`X_scaled` is again available for you to use and `MiniBatchKMeans` has been imported from `sklearn`.

**Explanation**

* Define the range to be between 1 and 10 clusters.
* Run MiniBatch K-means on all the clusters in the range using list comprehension.
* Fit each model on the scaled data and obtain the scores from the scaled data.
* Plot the cluster numbers and their respective scores.


```python
# Define the range of clusters to try
clustno = range(1, 10)

# Run MiniBatch Kmeans over the number of clusters
kmeans = [MiniBatchKMeans(n_clusters=i) for i in clustno]

# Obtain the score for each model
score = [kmeans[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans))]
```


```python
# Plot the models and their respective score 
plt.plot(clustno, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
```


![png](output_191_0.png)


**Now you can see that the optimal number of clusters should probably be at around 3 clusters, as that is where the elbow is in the curve. We'll use this in the next exercise as our baseline model, and see how well this does in detecting fraud**

## Assigning fraud vs. non-fraud

* ![clusters](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/clusters_1.JPG)
* Take the outliers of each cluster, and flag those as fraud.
* ![clusters](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/clusters_2.JPG)
1. Collect and store the cluster centroids in memory
    * Starting point to decide what's normal and not
1. Calculate the distance of each point in the dataset, to their own cluster centroid
* ![clusters](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/clusters_3.JPG)
    * Euclidean distance is depicted by the circles in this case
    * Define a cut-off point for the distances to define what's an outlier
        * Done based on the distributions of the distances collected
        * i.e. everything with a distance larger than the top 95th percentile, should be considered an outlier
        * the tail of the distribution of distances
        * anything outside the yellow circles is an outlier
        * ![clusters](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/clusters_4.JPG)
        * these are definitely outliers and can be described as abnormal or suspicious
            * doesn't necessarily mean they are fraudulent

#### Flagging Fraud Based on Distance to Centroid

```python
# Run the kmeans model on scaled data
kmeans = KMeans(n_clusters=6, random_state=42,n_jobs=-1).fit(X_scaled)

# Get the cluster number for each datapoint
X_clusters = kmeans.predict(X_scaled)

# Save the cluster centroids
X_clusters_centers = kmeans.cluster_centers_

# Calculate the distance to the cluster centroid for each point
dist = [np.linalg.norm(x-y) for x,y in zip(X_scaled, X_clusters_centers[X_clusters])]

# Create predictions based on distance
km_y_pred = np.array(dist)
km_y_pred[dist>=np.percentile(dist, 93)] = 1
km_y_pred[dist<np.percentile(dist, 93)] = 0
```

* `np.linalg.norm`: returns the vector norm, the vector of distance for each datapoint to their assigned cluster
* use the percentiles of the distances to determine which samples are outliers

#### Validating the Model Results

* without fraud labels, the usual performance metrics can't be run
    * check with the fraud analyst
    * investigate and describe cases that are flagged in more detail
        * is it fraudulent or just a rare case of legit data
        * avoid rare, legit cases by deleting certain features or removing the cases from the data
    * if there are past cases of fraud, see if the model can predict them using historic data

### Detecting outliers

In the next exercises, you're going to use the K-means algorithm to predict fraud, and compare those predictions to the actual labels that are saved, to sense check our results.

The fraudulent transactions are typically flagged as the observations that are furthest aways from the cluster centroid. You'll learn how to do this and how to determine the cut-off in this exercise. In the next one, you'll check the results.

Available are the scaled observations X_scaled, as well as the labels stored under the variable y.

**Explanation**

* Split the scaled data and labels y into a train and test set.
* Define the MiniBatch K-means model with 3 clusters, and fit to the training data.
* Get the cluster predictions from your test data and obtain the cluster centroids.
* Define the boundary between fraud and non fraud to be at 95% of distance distribution and higher.


```python
# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.3, random_state=0)

# Define K-means model 
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42).fit(X_train)

# Obtain predictions and calculate distance from cluster centroid
X_test_clusters = kmeans.predict(X_test)
X_test_clusters_centers = kmeans.cluster_centers_
dist = [np.linalg.norm(x-y) for x, y in zip(X_test, X_test_clusters_centers[X_test_clusters])]

# Create fraud predictions based on outliers on clusters 
km_y_pred = np.array(dist)
km_y_pred[dist >= np.percentile(dist, 95)] = 1
km_y_pred[dist < np.percentile(dist, 95)] = 0
```

### Checking model results

In the previous exercise you've flagged all observations to be fraud, if they are in the top 5th percentile in distance from the cluster centroid. I.e. these are the very outliers of the three clusters. For this exercise you have the scaled data and labels already split into training and test set, so y_test is available. The predictions from the previous exercise, km_y_pred, are also available. Let's create some performance metrics and see how well you did.

**Explanation 1/3**

* Obtain the area under the ROC curve from your test labels and predicted labels.


```python
def plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'],
                          normalize=False,
                          title='Fraud Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-
        examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
```


```python
# Obtain the ROC score
roc_auc_score(y_test, km_y_pred)
```




    0.8109115999408585



**Instructions 2/3**

* Obtain the confusion matrix from the test labels and predicted labels and plot the results.


```python
# Create a confusion matrix
km_cm = confusion_matrix(y_test, km_y_pred)

# Plot the confusion matrix in a figure to visualize results 
plot_confusion_matrix(km_cm)
```

    Confusion matrix, without normalization



![png](output_202_1.png)


**Instructions 3/3**

If you were to decrease the percentile used as a cutoff point in the previous exercise to 93% instead of 95%, what would that do to your prediction results?

**Possible Answers**

* **The number of fraud cases caught increases, but false positives also increase.**
* ~~The number of fraud cases caught decreases, and false positives decrease.~~
* ~~The number of fraud cases caught increases, but false positives would decrease.~~
* ~~Nothing would happen to the amount of fraud cases caught.~~

## Alternate clustering methods for fraud detection

* In addition to K-means, there are many different clustering methods, which can be used for fraud detection
* ![clustering methods](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/clustering_methods.JPG)
* K-means works well when the data is clustered in normal, round shapes
* There are methods to flag fraud other the cluster outliers
* ![clustering outlier](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/cluster_outlier.JPG)
    * Small clusters can be an indication of fraud
    * This approach can be used when fraudulent behavior has commonalities, which cause clustering
    * The fraudulent data would cluster in tiny groups, rather than be the outliers of larger clusters
* ![typical data](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/typical_data.JPG)
    * In this case there are 3 obvious clusters
    * The smallest dots are outliers and outside of what can be described as normal behavior
    * There are also small to medium clusters closely connected to the red cluster
    * Visualizing the data with something like [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) can be quite helpful

#### DBSCAN: Density-Based Spatial Clustering of Applications with Noise

* [DBscan](https://en.wikipedia.org/wiki/DBSCAN)
* DBSCAN vs. K-means
    * The number of clusters does not need to be predefined
        * The algorithm finds core samples of high density and expands clusters from them
        * Works well on data containing clusters of similar density
    * This type of algorithm can be used to identify fraud as very small clusters
    * Maximum allowed distance between points in a cluster must be assigned
    * Minimal number of data points in clusters must be assigned
    * Better performance on weirdly shaped data
    * Computationally heavier then MiniBatch K-means

#### Implementation of DBSCAN

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=10, n_jobs=-1).fit(X_scaled)

# Get the cluster labels (aka numbers)
pred_labels = db.labels_

# Count the total number of clusters
n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)

# Print model results
print(f'Estimated number of clusters: {n_clusters_}')
>>> Estimated number of clusters: 31
    
# Print model results
print(f'Silhouette Coefficient: {metrics.silhouette_score(X_scaled, pred_labels):0.3f}')
>>> Silhouette Coefficient: 0.359
    
# Get sample counts in each cluster 
counts = np.bincount(pred_labels[pred_labels>=0])
print(counts)
>>> [ 763, 496, 840, 355 1086, 676, 63, 306, 560, 134, 28, 18, 262, 128,
     332, 22, 22, 13, 31, 38, 36, 28, 14, 12, 30, 10, 11, 10, 21, 10, 5]
```

* start by defining the epsilon `eps`
    * Distance between data points allowed from which the cluster expands
* define minimum samples in the clusters
* conventional DBSCAN can't produce the optimal value of epsilon, so it requires sophisticated DBSCAN modifications to automatically determine the optimal epsilon value
* Fit DBSCAN to **scaled data**
* Use `labels_` method to get the assigned cluster label for each data point
* The cluster count can also be determine by counting the unique cluster labels from the cluster `label_` predictions
* Can have performance metrics such as **average silhouette score**
* The size of each cluster can be calculated with `np.bincount`
    * counts the number of occurrences of non-negative values in a `numpy` array
* sort `counts` and decide how many of the smaller clusters to flag as fraud
    * selecting the clusters to flag, is a trial-and-error step and depends on the number of cases the fraud team can manage

### DB scan

In this exercise you're going to explore using a **density based clustering** method (DBSCAN) to detect fraud. The advantage of DBSCAN is that you **do not need to define the number of clusters** beforehand. Also, DBSCAN can handle weirdly shaped data (i.e. non-convex) much better than K-means can. This time, you are not going to take the outliers of the clusters and use that for fraud, but take the **smallest clusters** in the data and label those as fraud. You again have the scaled dataset, i.e. X_scaled available. Let's give it a try!

**Explanation**

* Import `DBSCAN`.
* Initialize a DBSCAN model setting the maximum distance between two samples to 0.9 and the minimum observations in the clusters to 10, and fit the model to the scaled data.
* Obtain the predicted labels, these are the cluster numbers assigned to an observation.
* Print the number of clusters and the rest of the performance metrics.


```python
# Initialize and fit the DBscan model
db = DBSCAN(eps=0.9, min_samples=10, n_jobs=-1).fit(X_scaled)

# Obtain the predicted labels and calculate number of clusters
pred_labels = db.labels_
n_clusters = len(set(pred_labels)) - (1 if -1 in labels else 0)
```


```python
# Print performance metrics for DBscan
print(f'Estimated number of clusters: {n_clusters}')
print(f'Homogeneity: {homogeneity_score(labels, pred_labels):0.3f}')
print(f'Silhouette Coefficient: {silhouette_score(X_scaled, pred_labels):0.3f}')
```

    Estimated number of clusters: 23
    Homogeneity: 0.612
    Silhouette Coefficient: 0.713


**The number of clusters is much higher than with K-means. For fraud detection this is for now OK, as we are only interested in the smallest clusters, since those are considered as abnormal. Now have a look at those clusters and decide which one to flag as fraud.**

### Assessing smallest clusters

In this exercise you're going to have a look at the clusters that came out of DBscan, and flag certain clusters as fraud:

* you first need to figure out how big the clusters are, and **filter out the smallest**
* then, you're going to take the smallest ones and **flag those as fraud**
* last, you'll **check with the original labels** whether this does actually do a good job in detecting fraud.

Available are the DBscan model predictions, so `n_clusters` is available as well as the cluster labels, which are saved under `pred_labels`. Let's give it a try!

**Explanation 1/3**

* Count the samples within each cluster by running a bincount on the predicted cluster numbers under `pred_labels` and print the results.


```python
# Count observations in each cluster number
counts = np.bincount(pred_labels[pred_labels >= 0])

# Print the result
print(counts)
```

    [3252  145 2714   55  174  119  122   98   54   15   76   15   43   25
       51   47   42   15   25   20   19   10]


**Instructions 2/3**

* Sort the sample `counts` and take the top 3 smallest clusters, and print the results.


```python
# Sort the sample counts of the clusters and take the top 3 smallest clusters
smallest_clusters = np.argsort(counts)[:3]
```


```python
# Print the results 
print(f'The smallest clusters are clusters: {smallest_clusters}')
```

    The smallest clusters are clusters: [21 17  9]


**Instructions 3/3**

* Within `counts`, select the smallest clusters only, to print the number of samples in the three smallest clusters.


```python
# Print the counts of the smallest clusters only
print(f'Their counts are: {counts[smallest_clusters]}')
```

    Their counts are: [10 15 15]


**So now we know which smallest clusters you could flag as fraud. If you were to take more of the smallest clusters, you cast your net wider and catch more fraud, but most likely also more false positives. It is up to the fraud analyst to find the right amount of cases to flag and to investigate. In the next exercise you'll check the results with the actual labels.**

### Results verification

In this exercise you're going to **check the results** of your DBscan fraud detection model. In reality, you often don't have reliable labels and this where a fraud analyst can help you validate the results. He/She can check your results and see whether the cases you flagged are indeed suspicious. You can also **check historically known cases** of fraud and see whether your model flags them.

In this case, you'll **use the fraud labels** to check your model results. The predicted cluster numbers are available under `pred_labels` as well as the original fraud `labels`.

**Explanation**

* Create a dataframe combining the cluster numbers with the actual labels.
* Create a condition that flags fraud for the three smallest clusters: clusters 21, 17 and 9.
* Create a crosstab from the actual fraud labels with the newly created predicted fraud labels.


```python
# Create a dataframe of the predicted cluster numbers and fraud labels 
df = pd.DataFrame({'clusternr':pred_labels,'fraud':labels})

# Create a condition flagging fraud for the smallest clusters 
df['predicted_fraud'] = np.where((df['clusternr'].isin([21, 17, 9])), 1 , 0)
```


```python
# Run a crosstab on the results 
print(pd.crosstab(df['fraud'], df['predicted_fraud'], rownames=['Actual Fraud'], colnames=['Flagged Fraud']))
```

    Flagged Fraud     0   1
    Actual Fraud           
    0              6973  16
    1               176  24


**How does this compare to the K-means model? The good thing is: our of all flagged cases, roughly 2/3 are actually fraud! Since you only take the three smallest clusters, by definition you flag less cases of fraud, so you catch less but also have less false positives. However, you are missing quite a lot of fraud cases. Increasing the amount of smallest clusters you flag could improve that, at the cost of more false positives of course. In the next chapter you'll learn how to further improve fraud detection models by including text analysis.**

# Fraud detection using text

Use text data, text mining and topic modeling to detect fraudulent behavior.

## Using text data

* Types of useful text data:
    1. Emails from employees and/or clients
    1. Transaction descriptions
    1. Employee notes
    1. Insurance claim form description box
    1. Recorded telephone conversations
* Text mining techniques for fraud detection
    1. Word search
    1. Sentiment analysis
    1. Word frequencies and topic analysis
    1. Style
* Word search for fraud detection
    * Flagging suspicious words:
        1. Simple, straightforward and easy to explain
        1. Match results can be used as a filter on top of machine learning model
        1. Match results can be used as a feature in a machine learning model

#### Word counts to flag fraud with pandas

```python
# Using a string operator to find words
df['email_body'].str.contains('money laundering')

 # Select data that matches 
df.loc[df['email_body'].str.contains('money laundering', na=False)]

 # Create a list of words to search for
list_of_words = ['police', 'money laundering']
df.loc[df['email_body'].str.contains('|'.join(list_of_words), na=False)]

 # Create a fraud flag 
df['flag'] = np.where((df['email_body'].str.contains('|'.join(list_of_words)) == True), 1, 0)
```

### Word search with dataframes

In this exercise you're going to work with text data, containing emails from Enron employees. The **Enron scandal** is a famous fraud case. Enron employees covered up the bad financial position of the company, thereby keeping the stock price artificially high. Enron employees sold their own stock options, and when the truth came out, Enron investors were left with nothing. The goal is to find all emails that mention specific words, such as "sell enron stock".

By using string operations on dataframes, you can easily sift through messy email data and create flags based on word-hits. The Enron email data has been put into a dataframe called `df` so let's search for suspicious terms. Feel free to explore `df` in the Console before getting started.

**Instructions 1/2**

* Check the head of `df` in the console and look for any emails mentioning 'sell enron stock'.


```python
df = pd.read_csv(enron_emails_clean_file)
```


```python
mask = df['clean_content'].str.contains('sell enron stock', na=False)
```

**Instructions 2/2**

* Locate the data in `df` that meets the condition we created earlier.


```python
# Select the data from df using the mask
df[mask]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Message-ID</th>
      <th>From</th>
      <th>To</th>
      <th>Date</th>
      <th>content</th>
      <th>clean_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>154</th>
      <td>&lt;6336501.1075841154311.JavaMail.evans@thyme&gt;</td>
      <td>('sarah.palmer@enron.com')</td>
      <td>('sarah.palmer@enron.com')</td>
      <td>2002-02-01 14:53:35</td>
      <td>\nJoint Venture: A 1997 Enron Meeting Belies O...</td>
      <td>joint venture enron meeting belies officers cl...</td>
    </tr>
  </tbody>
</table>
</div>



**You see that searching for particular string values in a dataframe can be relatively easy, and allows you to include textual data into your model or analysis. You can use this word search as an additional flag, or as a feature in your fraud detection model. Let's look at how to filter the data using multiple search terms.**

### Using list of terms

Oftentimes you don't want to search on just one term. You probably can create a full **"fraud dictionary"** of terms that could potentially **flag fraudulent clients** and/or transactions. Fraud analysts often will have an idea what should be in such a dictionary. In this exercise you're going to **flag a multitude of terms**, and in the next exercise you'll create a new flag variable out of it. The 'flag' can be used either directly in a machine learning model as a feature, or as an additional filter on top of your machine learning model results. Let's first use a list of terms to filter our data on. The dataframe containing the cleaned emails is again available as `df`.

**Instructions**

* Create a list to search for including 'enron stock', 'sell stock', 'stock bonus', and 'sell enron stock'.
* Join the string terms in the search conditions.
* Filter data using the emails that match with the list defined under `searchfor`.


```python
# Create a list of terms to search for
searchfor = ['enron stock', 'sell stock', 'stock bonus', 'sell enron stock']

# Filter cleaned emails on searchfor list and select from df 
filtered_emails = df[df.clean_content.str.contains('|'.join(searchfor), na=False)]
filtered_emails.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Message-ID</th>
      <th>From</th>
      <th>To</th>
      <th>Date</th>
      <th>content</th>
      <th>clean_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;8345058.1075840404046.JavaMail.evans@thyme&gt;</td>
      <td>('advdfeedback@investools.com')</td>
      <td>('advdfeedback@investools.com')</td>
      <td>2002-01-29 23:20:55</td>
      <td>INVESTools Advisory\nA Free Digest of Trusted ...</td>
      <td>investools advisory free digest trusted invest...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;1512159.1075863666797.JavaMail.evans@thyme&gt;</td>
      <td>('richard.sanders@enron.com')</td>
      <td>('richard.sanders@enron.com')</td>
      <td>2000-09-20 19:07:00</td>
      <td>----- Forwarded by Richard B Sanders/HOU/ECT o...</td>
      <td>forwarded richard b sanders hou ect pm justin ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;26118676.1075862176383.JavaMail.evans@thyme&gt;</td>
      <td>('m..love@enron.com')</td>
      <td>('m..love@enron.com')</td>
      <td>2001-10-30 16:15:17</td>
      <td>hey you are not wearing your target purple shi...</td>
      <td>hey wearing target purple shirt today mine wan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;10369289.1075860831062.JavaMail.evans@thyme&gt;</td>
      <td>('leslie.milosevich@kp.org')</td>
      <td>('leslie.milosevich@kp.org')</td>
      <td>2002-01-30 17:54:18</td>
      <td>Leslie Milosevich\n1042 Santa Clara Avenue\nAl...</td>
      <td>leslie milosevich santa clara avenue alameda c...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;26728895.1075860815046.JavaMail.evans@thyme&gt;</td>
      <td>('rtwait@graphicaljazz.com')</td>
      <td>('rtwait@graphicaljazz.com')</td>
      <td>2002-01-30 19:36:01</td>
      <td>Rini Twait\n1010 E 5th Ave\nLongmont, CO 80501...</td>
      <td>rini twait e th ave longmont co rtwait graphic...</td>
    </tr>
  </tbody>
</table>
</div>



**By joining the search terms with the 'or' sign, i.e. |, you can search on a multitude of terms in your dataset very easily. Let's now create a flag from this which you can use as a feature in a machine learning model.**

### Creating a flag

This time you are going to **create an actual flag** variable that gives a **1 when the emails get a hit** on the search terms of interest, and 0 otherwise. This is the last step you need to make in order to actually use the text data content as a feature in a machine learning model, or as an actual flag on top of model results. You can continue working with the dataframe `df` containing the emails, and the `searchfor` list is the one defined in the last exercise.

**Instructions**

* Use a numpy where condition to flag '1' where the cleaned email contains words on the `searchfor` list and 0 otherwise.
* Join the words on the `searchfor` list with an "or" indicator.
* Count the values of the newly created flag variable.


```python
# Create flag variable where the emails match the searchfor terms
df['flag'] = np.where((df['clean_content'].str.contains('|'.join(searchfor)) == True), 1, 0)

# Count the values of the flag variable
count = df['flag'].value_counts()
print(count)
```

    0    1776
    1     314
    Name: flag, dtype: int64


**You have now managed to search for a list of strings in several lines of text data. These skills come in handy when you want to flag certain words based on what you discovered in your topic model, or when you know beforehand what you want to search for. In the next exercises you're going to learn how to clean text data and to create your own topic model to further look for indications of fraud in your text data.**

## Text mining to detect fraud

#### Cleaning your text data

**Must dos when working with textual data:**

1. Tokenization
    * Split the text into sentences and the sentences in words
    * transform everything to lowercase
    * remove punctuation
1. Remove all stopwords
1. Lemmatize 
    * change from third person into first person
    * change past and future tense verbs to present tense
    * this makes it possible to combine all words that point to the same thing
1. Stem the words
    * reduce words to their root form
    * e.g. walking and walked to walk

* **Unprocessed Text**
    * ![](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/text_df.JPG)
* **Processed Text**
    * ![](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/text_processed.JPG)

#### Data Preprocessing I

* Tokenizers divide strings into list of substrings
* nltk word tokenizer can be used to find the words and punctuation in a string
    * it splits the words on whitespace, and separated the punctuation out

```python
from nltk import word_tokenize
from nltk.corpus import stopwords 
import string

# 1. Tokenization
text = df.apply(lambda row: word_tokenize(row["email_body"]), axis=1)
text = text.rstrip()  # remove whitespace
# replace with lowercase
# text = re.sub(r'[^a-zA-Z]', ' ', text)
text = text.lower()

 # 2. Remove all stopwords and punctuation
exclude = set(string.punctuation)
stop = set(stopwords.words('english'))
stop_free = " ".join([word for word in text if((word not in stop) and (not word.isdigit()))])
punc_free = ''.join(word for word in stop_free if word not in exclude)
```

#### Data Preprocessing II

```python
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Lemmatize words
lemma = WordNetLemmatizer()
normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

# Stem words
porter= PorterStemmer()
cleaned_text = " ".join(porter.stem(token) for token in normalized.split())
print (cleaned_text)

['philip','going','street','curious','hear','perspective','may','wish',
'offer','trading','floor','enron','stock','lower','joined','company',
'business','school','imagine','quite','happy','people','day','relate',
'somewhat','stock','around','fact','broke','day','ago','knowing',
'imagine','letting','event','get','much','taken','similar',
'problem','hope','everything','else','going','well','family','knee',
'surgery','yet','give','call','chance','later']
```

### Removing stopwords

In the following exercises you're going to **clean the Enron emails**, in order to be able to use the data in a topic model. Text cleaning can be challenging, so you'll learn some steps to do this well. The dataframe containing the emails `df` is available. In a first step you need to **define the list of stopwords and punctuations** that are to be removed in the next exercise from the text data. Let's give it a try.

**Instructions**

* Import the stopwords from `ntlk`.
* Define 'english' words to use as stopwords under the variable `stop`.
* Get the punctuation set from the `string` package and assign it to `exclude`.


```python
# Define stopwords to exclude

nltk.download('stopwords')

stop = set(stopwords.words('english'))
stop.update(("to", "cc", "subject", "http", "from", "sent", "ect", "u", "fwd", "www", "com", 'html'))

# Define punctuations to exclude and lemmatizer
exclude = set(string.punctuation)
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.


### Cleaning text data

Now that you've defined the **stopwords and punctuations**, let's use these to clean our enron emails in the dataframe `df` further. The lists containing stopwords and punctuations are available under `stop` and `exclude` There are a few more steps to take before you have cleaned data, such as **"lemmatization"** of words, and **stemming the verbs**. The verbs in the email data are already stemmed, and the lemmatization is already done for you in this exercise.

**Instructions 1/2**

* Use the previously defined variables `stop` and `exclude` to finish of the function: Strip the words from whitespaces using `rstrip`, and exclude stopwords and punctuations. Finally lemmatize the words and assign that to `normalized`.


```python
# Import the lemmatizer from nltk
lemma = WordNetLemmatizer()

def clean(text, stop):
    text = str(text).rstrip()
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and (not i.isdigit()))])
    punc_free = ''.join(i for i in stop_free if i not in exclude)
    normalized = " ".join(lemma.lemmatize(i) for i in punc_free.split())      
    return normalized
```

**Instructions 2/2**

* Apply the function `clean(text,stop)` on each line of text data in our dataframe, and take the column `df['clean_content']` for this.


```python
# Clean the emails in df and print results

nltk.download('wordnet')

text_clean=[]
for text in df['clean_content']:
    text_clean.append(clean(text, stop).split())    
```

    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Unzipping corpora/wordnet.zip.



```python
text_clean[0][:10]
```




    ['investools',
     'advisory',
     'free',
     'digest',
     'trusted',
     'investment',
     'advice',
     'unsubscribe',
     'free',
     'newsletter']



**Now that you have cleaned your data entirely with the necessary steps, including splitting the text into words, removing stopwords and punctuations, and lemmatizing your words. You are now ready to run a topic model on this data. In the following exercises you're going to explore how to do that.**

## Topic modeling on fraud

1. Discovering topics in text data
1. "What is the text about"
1. Conceptually similar to clustering data
1. Compare topics of fraud cases to non-fraud cases and use as a feature or flag
1. Or.. is there a particular topic in the data that seems to point to fraud?

#### Latent Dirichlet Allocation (LDA)

* With LDA you obtain:
    * "topics per text item" model (i.e. probabilities)
    * "words per topic" model
* Creating your own topic model:
    * Clean your data
    * Create a bag of words with dictionary and corpus
        * Dictionary contain words and word frequency from the entire text
        * Corpus: word count for each line of text
    * Feed dictionary and corpus into the LDA model
* LDA:
    * ![lda](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/lda.JPG)
    1. [LDA2vec: Word Embeddings in Topic Models](https://www.datacamp.com/community/tutorials/lda2vec-topic-model)
    1. see how each word in the dataset is associated with each topic
    1. see how each text item in the data associates with topics (in the form of probabilities)
        1. image on the right 

#### Bag of words: dictionary and corpus

* use the `Dictionary` function in `corpora` to create a `dict` from the text data
    * contains word counts
* filter out words that appear in less than 5 emails and keep only the 50000 most frequent words
    * this is a way of cleaning the outlier noise
* create the corpus, which for each email, counts the number of words and the count for each word (`doc2bow`)
* `doc2bow`
    * Document to Bag of Words
    * converts text data into bag-of-words format
    * each row is now a list of words with the associated word count
    
```python
from gensim import corpora

 # Create dictionary number of times a word appears
dictionary = corpora.Dictionary(cleaned_emails)

# Filter out (non)frequent words 
dictionary.filter_extremes(no_below=5, keep_n=50000)

# Create corpus
corpus = [dictionary.doc2bow(text) for text in cleaned_emails]
```

#### Latent Dirichlet Allocation (LDA) with gensim

* Run the LDA model after cleaning the text date, and creating the dictionary and corpus
* Pass the corpus and dictionary into the model
* As with K-means, beforehand, pick the number of topics to obtain, even if there is uncertainty about what topics exist
* The calculated LDA model, will contain the associated words for each topic, and topic scores per email
* Use `print_topics` to obtain the top words from the topics

```python
import gensim

# Define the LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, 
id2word=dictionary, passes=15)

# Print the three topics from the model with top words
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

>>> (0, '0.029*"email" + 0.016*"send" + 0.016*"results" + 0.016*"invoice"')
>>> (1, '0.026*"price" + 0.026*"work" + 0.026*"management" + 0.026*"sell"')
>>> (2, '0.029*"distribute" + 0.029*"contact" + 0.016*"supply" + 0.016*"fast"')
```

### Create dictionary and corpus

In order to run an LDA topic model, you first need to **define your dictionary and corpus** first, as those need to go into the model. You're going to continue working on the cleaned text data that you've done in the previous exercises. That means that `text_clean` is available for you already to continue working with, and you'll use that to create your dictionary and corpus.

This exercise will take a little longer to execute than usual.

**Instructions**

* Import the gensim package and corpora from gensim separately.
* Define your dictionary by running the correct function on your clean data `text_clean`.
* Define the corpus by running `doc2bow` on each piece of text in `text_clean`.
* Print your results so you can see `dictionary` and `corpus` look like.


```python
# Define the dictionary
dictionary = corpora.Dictionary(text_clean)

# Define the corpus 
corpus = [dictionary.doc2bow(text) for text in text_clean]
```


```python
print(dictionary)
```

    Dictionary(33980 unique tokens: ['account', 'accurate', 'acquiring', 'acre', 'address']...)



```python
corpus[0][:10]
```




    [(0, 2),
     (1, 1),
     (2, 1),
     (3, 1),
     (4, 1),
     (5, 6),
     (6, 1),
     (7, 2),
     (8, 4),
     (9, 1)]



**These are the two ingredients you need to run your topic model on the enron emails. You are now ready for the final step and create your first fraud detection topic model.**

### LDA model

Now it's time to **build the LDA model**. Using the `dictionary` and `corpus`, you are ready to discover which topics are present in the Enron emails. With a quick print of words assigned to the topics, you can do a first exploration about whether there are any obvious topics that jump out. Be mindful that the topic model is **heavy to calculate** so it will take a while to run. Let's give it a try!

**Instructions**

* Build the LDA model from gensim models, by inserting the `corpus` and `dictionary`.
* Save the 5 topics by running `print_topics` on the model results, and select the top 5 words.


```python
# Define the LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=5)

# Save the topics and top 5 words
topics = ldamodel.print_topics(num_words=5)

# Print the results
for topic in topics:
    print(topic)
```

    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)


    (0, '0.035*"td" + 0.035*"image" + 0.026*"net" + 0.024*"money" + 0.023*"tr"')
    (1, '0.023*"enron" + 0.011*"market" + 0.009*"company" + 0.008*"employee" + 0.008*"energy"')
    (2, '0.015*"enron" + 0.007*"company" + 0.005*"said" + 0.005*"bakernet" + 0.004*"e"')
    (3, '0.039*"enron" + 0.013*"pm" + 0.012*"message" + 0.010*"original" + 0.008*"e"')
    (4, '0.016*"enron" + 0.009*"hou" + 0.008*"please" + 0.006*"e" + 0.006*"amazon"')


    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
    /usr/local/lib/python3.6/dist-packages/gensim/models/ldamodel.py:1077: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)


**You have now successfully created your first topic model on the Enron email data. However, the print of words doesn't really give you enough information to find a topic that might lead you to signs of fraud. You'll therefore need to closely inspect the model results in order to be able to detect anything that can be related to fraud in your data. You'll learn more about this in the next video.**

## Flagging fraud based on topic

#### Using your LDA model results for fraud detection

1. Are there any suspicious topics? (no labels)
    1. if you don't have labels, first check for the frequency of suspicious words within topics and check whether topics seem to describe the fraudulent behavior
    1. for the Enron email data, a suspicious topic would be one where employees are discussing stock bonuses, selling stock, stock price, and perhaps mentions of accounting or weak financials
    1. Defining suspicious topics does require some pre-knowledge about the fraudulent behavior
    1. If the fraudulent topic is noticeable, flag all instances that have a high probability for this topic
1. Are the topics in fraud and non-fraud cases similar? (with labels)
    1. If there a previous cases of fraud, ran a topic model on the fraud text only, and on the non-fraud text
    1. Check whether the results are similar
        1. Whether the frequency of the topics are the same in fraud vs non-fraud
1. Are fraud cases associated more with certain topics? (with labels)
    1. Check whether fraud cases have a higher probability score for certain topics
        1. If so, run a topic model on new data and create a flag directly on the instances that score high on those topics

#### To understand topics, you need to visualize

```python
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
```

![topics](https://raw.githubusercontent.com/trenton3983/DataCamp/master/Images/fraud_detection/topics2.jpg)

* Each bubble on the left-hand side, represents a topic
* The larger the bubble, the more prevalent that topic is
* Click on each topic to get the details per topic in the right panel
* The words are the most important keywords that form the selected topic.
* A good topic model will have fairly big, non-overlapping bubbles, scattered throughout the chart
* A model with too many topics, will typically have many overlaps, or small sized bubbles, clustered in one region
* In the case of the model above, there is a slight overlap between topic 2 and 3, which may point to 1 topic too many


```python
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
```


```python
pyLDAvis.display(lda_display)
```





<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el61140356880911240231555628"></div>
<script type="text/javascript">

var ldavis_el61140356880911240231555628_data = {"mdsDat": {"x": [-0.31713042171589706, 0.07663521902196455, 0.047413898309582585, 0.09567214136251291, 0.09740916302183716], "y": [0.020188413028788498, -0.14881063282295054, -0.10929371651256399, 0.10745731742118034, 0.13045861888554583], "topics": [1, 2, 3, 4, 5], "cluster": [1, 1, 1, 1, 1], "Freq": [3.873427461463814, 21.543073670008567, 49.6405200905335, 17.609586048889486, 7.333392729104625]}, "tinfo": {"Term": ["enron", "image", "td", "net", "money", "pm", "message", "tr", "hou", "original", "width", "please", "class", "employee", "market", "e", "development", "thanks", "right", "energy", "know", "table", "mail", "conference", "height", "f", "click", "corp", "bakernet", "team", "width", "td", "img", "src", "href", "align", "nbsp", "scoop", "bodydefault", "wj", "height", "cellspacing", "cellpadding", "valign", "script", "tr", "bgcolor", "colspan", "stocklookup", "rect", "coords", "syncrasy", "ctr", "deviation", "font", "linkbarseperator", "ffffff", "std", "npcc", "ecar", "br", "wscc", "map", "gif", "rk", "sw", "table", "sp", "hp", "nc", "matrix", "border", "class", "image", "net", "money", "ne", "se", "clear", "click", "center", "right", "euci", "donate", "brochure", "congestion", "pocketbook", "netted", "underhanded", "astronomical", "financially", "repair", "wiped", "anc", "wyndham", "transition", "declared", "competitive", "nerc", "reassigned", "dun", "mpc", "svc", "mws", "emerging", "maximize", "tepc", "coronado", "curtailment", "denver", "relieve", "aps", "ref", "aggressively", "devastated", "afford", "design", "bankrupt", "seller", "hurt", "status", "conference", "obtained", "consumer", "transmission", "market", "fund", "customer", "saving", "asset", "employee", "generation", "pdf", "link", "capacity", "t", "million", "service", "management", "retirement", "energy", "process", "california", "trading", "price", "enron", "time", "company", "made", "power", "bill", "stock", "business", "please", "new", "also", "plan", "year", "bakernet", "classmate", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "preferred", "worker", "detected", "awarded", "alert", "locate", "interchange", "assign", "nytimes", "pension", "todaysheadlines", "andersen", "labour", "employer", "skadden", "fx", "srrs", "tape", "emaillink", "itcapps", "redshirt", "military", "circuit", "tyco", "auditor", "auth", "txt", "partnership", "westdesk", "parsing", "court", "bush", "former", "said", "house", "sept", "federal", "sec", "act", "board", "law", "government", "school", "schedule", "member", "copyright", "share", "friend", "executive", "investor", "mailto", "committee", "date", "game", "even", "c", "mr", "say", "company", "year", "one", "news", "new", "enron", "stock", "two", "e", "see", "b", "dynegy", "would", "financial", "day", "u", "business", "may", "time", "make", "final", "energy", "get", "image", "power", "whalley", "exotica", "sat", "fri", "savita", "beth", "bergfelt", "backout", "tagline", "laurel", "xll", "krishna", "fran", "stwbom", "stsw", "kern", "eix", "gregwhalley", "pt", "pcg", "enform", "addin", "pinnamaneni", "krishnarao", "headcount", "veggie", "vegetarian", "fagan", "lover", "permian", "wolfe", "pseg", "tyrell", "hr", "greg", "server", "impacted", "cherry", "original", "pm", "thru", "tonight", "outage", "development", "fw", "tycholiz", "eol", "thanks", "message", "scheduled", "know", "november", "let", "recipient", "intended", "enron", "corp", "mail", "hou", "going", "london", "mark", "desk", "contact", "e", "like", "call", "team", "please", "thursday", "may", "would", "get", "time", "go", "devon", "sfs", "darrell", "schoolcraft", "hanagriff", "rfp", "bradford", "eei", "caiso", "neumin", "commoditylogic", "kowalke", "reliantenergy", "agave", "powersrc", "lokay", "dietz", "dbcaps", "royalty", "emailed", "graf", "certificate", "epenergy", "pampa", "chanley", "raquel", "simulation", "brackett", "lynn", "pnm", "amazon", "coral", "gift", "tw", "blair", "buchanan", "shipping", "apse", "stack", "leg", "enronxgate", "master", "doc", "hou", "kimberly", "january", "team", "please", "request", "richard", "available", "enron", "gas", "thanks", "credit", "e", "kim", "message", "mail", "pm", "energy", "john", "review", "order", "original", "product", "attached", "know", "would"], "Freq": [8966.0, 987.0, 599.0, 693.0, 642.0, 1263.0, 1434.0, 397.0, 918.0, 970.0, 350.0, 1342.0, 350.0, 1156.0, 1332.0, 1891.0, 715.0, 749.0, 505.0, 1350.0, 851.0, 248.0, 819.0, 638.0, 223.0, 454.0, 357.0, 1086.0, 1050.0, 538.0, 349.1910836505809, 596.8950865691726, 174.31215090267062, 185.2706864494839, 157.34935273046173, 178.96245766069427, 130.95132677358285, 110.30894992760972, 92.49118923698222, 82.48089699854754, 219.85800035952704, 76.77565165806114, 76.6871367494715, 48.33222717169132, 178.60558146205344, 387.38458643512257, 43.06242346489007, 39.5522682934636, 37.709045771587654, 37.70904179338952, 37.70904179338952, 61.54239103951399, 61.54239103951399, 34.45727456928011, 77.26419030230565, 30.86127783841333, 24.452708947066313, 34.210805303587065, 34.210805303587065, 34.210805303587065, 197.14121596323443, 100.03029639912384, 39.82514881660053, 168.30615378288644, 68.41306543757186, 69.991948627715, 211.178447224126, 107.12036046814355, 69.81331957488393, 67.04213062054319, 65.93602047542025, 129.75397665414516, 251.8940480647419, 593.9310425254563, 439.57918252866244, 406.33223496114897, 107.68827212165459, 115.48393328177126, 129.3443734173719, 118.73986567933204, 94.94741628827319, 91.97302097299949, 170.40982707430655, 191.57564981506, 125.87333625136608, 111.22632233465129, 94.48210501092446, 97.53004225167267, 93.85348938937246, 94.4670816042498, 97.50967546839723, 96.86699891173218, 96.30103244481833, 73.92740746024097, 73.92604119313911, 199.47572702184647, 192.42570027313633, 91.43319976716886, 78.73286202409435, 51.63718561478016, 51.623556132436136, 49.82455296615719, 49.798964498087614, 52.16912815884577, 59.56561701251447, 52.77242635317027, 48.5213720031539, 48.5213720031539, 50.94754155219665, 61.25858789508701, 48.44074564980711, 48.44074564980711, 159.54121099916068, 98.14409910799357, 94.98354163164296, 96.57868889601174, 150.9078964057837, 102.13377605167791, 215.93326057490526, 105.80091439330057, 296.59205031960727, 565.9781935788648, 118.53569644910083, 352.6060366407864, 225.41269829987147, 1013.358030491099, 433.4667020298177, 448.2422999345882, 204.34355416805016, 322.73616317832176, 788.5433478548546, 185.05904250643243, 239.67303096953216, 272.4706765451307, 200.85965049855744, 146.55509111107645, 532.2195588690975, 642.502382501062, 407.7167599957116, 201.27804905557932, 738.0512932034771, 297.77748339679187, 388.9955597070446, 397.2424585627638, 426.28210626874335, 2144.9774862249083, 666.4158845440091, 842.0622538757564, 364.2372731423813, 499.5192446130303, 383.9935396867881, 442.4709664385173, 369.7123645295169, 451.15675232634044, 475.3526236122656, 321.3439646587591, 278.11529568181163, 294.6717757365627, 1048.581781852642, 325.98439682013117, 403.1928577674949, 271.44386804500095, 243.93362561633134, 144.71893103104156, 143.650564800777, 334.7431874546441, 112.9396592219077, 110.24051805818458, 113.77589882775145, 136.31091773183974, 107.3854829375489, 92.9868387086971, 183.97329959271255, 88.43426612117895, 115.21880057627902, 88.35521025757419, 96.42451012713231, 80.24958484004772, 81.08631515943419, 75.70067667184216, 75.70040900990976, 71.07215182455442, 70.04974059892139, 71.80762945865652, 62.98493135133192, 65.53733641997243, 89.66373934313468, 141.37111754705438, 239.3208928230551, 134.3114958426309, 134.31038695748242, 253.41760164635534, 232.8917041900737, 177.2737851538152, 1113.1045202262462, 240.4021450634937, 210.20915257800877, 180.01641500864991, 181.17382344175897, 169.5674030000734, 337.3665293082733, 281.6820899089107, 250.16456597804972, 169.94284347058402, 656.6966210644699, 461.5713272622525, 200.97470119801147, 473.44786957695317, 182.11283248383734, 335.3616394681598, 326.78246273609244, 636.2906361539027, 269.21727765002385, 626.8633789619346, 240.82097225866548, 252.71607246734575, 517.5047177503582, 495.7449752605583, 420.33971451434286, 1468.925014209525, 763.9629816962953, 618.0379277604482, 419.52454372632076, 945.4077427426728, 3285.450395033864, 712.8347092018287, 374.99829299832703, 969.7171563256608, 498.59254133869945, 385.40965389507954, 333.67888713659073, 618.770046874589, 377.31599041677003, 523.6310660255403, 485.07994732673006, 461.20970324577615, 471.67482595352175, 554.7399594062732, 387.61123701808685, 386.9480472244418, 437.69618879167757, 390.7284278827972, 389.07478697282806, 388.08864393903497, 197.28322379470615, 98.1729700517332, 208.1233697891942, 218.1398658033112, 86.08089091751935, 123.16243015479324, 84.20561818493266, 72.764717365441, 70.1675187980795, 87.8438773388012, 56.63247872490017, 57.3222613935694, 58.27542450517243, 56.13568165416574, 56.13568165416574, 72.19807254818821, 56.20033422585568, 56.20033422585568, 147.02380936260127, 56.44575084842972, 70.16817441196125, 56.78580194372706, 44.32901158931224, 43.97197331226519, 43.57425532422477, 42.19828908102332, 42.19828908102332, 42.16878193486985, 43.293661625788936, 43.411260670273414, 128.57601031387048, 84.93569171774128, 79.32738067136127, 84.31213509407816, 397.56226843982347, 113.41677049596169, 73.91450586239128, 69.99209913070035, 812.6567191455608, 1037.1821013839456, 235.96569074464514, 89.98149509653604, 304.0597198655743, 598.3663648532408, 209.24196462979864, 97.48972091450617, 115.24067927202012, 528.9404176990132, 918.495117659681, 257.85389824215963, 546.8451333369585, 278.1119512123084, 380.8400451278398, 183.6898069572014, 191.24815784753707, 2992.612556405434, 589.4009309284337, 471.2249100562244, 509.25995603720486, 289.6166455232311, 176.97051523419526, 331.34902825103774, 217.56231066677122, 248.10594167400444, 607.7059960528966, 344.64321386996494, 291.2618294336514, 276.20285784637383, 419.1457646910041, 261.87452045481064, 297.82393659814096, 309.75699545496934, 269.39635591267074, 282.3685039928859, 234.1366274870337, 128.17646015318485, 78.56964731406482, 88.65290636571842, 78.86672214992078, 58.96046368591592, 79.26292235440236, 56.967581327058916, 59.41006776747674, 52.92611178248106, 49.13250461584565, 39.677096053616545, 40.145616126614435, 61.44895782439997, 39.346437142181095, 40.03885732883434, 30.845672309856305, 31.055680109673045, 30.41050471110934, 29.58996191631915, 30.062581144538868, 30.070159968502058, 63.244259771802234, 39.45262352692149, 29.507959982506264, 29.532737558787407, 38.30808365403294, 77.23609816181605, 25.315650187278198, 118.41304441921648, 30.207853548615375, 178.54257135772195, 43.416335279105844, 120.856103272674, 58.40398406444191, 73.42713375586541, 40.41128732866714, 71.49115753808059, 39.326843293197015, 40.33672676437041, 53.55811603231868, 136.35249167879772, 66.76356058441098, 131.68155608430828, 301.74659994300475, 81.3387330523865, 143.00263573907066, 173.0106813610143, 271.8882102039417, 144.2911676184469, 102.83785130737667, 128.53528781157416, 533.2648683627722, 172.37849634916415, 163.64874558864372, 109.1236544678779, 187.81362902828897, 95.4693514775265, 159.54713501614432, 124.16019184585502, 138.2027716391384, 139.62845669552797, 104.83199523444618, 99.00847585751026, 96.93701097957755, 110.38925969866389, 93.3720602270524, 90.08747135597395, 101.8852957553047, 93.22213320115303], "Total": [8966.0, 987.0, 599.0, 693.0, 642.0, 1263.0, 1434.0, 397.0, 918.0, 970.0, 350.0, 1342.0, 350.0, 1156.0, 1332.0, 1891.0, 715.0, 749.0, 505.0, 1350.0, 851.0, 248.0, 819.0, 638.0, 223.0, 454.0, 357.0, 1086.0, 1050.0, 538.0, 350.5992700969288, 599.7372104680098, 175.5375318088239, 186.59595189769868, 158.79486713880635, 180.60950475331262, 132.28824714644716, 111.47509208699046, 93.61906458235978, 83.62441058875359, 223.0618414563693, 77.97705857614066, 77.89594044007538, 49.55097365907504, 183.4632897561405, 397.947836265786, 44.285042499631906, 40.705125836473634, 38.83690124594079, 38.83691469740146, 38.83691469740146, 63.52493635217664, 63.52493635217664, 35.61156129740661, 79.89809785084694, 31.98913086358715, 25.584927231468633, 35.81348144564147, 35.81348144564147, 35.81348144564147, 207.51836330495715, 106.64244042344657, 41.79411375037365, 182.2296043333087, 72.93439194099555, 74.85440651123496, 248.24006290339554, 120.51670844416023, 75.92455107493521, 73.4187386014877, 73.21872216064753, 159.6065114132404, 350.5055885992806, 987.3709020248958, 693.9455982609157, 642.2108680298634, 137.58416008754844, 170.43458546123057, 217.0563867351007, 357.87396688785793, 293.7929123292649, 505.7172714625118, 171.29198744907197, 192.69140693153412, 126.75843540329481, 112.2856668777107, 95.3879004685197, 98.47098435726369, 94.76049149420504, 95.38751261467735, 98.46035475620113, 97.8290550786742, 97.27668671077792, 74.79040792439793, 74.79471996449156, 202.17121286876662, 195.09688789024065, 92.73176920658493, 79.86051469568781, 52.50933251290428, 52.49644354366198, 50.67456912247521, 50.65836057810632, 53.077346705765024, 60.60506505946615, 53.706336318953696, 49.39042758943654, 49.39042758943654, 51.86115184608321, 62.373231252577945, 49.33828505170873, 49.33828505170873, 162.88644432603647, 100.03210825772112, 96.95984059087827, 98.78209619576623, 155.95950698498496, 105.0502780998245, 228.3588409035685, 109.9778872454112, 320.74321446215913, 638.6135290573405, 125.40791271985223, 395.2796800487442, 255.86846672293584, 1332.1208341347512, 538.6713233447637, 562.8706829717175, 234.67560212844788, 403.7905285071862, 1156.2016733167263, 211.17349578145453, 291.0592257658909, 342.9761038793888, 238.7528655553662, 163.21785557852007, 806.1073624735599, 1025.22757268474, 598.8996607202039, 246.13492101073012, 1350.9641754382383, 425.39364661798237, 651.1514969182482, 671.261664088899, 769.323762659313, 8966.123260341796, 1561.2978654306016, 2418.0056428919634, 626.7916537634133, 1053.1078433727498, 723.34624646324, 1174.8910835055835, 851.3091936157641, 1342.3204259470901, 1597.421016909452, 884.8508120913231, 613.1716347378817, 1110.4074617982906, 1050.2280799159164, 326.7462296339727, 404.1805607311994, 272.3303826526922, 245.05687884825227, 145.44502235211115, 144.43845914776747, 336.74690041976464, 113.64769712062062, 110.93746724206537, 114.52651335257165, 137.28610987915877, 108.22412892474716, 93.77749984460004, 185.5534835607316, 89.20474466244264, 116.22704784906169, 89.13543986229085, 97.28286991992684, 80.96844479325912, 81.84458790047779, 76.41811602245575, 76.41795736946894, 71.84169840002352, 70.90688017832984, 72.69349695769733, 63.768782808033095, 66.39589544292797, 90.83967330752763, 143.30428033283965, 242.7814837884347, 136.1584198555897, 136.1574657779105, 259.59311005919886, 238.61396169883386, 181.13654145764085, 1185.864111412001, 248.86061548523608, 217.46500568257193, 185.55016308355962, 187.12707735494166, 175.1166953790903, 358.5835850832553, 297.6607052113437, 264.4148143985684, 175.80249718092378, 741.0617515425437, 512.8948049240333, 210.73393748609564, 532.2827949649837, 190.1466661151588, 373.88323177534414, 365.0162282242885, 784.343854833461, 295.66267525112164, 773.7004738267725, 260.9629015929286, 276.5305268035051, 643.5819161487908, 628.3772786570361, 518.1874894652058, 2418.0056428919634, 1110.4074617982906, 860.6782055712563, 538.0404294324055, 1597.421016909452, 8966.123260341796, 1174.8910835055835, 474.55862487808116, 1891.7995445563124, 754.6595262973563, 510.1278792603749, 406.75384649159287, 1197.4807386634457, 511.5601970718928, 951.9844582111198, 866.1875189246829, 851.3091936157641, 953.839517997262, 1561.2978654306016, 569.7665180702564, 587.244154654381, 1350.9641754382383, 733.39919713546, 987.3709020248958, 1053.1078433727498, 198.67681787080156, 99.21201713928095, 210.4059070237912, 220.6737705590507, 87.08630601640738, 124.65035639836128, 85.27767713505321, 73.7951909499931, 71.26805958174121, 89.26543521420679, 57.620656193145265, 58.32921838654355, 59.31145834933287, 57.15639215610471, 57.15639215610471, 73.51091276615315, 57.243178422133845, 57.243178422133845, 149.87549021588546, 57.568754796853185, 71.5674861109924, 57.96951468238169, 45.34852867879268, 44.989474268372476, 44.59385525679092, 43.197878616752526, 43.197878616752526, 43.175643399626615, 44.32857669215346, 44.482530956060245, 131.8646619080557, 87.04102630988511, 81.61356074471315, 86.88995248703476, 419.8110019677773, 117.62387160070656, 76.15810535803192, 72.12132365521427, 970.906654165863, 1263.187166242923, 262.1410927171737, 94.11931599640975, 345.17088638542276, 715.5172869473591, 232.7582251674515, 103.19148215986023, 125.18119649905219, 749.3291990102223, 1434.1453963675096, 333.39544812714, 851.9928942957498, 379.24735034305644, 556.784462542558, 229.6088114500323, 244.34263273221381, 8966.123260341796, 1086.9297519844208, 819.0247203329868, 918.976417854994, 433.40103011741917, 227.85940502446135, 539.9870073567242, 331.1040312988968, 416.035078588583, 1891.7995445563124, 730.0498829768812, 555.8424388829602, 538.2421695994915, 1342.3204259470901, 503.0960251406729, 953.839517997262, 1197.4807386634457, 733.39919713546, 1561.2978654306016, 532.4893089567906, 129.34433368643963, 79.72509964807605, 89.9781358796307, 80.14670745844037, 60.08736493864399, 80.79366931518919, 58.142708408726385, 60.7626072154041, 54.13731265390968, 50.28435787839459, 40.719782542696755, 41.2082299127118, 63.29821082913146, 40.53311180215429, 41.25453268339434, 31.905830161625207, 32.130997480728176, 31.471044685315885, 30.63636425670967, 31.149270854093004, 31.16158179818707, 65.6903493530992, 40.97861951731367, 30.652173945480346, 30.679051398405708, 39.82833674270956, 80.42447272864595, 26.380629139777987, 123.67816841467987, 31.57923507571102, 187.27230148640294, 45.616654767394245, 134.1471186122996, 62.449699452776386, 81.47806387172693, 42.9232994109583, 81.65722678306464, 42.31089383201742, 43.582306089470016, 60.72517302366606, 187.76703619881934, 84.01155002395987, 255.36299000950285, 918.976417854994, 132.86257845449046, 342.28744034611077, 538.2421695994915, 1342.3204259470901, 442.37779456328326, 249.46863749314434, 398.17668297669985, 8966.123260341796, 817.9969593374582, 749.3291990102223, 331.8249148601747, 1891.7995445563124, 259.8062096752741, 1434.1453963675096, 819.0247203329868, 1263.187166242923, 1350.9641754382383, 456.2631172446392, 378.4554881174231, 348.7504126017838, 970.906654165863, 344.6806821812854, 256.2423871657579, 851.9928942957498, 1197.4807386634457], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -3.890399932861328, -3.354300022125244, -4.58519983291626, -4.524199962615967, -4.687600135803223, -4.558899879455566, -4.871200084686279, -5.042699813842773, -5.218900203704834, -5.333499908447266, -4.353000164031982, -5.405099868774414, -5.406300067901611, -5.8678998947143555, -4.560800075531006, -3.786600112915039, -5.983399868011475, -6.068399906158447, -6.116099834442139, -6.116099834442139, -6.116099834442139, -5.626299858093262, -5.626299858093262, -6.206299781799316, -5.398799896240234, -6.316500186920166, -6.549300193786621, -6.213500022888184, -6.213500022888184, -6.213500022888184, -4.462100028991699, -5.140600204467773, -6.061500072479248, -4.620200157165527, -5.520500183105469, -5.497600078582764, -4.3933000564575195, -5.0721001625061035, -5.500199794769287, -5.5406999588012695, -5.557300090789795, -4.88040018081665, -4.2170000076293945, -3.359299898147583, -3.6602001190185547, -3.7388999462127686, -5.066800117492676, -4.9969000816345215, -4.883500099182129, -4.969099998474121, -5.192699909210205, -5.2245001792907715, -6.323699951171875, -6.206699848175049, -6.626699924468994, -6.750400066375732, -6.91349983215332, -6.881800174713135, -6.920199871063232, -6.913700103759766, -6.881999969482422, -6.888599872589111, -6.894499778747559, -7.158899784088135, -7.158899784088135, -6.166299819946289, -6.202199935913086, -6.946300029754639, -7.095900058746338, -7.5177001953125, -7.51800012588501, -7.553400039672852, -7.553899765014648, -7.507500171661377, -7.374899864196777, -7.495999813079834, -7.579899787902832, -7.579899787902832, -7.531099796295166, -7.346799850463867, -7.581600189208984, -7.581600189208984, -6.389599800109863, -6.875500202178955, -6.908199787139893, -6.891600131988525, -6.445300102233887, -6.835700035095215, -6.086999893188477, -6.8003997802734375, -5.769599914550781, -5.1234002113342285, -6.686699867248535, -5.59660005569458, -6.044000148773193, -4.540900230407715, -5.390100002288818, -5.356599807739258, -6.142099857330322, -5.685100078582764, -4.791800022125244, -6.241300106048584, -5.982699871063232, -5.854400157928467, -6.159299850463867, -6.4745001792907715, -5.184899806976318, -4.996600151062012, -5.451399803161621, -6.157299995422363, -4.857900142669678, -5.765600204467773, -5.4984002113342285, -5.477399826049805, -5.406799793243408, -3.791100025177002, -4.960000038146973, -4.726099967956543, -5.5640997886657715, -5.248300075531006, -5.511300086975098, -5.36959981918335, -5.549200057983398, -5.350100040435791, -5.297900199890137, -5.6894001960754395, -5.833899974822998, -5.776100158691406, -5.3414998054504395, -6.509799957275391, -6.297299861907959, -6.69290018081665, -6.799799919128418, -7.321899890899658, -7.3292999267578125, -6.48330020904541, -7.569799900054932, -7.593999862670898, -7.5625, -7.381800174713135, -7.620299816131592, -7.764200210571289, -7.081900119781494, -7.8144001960754395, -7.549900054931641, -7.815299987792969, -7.72790002822876, -7.911600112915039, -7.901199817657471, -7.969900131225586, -7.969900131225586, -8.032999992370605, -8.047499656677246, -8.022700309753418, -8.153800010681152, -8.114100456237793, -7.800600051879883, -7.345300197601318, -6.818900108337402, -7.396500110626221, -7.396500110626221, -6.76170015335083, -6.846099853515625, -7.11899995803833, -5.281799793243408, -6.8144001960754395, -6.948599815368652, -7.103600025177002, -7.0971999168396, -7.163400173187256, -6.475500106811523, -6.655900001525879, -6.774600028991699, -7.161200046539307, -5.809500217437744, -6.162099838256836, -6.993500232696533, -6.13670015335083, -7.092100143432617, -6.481500148773193, -6.507400035858154, -5.841000080108643, -6.701200008392334, -5.855999946594238, -6.812600135803223, -6.764400005340576, -6.047699928283691, -6.09060001373291, -6.2555999755859375, -5.00439977645874, -5.658199787139893, -5.870100021362305, -6.257599830627441, -5.445099830627441, -4.199399948120117, -5.727399826049805, -6.369800090789795, -5.4197001457214355, -6.08489990234375, -6.342400074005127, -6.486499786376953, -5.86899995803833, -6.36359977722168, -6.035900115966797, -6.112400054931641, -6.162799835205078, -6.140399932861328, -5.9781999588012695, -6.336699962615967, -6.338399887084961, -6.215199947357178, -6.328700065612793, -6.332900047302246, -6.3354997634887695, -5.9756999015808105, -6.673600196838379, -5.9222002029418945, -5.875199794769287, -6.804999828338623, -6.446800231933594, -6.827099800109863, -6.973100185394287, -7.009399890899658, -6.784800052642822, -7.223700046539307, -7.211599826812744, -7.195199966430664, -7.232600212097168, -7.232600212097168, -6.980899810791016, -7.231400012969971, -7.231400012969971, -6.269700050354004, -7.227099895477295, -7.009399890899658, -7.2210001945495605, -7.468699932098389, -7.476799964904785, -7.485899925231934, -7.51800012588501, -7.51800012588501, -7.518700122833252, -7.492300033569336, -7.48960018157959, -6.403800010681152, -6.818399906158447, -6.88670015335083, -6.825799942016602, -5.275000095367432, -6.529300212860107, -6.957399845123291, -7.011899948120117, -4.559999942779541, -4.316100120544434, -5.796599864959717, -6.760700225830078, -5.543099880218506, -4.866099834442139, -5.916800022125244, -6.680600166320801, -6.513299942016602, -4.989500045776367, -4.437600135803223, -5.707900047302246, -4.956200122833252, -5.632299900054932, -5.317999839782715, -6.047100067138672, -6.006800174713135, -3.2564001083374023, -4.881199836730957, -5.105000019073486, -5.027400016784668, -5.591800212860107, -6.0843000411987305, -5.457200050354004, -5.877799987792969, -5.746500015258789, -4.850599765777588, -5.417799949645996, -5.586100101470947, -5.639200210571289, -5.222099781036377, -5.692500114440918, -5.563799858093262, -5.524499893188477, -5.664100170135498, -5.617099761962891, -5.8043999671936035, -5.530900001525879, -6.020299911499023, -5.899600028991699, -6.016600131988525, -6.307499885559082, -6.011600017547607, -6.341800212860107, -6.299900054931641, -6.41540002822876, -6.489799976348877, -6.70359992980957, -6.691800117492676, -6.26609992980957, -6.711900234222412, -6.694499969482422, -6.9552998542785645, -6.948500156402588, -6.9695000648498535, -6.9969000816345215, -6.980999946594238, -6.980800151824951, -6.237299919128418, -6.709199905395508, -6.99970006942749, -6.998799800872803, -6.738699913024902, -6.037499904632568, -7.152900218963623, -5.610099792480469, -6.976200103759766, -5.19950008392334, -6.613500118255615, -5.589700222015381, -6.31689977645874, -6.0879998207092285, -6.685200214385986, -6.114799976348877, -6.712399959564209, -6.687099933624268, -6.403600215911865, -5.469099998474121, -6.183199882507324, -5.503900051116943, -4.674699783325195, -5.9857001304626465, -5.421500205993652, -5.230999946594238, -4.778900146484375, -5.412499904632568, -5.751200199127197, -5.52810001373291, -4.105299949645996, -5.234600067138672, -5.286600112915039, -5.691800117492676, -5.148900032043457, -5.825500011444092, -5.311999797821045, -5.56279993057251, -5.455599784851074, -5.445300102233887, -5.73199987411499, -5.789100170135498, -5.810299873352051, -5.680300235748291, -5.847700119018555, -5.883500099182129, -5.760499954223633, -5.849299907684326], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 3.247, 3.2463, 3.244, 3.2439, 3.2419, 3.2419, 3.2409, 3.2405, 3.2389, 3.2373, 3.2366, 3.2355, 3.2354, 3.2261, 3.2242, 3.2241, 3.223, 3.2223, 3.2216, 3.2216, 3.2216, 3.2193, 3.2193, 3.2181, 3.2175, 3.2151, 3.2058, 3.2052, 3.2052, 3.2052, 3.1997, 3.187, 3.2028, 3.1715, 3.187, 3.1839, 3.0893, 3.1332, 3.1671, 3.1602, 3.1463, 3.044, 2.9207, 2.7427, 2.7945, 2.7933, 3.006, 2.8618, 2.7334, 2.1478, 2.1215, 1.5465, 1.53, 1.5293, 1.5281, 1.5256, 1.5256, 1.5255, 1.5255, 1.5254, 1.5254, 1.5252, 1.525, 1.5235, 1.5234, 1.5217, 1.5213, 1.521, 1.5209, 1.5184, 1.5183, 1.5182, 1.518, 1.5179, 1.5178, 1.5176, 1.5174, 1.5174, 1.5173, 1.5171, 1.5168, 1.5168, 1.5144, 1.5161, 1.5145, 1.5126, 1.5022, 1.507, 1.4792, 1.4964, 1.4568, 1.4144, 1.4788, 1.4209, 1.4084, 1.2616, 1.3178, 1.3074, 1.3967, 1.3111, 1.1524, 1.4031, 1.3409, 1.305, 1.3623, 1.4274, 1.12, 1.0678, 1.1506, 1.3339, 0.9306, 1.1784, 1.0199, 1.0105, 0.9447, 0.1048, 0.6838, 0.4803, 0.9923, 0.7893, 0.9019, 0.5586, 0.7011, 0.4448, 0.323, 0.5222, 0.7445, 0.2085, 0.6988, 0.698, 0.6979, 0.6971, 0.6958, 0.6954, 0.6949, 0.6944, 0.6941, 0.6941, 0.6938, 0.6932, 0.6926, 0.6919, 0.6918, 0.6917, 0.6917, 0.6916, 0.6915, 0.6914, 0.6911, 0.6909, 0.6909, 0.6896, 0.6882, 0.6881, 0.688, 0.6873, 0.6873, 0.6868, 0.686, 0.6867, 0.6867, 0.6763, 0.6761, 0.6788, 0.637, 0.6658, 0.6664, 0.6701, 0.668, 0.6682, 0.6394, 0.6452, 0.645, 0.6665, 0.5795, 0.5949, 0.6529, 0.5832, 0.6572, 0.5916, 0.5897, 0.4912, 0.6067, 0.4899, 0.62, 0.6103, 0.4823, 0.4633, 0.4911, 0.202, 0.3264, 0.3692, 0.4516, 0.1758, -0.3036, 0.2007, 0.4649, 0.0321, 0.2859, 0.42, 0.5023, 0.0401, 0.396, 0.1026, 0.1206, 0.0874, -0.0038, -0.3344, 0.3151, 0.2832, -0.4267, 0.0707, -0.2309, -0.2979, 1.7297, 1.7262, 1.7258, 1.7252, 1.7251, 1.7247, 1.7241, 1.7227, 1.7212, 1.7207, 1.7194, 1.7193, 1.7191, 1.7187, 1.7187, 1.7187, 1.7183, 1.7183, 1.7175, 1.717, 1.717, 1.7161, 1.714, 1.7139, 1.7136, 1.7133, 1.7133, 1.7131, 1.7131, 1.7123, 1.7115, 1.7122, 1.7083, 1.7066, 1.6823, 1.7003, 1.7068, 1.7068, 1.5588, 1.5396, 1.6315, 1.6918, 1.6099, 1.5579, 1.6302, 1.6799, 1.654, 1.3884, 1.2911, 1.4798, 1.2933, 1.4266, 1.3569, 1.5136, 1.4917, 0.6394, 1.1247, 1.1839, 1.1464, 1.3336, 1.484, 1.2484, 1.3168, 1.2198, 0.6011, 0.9861, 1.0905, 1.0696, 0.5728, 1.0838, 0.5727, 0.3845, 0.7352, 0.0267, 0.9151, 2.6037, 2.5981, 2.5979, 2.5966, 2.5938, 2.5936, 2.5923, 2.5902, 2.5901, 2.5896, 2.5868, 2.5866, 2.5831, 2.583, 2.5828, 2.5789, 2.5787, 2.5785, 2.578, 2.5772, 2.5771, 2.5748, 2.5748, 2.5747, 2.5747, 2.5738, 2.5723, 2.5715, 2.5692, 2.5683, 2.565, 2.5633, 2.5084, 2.5458, 2.5087, 2.5524, 2.4798, 2.5396, 2.5353, 2.4871, 2.2928, 2.3829, 1.9504, 1.4991, 2.122, 1.7399, 1.4778, 1.016, 1.4924, 1.7266, 1.482, -0.2095, 1.0556, 1.0913, 1.5006, 0.3029, 1.6116, 0.4167, 0.7262, 0.4001, 0.3431, 1.142, 1.2718, 1.3324, 0.4385, 1.3067, 1.5674, 0.489, 0.0597]}, "token.table": {"Topic": [3, 2, 3, 5, 3, 4, 2, 3, 3, 5, 2, 3, 2, 3, 1, 3, 1, 2, 3, 4, 5, 1, 2, 3, 5, 2, 2, 3, 4, 2, 3, 5, 2, 3, 4, 3, 2, 3, 2, 3, 4, 5, 3, 3, 5, 1, 2, 3, 4, 5, 3, 1, 2, 3, 4, 5, 4, 3, 4, 2, 3, 4, 3, 4, 1, 3, 1, 2, 3, 4, 5, 3, 5, 1, 2, 3, 4, 1, 1, 2, 3, 5, 1, 3, 5, 5, 2, 4, 5, 2, 3, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 1, 3, 1, 2, 3, 4, 5, 3, 5, 3, 5, 3, 4, 3, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5, 2, 3, 4, 5, 2, 3, 2, 3, 4, 2, 1, 2, 3, 5, 1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5, 2, 1, 2, 3, 4, 5, 2, 3, 2, 3, 4, 5, 1, 3, 2, 1, 2, 3, 4, 5, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5, 2, 3, 2, 3, 2, 3, 1, 2, 3, 4, 5, 3, 2, 3, 2, 3, 4, 5, 1, 5, 5, 2, 3, 4, 5, 2, 3, 2, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 2, 5, 4, 3, 5, 3, 2, 3, 1, 2, 3, 4, 5, 3, 1, 2, 3, 4, 5, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 5, 2, 1, 2, 3, 4, 5, 2, 3, 4, 4, 1, 2, 3, 4, 5, 4, 2, 3, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 1, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5, 3, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 4, 5, 4, 1, 3, 1, 2, 3, 4, 5, 2, 3, 4, 1, 3, 5, 2, 3, 4, 5, 1, 3, 2, 3, 1, 2, 3, 4, 5, 1, 3, 3, 4, 1, 2, 3, 4, 3, 2, 3, 4, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 4, 1, 3, 4, 5, 3, 4, 5, 1, 2, 3, 4, 5, 5, 4, 4, 3, 2, 4, 2, 3, 4, 5, 3, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 5, 1, 2, 3, 4, 5, 3, 4, 3, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4, 5, 3, 1, 2, 3, 4, 1, 2, 3, 5, 2, 2, 3, 4, 2, 1, 3, 1, 3, 5, 1, 2, 3, 4, 2, 3, 1, 2, 3, 4, 5, 2, 3, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 2, 3, 2, 3, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 4, 5, 5, 3, 5, 2, 3, 4, 4, 1, 2, 3, 4, 3, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 5, 2, 3, 1, 2, 3, 4, 5, 3, 5, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 1, 3, 4, 1, 5, 2, 1, 2, 3, 4, 1, 3, 2, 3, 5, 2, 3, 5, 2, 2, 3, 1, 2, 3, 4, 5, 2, 3, 1, 2, 3, 4, 5, 2, 3, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 5, 2, 3, 4, 5, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 1, 1, 3, 1, 2, 3, 4, 5, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 3, 4, 1, 2, 3, 4, 5, 5, 2, 3, 4, 5, 1, 2, 3, 5, 1, 3, 5, 3, 1, 3, 4, 1, 3, 3, 3, 5, 1, 2, 3, 4, 5, 1, 3, 1, 2, 3, 4, 5, 1, 4, 4, 2, 1, 2, 3, 4, 5, 1, 3, 2, 3, 1, 2, 3, 3, 4, 3, 1, 3, 1, 2, 3, 4, 5, 2, 1, 2, 3, 4, 5, 2, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 3, 4, 1, 3, 1, 2, 3, 4, 5, 2, 3, 2, 3, 4, 5, 3, 5, 1, 2, 3, 4, 5, 3, 5, 1, 2, 4, 3, 1, 4, 1, 2, 3, 4, 5, 2, 3, 1, 3, 4, 4, 3, 5, 3, 4, 1, 3, 2, 3, 1, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 1, 3, 4, 2, 4, 1, 2, 3, 4, 5], "Freq": [0.9970791254060718, 0.017131433376501536, 0.9707812246684204, 0.011420955584334359, 0.01725044629887032, 0.9832754390356083, 0.9819593199132516, 0.020246583915737144, 0.024671187469668964, 0.9621763113170896, 0.9796854400740448, 0.00999679020483719, 0.0029695893228815808, 0.9948124231653296, 0.9910884825496257, 0.005536807165081707, 0.001130134013932275, 0.36277301847226023, 0.38424556473697347, 0.20907479257747083, 0.042945092529426444, 0.02135927186375943, 0.005339817965939858, 0.02135927186375943, 0.9558274159032346, 0.9894316938985422, 0.005389281735972908, 0.9916278394190151, 0.005389281735972908, 0.9728753228794607, 0.047269150303002196, 0.9217484309085429, 0.7999197038972934, 0.1981225272810634, 0.0024765315910132923, 0.9954026946498338, 0.9854539386063847, 0.010483552538365795, 0.06634343438661087, 0.039025549639182866, 0.5463576949485601, 0.3512299467526458, 0.9940373506481547, 0.9907565353666012, 0.011008405948517791, 0.045206062458090515, 0.09794646865919612, 0.25867913517685126, 0.27374782266288145, 0.3239767809496487, 0.9969643878067205, 0.08233229687602064, 0.060769076265634284, 0.7547127213635225, 0.06664995461392147, 0.03332497730696073, 0.9892243527016286, 0.9988306540841921, 0.0009521741221012318, 0.9709636361273984, 0.019038502669164674, 0.9850174491381869, 0.008022439958407904, 0.9867601148841723, 0.9709824711212011, 0.02258098770049305, 0.0027649275983374286, 0.5308660988807863, 0.24469609245286242, 0.1728079748960893, 0.048386232970905003, 0.08591269437894701, 0.895946669951876, 0.011155000302290155, 0.04183125113358808, 0.9398087754679455, 0.005577500151145077, 0.9827058239731136, 0.8145031104866042, 0.01879622562661394, 0.08771571959086506, 0.07518490250645576, 0.9493135781458533, 0.04818850650486565, 0.947665041176133, 0.9803464881495806, 0.9940166869298933, 0.04659474055923578, 0.9318948111847156, 0.01676347842985228, 0.9764726185388954, 0.00419086960746307, 0.002349322684400251, 0.43462469661404646, 0.5415188787542579, 0.019969242817402134, 0.0011746613422001255, 0.010876626307165627, 0.05749073905216116, 0.8048703467302563, 0.09788963676449063, 0.029522271405163843, 0.018471548567488456, 0.9789920740768882, 0.597403218515274, 0.39622115778133854, 0.0030714818432661904, 0.0030714818432661904, 0.00719628391102798, 0.13133218137626065, 0.30224392426317515, 0.5235296545272856, 0.03418234857738291, 0.04188431404472934, 0.8418747122990596, 0.0376958826402564, 0.004188431404472934, 0.07120333387603987, 0.9884982396385006, 0.012837639475824683, 0.9874699226415855, 0.01282428470963098, 0.32335701786273824, 0.22464803346253395, 0.24847434004189362, 0.19401421071764297, 0.006807516165531332, 0.015222936243264493, 0.9590449833256631, 0.032595531948291165, 0.9778659584487349, 0.027731049551465114, 0.970586734301279, 0.9904599862887199, 0.7189614322757684, 0.10270877603939547, 0.17688733651229221, 0.0028530215566498745, 0.9977161798169527, 0.5943156151283113, 0.018428391166769344, 0.31328264983507886, 0.004607097791692336, 0.06910646687538503, 0.33251929732371227, 0.25707374246875236, 0.29898793961039677, 0.07544555485495993, 0.039119917332201444, 0.9826772225368774, 0.03382232806865622, 0.9098206250468523, 0.01691116403432811, 0.03720456087552184, 0.9823235170290503, 0.3482208581585269, 0.6075254639369073, 0.03763432077485267, 0.006203459468382308, 0.9813249631555401, 0.010783790803907034, 0.8862950348632206, 0.02818606117939571, 0.08455818353818713, 0.9885500356950196, 0.025298543043666813, 0.8930385694414384, 0.055656794696066984, 0.025298543043666813, 0.024036434701432945, 0.10576031268630497, 0.17065868638017392, 0.596103580595537, 0.10335666921616167, 0.9784505359418405, 0.0427078813567645, 0.9538093503010738, 0.021921817921527587, 0.021921817921527587, 0.9426381706256862, 0.9920950757364967, 0.003680090633913693, 0.13708337611328505, 0.2382858685459116, 0.5418933458437912, 0.07912194862914439, 0.01926091181256612, 0.9746021377158456, 0.042190924710699795, 0.31643193533024844, 0.31040466037157705, 0.3284864852475912, 0.9759946811481632, 0.015741849695938115, 0.9833950497544098, 0.010659642048369894, 0.7959199396116187, 0.07461749433858925, 0.04796838921766452, 0.0692876733144043, 0.9891291826612277, 0.05945453254342863, 0.8103911283636903, 0.011632408541105602, 0.11890906508685727, 0.058824489745588876, 0.17122128265233905, 0.5504291540480102, 0.14285947509643013, 0.07773236144952816, 0.9532572019764485, 0.9841264106069035, 0.010251316777155245, 0.9779836441851617, 0.0160325187571338, 0.9682000342213032, 0.025647683025729884, 0.006040397612056087, 0.2234947116460752, 0.08154536776275717, 0.6584033397141135, 0.03322218686630848, 0.9969402710047114, 0.97978708938737, 0.020627096618681475, 0.03773493735586908, 0.12438553424712401, 0.8357589829188782, 0.0027951805448791912, 0.9547461206783997, 0.9896065513801432, 0.9648004242194306, 0.11747982743655846, 0.12531181593232904, 0.2427916433688875, 0.5169112407208573, 0.9964118434623305, 0.0051896450180329715, 0.9905432918851144, 0.06637921247183107, 0.8211354431700584, 0.007375468052425674, 0.10325655273395944, 0.026429861527282796, 0.04070198675201551, 0.5127393136292863, 0.3213871161717588, 0.09937627934258332, 0.9493631623500771, 0.027922445951472856, 0.016457489989772645, 0.970991909396586, 0.978282505332493, 0.03210348019650676, 0.9631044058952029, 0.9945285745812827, 0.990016262520757, 0.016500271042012616, 0.0008649010143112491, 0.6824069002915755, 0.3139590681949834, 0.0017298020286224982, 0.0008649010143112491, 0.9894426652679401, 0.008882544939511315, 0.5462765137799459, 0.324212890292163, 0.01776508987902263, 0.10362969096096535, 0.013972825571225493, 0.9780977899857846, 0.0011153092267012614, 0.23923382912742056, 0.36637908097136435, 0.3338120515516875, 0.059445981783177226, 0.005325748439364715, 0.01065149687872943, 0.12781796254475317, 0.13314371098411787, 0.7243017877536012, 0.023965260629400634, 0.015976840419600423, 0.007988420209800211, 0.9186683241270243, 0.031953680839200846, 0.02440296944550548, 0.9517158083747138, 0.9924573970545114, 0.014464949118772297, 0.028929898237544594, 0.9149080317623478, 0.0036162372796930743, 0.03977861007662382, 0.05349263700603036, 0.8960016698510085, 0.048143373305427326, 0.9877835651947341, 0.162870789727493, 0.01980860956145185, 0.7087080309763885, 0.10564591766107655, 0.00440191323587819, 0.9727706802480034, 0.026946890894126177, 0.9700880721885424, 0.9380523064564661, 0.06641189987315109, 0.6590103910489608, 0.2707562071751544, 0.003405738455033389, 0.15051992012032692, 0.7369611673423798, 0.0410508873055437, 0.0723277538240532, 0.9953244657980258, 0.010156372099979856, 0.9637275738872146, 0.02503188503603155, 0.016562091645663645, 0.9771634070941551, 0.9778886173796532, 0.004531576169957214, 0.9878836050506727, 0.004531576169957214, 0.02103639302083516, 0.01051819651041758, 0.9571558824479998, 0.01051819651041758, 0.0037128391902903433, 0.8038296846978593, 0.14851356761161372, 0.04084123109319378, 0.004296303596921559, 0.042963035969215586, 0.8979274517566057, 0.051555643163058704, 0.9868129926575689, 0.06897532135842772, 0.9235029137433934, 0.007663924595380859, 0.020782473340450136, 0.22738470831316032, 0.33985456403794934, 0.20048974281375426, 0.21026973026808374, 0.876056909108794, 0.06156075577521255, 0.004735442751939427, 0.05682531302327312, 0.012271625105607644, 0.040905417018692145, 0.533133935143621, 0.36678523926760626, 0.04635947262118443, 0.9219138713198217, 0.005487582567379892, 0.06585099080855869, 0.007454502268439424, 0.007454502268439424, 0.08199952495283368, 0.9019947744811704, 0.02441363569433635, 0.05070524336516011, 0.4826387979572647, 0.4394454424980543, 0.0018779719764874115, 0.06691262360900059, 0.23534784855579519, 0.669126236090006, 0.02768798218303473, 0.045383236288386156, 0.9454840893413783, 0.003781936357365513, 0.9627239141546194, 0.007146072842155446, 0.016674169965029375, 0.9480456637259559, 0.028584291368621784, 0.978282505332493, 0.9819036008692624, 0.9866830249734804, 0.986273575810284, 0.008966123416457127, 0.001088167204914926, 0.0837888747784493, 0.03264501614744778, 0.5538771073016974, 0.3286264958843077, 0.008036627234487621, 0.9643952681385145, 0.02410988170346286, 0.9219679143168347, 0.06585485102263104, 0.013170970204526209, 0.011508810528457989, 0.011508810528457989, 0.966740084390471, 0.011508810528457989, 0.988696944862598, 0.006297432769825465, 0.9638301176259669, 0.03637094783494215, 0.6015976354800688, 0.001012790632121328, 0.39397555589519656, 0.002025581264242656, 0.001012790632121328, 0.9912410081594493, 0.005696787403215226, 0.026261157503822703, 0.9716628276414401, 0.04911136409482558, 0.03274090939655039, 0.13505625126077037, 0.7816892118426405, 0.9915495885621779, 0.10136535621990186, 0.8958505806461596, 0.0027396042221595096, 0.994530639343732, 0.011686084642648063, 0.14607605803310078, 0.15776214267574884, 0.2658584256202434, 0.4177775259746682, 0.0043834356195125886, 0.06355981648293253, 0.21478834535611685, 0.4865613537658974, 0.23013037002441092, 0.01360342243581055, 0.9794464153783596, 0.019245113526152383, 0.007698045410460954, 0.6042965647211849, 0.3656571569968953, 0.015053147569953731, 0.36880211546386643, 0.6096524765831262, 0.00586859354517617, 0.015258343217458042, 0.2171379611715183, 0.642024133842273, 0.11971930832159386, 0.9706798880885905, 0.9772117915632794, 0.9780065385412144, 0.9864946122877042, 0.011202544384624787, 0.9858239058469813, 0.0335952976826412, 0.9473873946504817, 0.013438119073056478, 0.006719059536528239, 0.0988058115151299, 0.8892523036361691, 0.0017960271294811224, 0.03053246120117908, 0.1598464145238199, 0.6842863363323076, 0.12392587193419745, 0.035614004749896455, 0.09999316718240159, 0.37257728046045524, 0.47257044764285683, 0.019176771788405784, 0.08163834064033364, 0.7930581662203839, 0.07580703059459552, 0.014578275114345294, 0.034987860274428705, 0.9690791579238226, 0.9943008337429557, 0.9716092589650058, 0.013166013488352352, 0.04827538279062529, 0.15799216186022824, 0.7767947958127888, 0.008777342325568234, 0.022558811372281405, 0.9700288890081005, 0.03234200547495514, 0.9540891615111765, 0.0031908529540741358, 0.5807352376414927, 0.2632453687111162, 0.09891644157629821, 0.054244500219260305, 0.017093501151354856, 0.01465157241544702, 0.24175094485487583, 0.5750742173062956, 0.1513995816262859, 0.012749510228678072, 0.0038248530686034216, 0.8108688505439253, 0.15554402478987248, 0.015299412274413686, 0.036857202615423856, 0.06318377591215518, 0.680980695942117, 0.16322475443973422, 0.05616335636636016, 0.0016697287802725667, 0.6812493423512072, 0.2153950126551611, 0.06845887999117523, 0.03339457560545133, 0.9570725733989848, 0.02392681433497462, 0.02392681433497462, 0.005555689227941277, 0.05185309946078525, 0.24074653321078868, 0.6129777114828543, 0.08889102764706043, 0.01126023977377616, 0.7604415260556833, 0.1583940394844513, 0.04429027644351956, 0.026273892805477705, 0.07141875132989232, 0.023806250443297445, 0.08332187655154105, 0.03570937566494616, 0.7975093898504644, 0.9014087934393461, 0.013657708991505244, 0.054630835966020974, 0.013657708991505244, 0.013657708991505244, 0.9868481753296506, 0.018619776893012276, 0.008387155123114708, 0.16354952490073682, 0.4948421522637678, 0.3124215283360229, 0.020967887807786773, 0.007798870180781913, 0.040944068449105045, 0.900769505880311, 0.05069265617508244, 0.0153401461635082, 0.011853749308165427, 0.22103756062873178, 0.6401024626409331, 0.11156469937096873, 0.9872102654065579, 0.00496211818203211, 0.6599617182102706, 0.3337024477416594, 0.0012405295455080274, 0.6321911076427325, 0.18529739361942157, 0.16505482120721587, 0.015571209547850552, 0.9866882119738434, 0.20847348312779923, 0.789334714743423, 0.0015914006345633528, 0.9797023255186943, 0.9902618170983774, 0.007559250512201354, 0.9125735646817876, 0.06810250482699907, 0.013620500965399816, 0.784974083726475, 0.043609671318137505, 0.15990212816650418, 0.007268278553022918, 0.9892247789916351, 0.012521832645463735, 0.634055466455405, 0.07205175755175057, 0.16283697206695627, 0.018733456963455148, 0.11240074178073088, 0.9952170239757642, 0.010155275754854738, 0.9744581032236581, 0.008138117542206402, 0.29735429481138775, 0.5915785444142346, 0.08826727488085405, 0.013772198917580065, 0.053899295319857175, 0.7806104839427591, 0.1449705184465124, 0.02044456029373893, 0.00263680154678847, 0.02109441237430776, 0.24258574230453925, 0.7330308300071946, 0.00263680154678847, 0.9493631623500771, 0.027922445951472856, 0.007284058095026616, 0.9906319009236199, 0.9489034417296552, 0.047843871011579255, 0.007973978501929876, 0.01742811645851318, 0.07668371241745799, 0.7180383980907429, 0.13710118280697034, 0.049960600514404443, 0.2093187487733645, 0.3498203746623352, 0.16057328673025223, 0.2781358716577583, 0.005149825658879288, 0.007209755922431003, 0.03810870987570673, 0.8373616521337722, 0.11329616449534433, 0.0028971157170057083, 0.8807231779697354, 0.11588462868022834, 0.9787234032196105, 0.9841546273971522, 0.007344437517889196, 0.012356790778222066, 0.9844243319983579, 0.004118930259407356, 0.972749891805217, 0.03092154174573052, 0.8245744465528139, 0.12025044012228535, 0.024050088024457072, 0.9886889463846057, 0.9666716141326427, 0.9702630114342977, 0.001630864742181819, 0.4533803983265457, 0.47784336945927297, 0.06523458968727276, 0.001630864742181819, 0.04693365219079461, 0.33598535139759317, 0.10206206904982319, 0.3121460359990943, 0.20263418088724022, 0.0015832966431637897, 0.028499339576948213, 0.03958241607909474, 0.8209393094804249, 0.10924746837830149, 0.03166637816281827, 0.03166637816281827, 0.949991344884548, 0.9854499316820822, 0.010483509911511512, 0.0009495703657446294, 0.47478518287231475, 0.3684333019089162, 0.15383039925062997, 0.0028487110972338883, 0.024239760699132033, 0.9695904279652813, 0.9951148210503238, 0.028596542922257908, 0.5537330584037213, 0.3353594579064791, 0.03899528580307896, 0.04289481438338686, 0.02585840218219894, 0.7005276227541167, 0.06582138737287004, 0.17160575993641114, 0.03761222135592573, 0.06962966374592805, 0.33364213878257193, 0.2262964071742662, 0.09864202364006475, 0.2698149470154712, 0.01148883512057614, 0.01148883512057614, 0.976550985248972, 0.0066722050320540596, 0.0066722050320540596, 0.9808141397119468, 0.025107751962126477, 0.9540945745608062, 0.9903001525913682, 0.05226280265211621, 0.11323607241291846, 0.030486634880401126, 0.8013629739991153, 0.9784505359418405, 0.9882839852235002, 0.9822794073627213, 0.006139246296017008, 0.012278492592034016, 0.015798234845838238, 0.015798234845838238, 0.9636923255961325, 0.9728753228794607, 0.9915254718753291, 0.01022191208118896, 0.045210225842696426, 0.14241221140449375, 0.48374941651685177, 0.004521022584269643, 0.3255136260674143, 0.8166252849234568, 0.17876374396334377, 0.00528463732934284, 0.35935533839531314, 0.14796984522159953, 0.22459708649707072, 0.2615895478024706, 0.012377207378697432, 0.012377207378697432, 0.9777993829170971, 0.08017039817499953, 0.44494570987124743, 0.06012779863124965, 0.4128775506012476, 0.1819198298961396, 0.04350256801864208, 0.7019732566644518, 0.06920863093874877, 0.003954778910785643, 0.9323447853656268, 0.0548438109038604, 0.9792284668187969, 0.029514342885649622, 0.938556103763658, 0.03035760982523961, 0.0008432669395899892, 0.004752718277471778, 0.9885654017141299, 0.017044805526100795, 0.8692850818311406, 0.11079123591965517, 0.9875260983488872, 0.08298154794199693, 0.8105174450148537, 0.08298154794199693, 0.02315764128613868, 0.004048245633721352, 0.8865657937849761, 0.05667543887209893, 0.05397660844961803, 0.011997764284036068, 0.176967023189532, 0.7738557963203264, 0.03599329285210821, 0.01706460401931952, 0.9669942277614393, 0.011376402679546345, 0.985692394674666, 0.9867675185606543, 0.9756720281094211, 0.02180272688512673, 0.6747456784594903, 0.011734707451469397, 0.3051023937382043, 0.0058673537257346985, 0.0058673537257346985, 0.026719810252345395, 0.9672571311349033, 0.005343962050469079, 0.053004035072948805, 0.02385181578282696, 0.6612253375350363, 0.23719305695144588, 0.025176916659650682, 0.9458797353556923, 0.004379072848868945, 0.004379072848868945, 0.04379072848868945, 0.032189087058070026, 0.9656726117421007, 0.025505026821290067, 0.9606893436019259, 0.0224340435360809, 0.6271778258130443, 0.23994672651634352, 0.07217909659434724, 0.039015727888836345, 0.990905001671032, 0.10708593352852923, 0.8886253782279706, 0.003757401176439622, 0.001878700588219811, 0.09797050812481382, 0.024492627031203456, 0.012246313515601728, 0.8694882596077227, 0.024868052374406566, 0.012434026187203283, 0.9574200164146528, 0.9872616339354465, 0.8878436972046659, 0.09127365111449837, 0.016595209293545157, 0.9914470175720979, 0.005359173067957286, 0.9880392318793834, 0.045890182953931, 0.9178036590786199, 0.0031177588641332853, 0.9259743826475858, 0.031177588641332854, 0.0031177588641332853, 0.03741310636959942, 0.9493631623500771, 0.027922445951472856, 0.005957999084573641, 0.37620508505450706, 0.6068647639001438, 0.010213712716411956, 0.000851142726367663, 0.9784508748357399, 0.9797679294916588, 0.9797679294916588, 0.9870039106952299, 0.935148687465629, 0.0133592669637947, 0.0267185339275894, 0.0133592669637947, 0.0133592669637947, 0.9759946811481632, 0.015741849695938115, 0.900636756186776, 0.09802849046930895, 0.8499836711776544, 0.004028358631173718, 0.1409925520910801, 0.014031531177764784, 0.9822071824435348, 0.9896805894910876, 0.995435983593758, 0.0033347939148869613, 0.0018578997642345724, 0.035300095520456876, 0.1281950837321855, 0.512780334928742, 0.32141665921258106, 0.9920950757364967, 0.012010742423874534, 0.034697700335637544, 0.028025065655707244, 0.7059647491366253, 0.21886241750171373, 0.09536848931568587, 0.9002785391400746, 0.007950768442031603, 0.06161845542574492, 0.2822522796921219, 0.52077533295307, 0.12919998718301354, 0.008326406054756654, 0.4265681871129178, 0.35547348926076483, 0.1806189621108751, 0.028822174804926878, 0.9917091003077663, 0.03187443478780102, 0.9562330436340306, 0.9724892680193541, 0.025128921654246874, 0.02979464055517892, 0.5914236150203016, 0.28006962121868184, 0.03724330069397365, 0.06256874516587574, 0.9843142214770947, 0.009892605240975827, 0.8793580658129265, 0.05080735491363575, 0.06253212912447477, 0.003908258070279673, 0.04803866193573212, 0.928747464090821, 0.023179433316223068, 0.08850329084376081, 0.79020795396215, 0.06532385752753773, 0.033715539369051736, 0.9839203663178259, 0.006978158626367559, 0.038762889303240705, 0.009690722325810176, 0.9400000656035871, 0.9879442138585677, 0.01225286571098148, 0.967976391167537, 0.03694350161005088, 0.22281549408561938, 0.5599249462773337, 0.1512374597161458, 0.02886211063285225, 0.991974593185267, 0.010552921204098585, 0.9686994312211464, 0.02018123815044055, 0.9722699666022957, 0.9722699666022957, 0.984147731312695, 0.007344386054572351, 0.005033299862142419, 0.9915600728420566, 0.9954384671237716, 0.0028522592181196896, 0.9868757175645408, 0.0102799553912973, 0.9805749233110642, 0.007583532885385644, 0.007583532885385644, 0.007583532885385644, 0.978275742214748, 0.9956872100337705, 0.025052595028362757, 0.12192262913803208, 0.5169185440852182, 0.2588768152930818, 0.07766304458792454, 0.9377129743367524, 0.04688564871683762, 0.009377129743367523, 0.9893746515145875, 0.9892285816554255, 0.004502851585581534, 0.26566824354931057, 0.6880357222768585, 0.04052566427023381, 0.0018011406342326138], "Term": ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "act", "act", "act", "addin", "addin", "afford", "afford", "agave", "agave", "aggressively", "aggressively", "alert", "alert", "align", "align", "also", "also", "also", "also", "also", "amazon", "amazon", "amazon", "amazon", "anc", "andersen", "andersen", "andersen", "aps", "apse", "apse", "asset", "asset", "asset", "assign", "astronomical", "astronomical", "attached", "attached", "attached", "attached", "auditor", "auth", "auth", "available", "available", "available", "available", "available", "awarded", "b", "b", "b", "b", "b", "backout", "bakernet", "bakernet", "bankrupt", "bankrupt", "bergfelt", "beth", "beth", "bgcolor", "bgcolor", "bill", "bill", "bill", "bill", "bill", "blair", "blair", "board", "board", "board", "board", "bodydefault", "border", "border", "border", "border", "br", "br", "brackett", "bradford", "brochure", "buchanan", "buchanan", "bush", "bush", "bush", "business", "business", "business", "business", "business", "c", "c", "c", "c", "c", "caiso", "caiso", "california", "california", "california", "california", "call", "call", "call", "call", "call", "capacity", "capacity", "capacity", "capacity", "capacity", "cellpadding", "cellpadding", "cellspacing", "cellspacing", "center", "center", "center", "center", "center", "certificate", "certificate", "chanley", "chanley", "cherry", "cherry", "circuit", "class", "class", "class", "class", "classmate", "clear", "clear", "clear", "clear", "clear", "click", "click", "click", "click", "click", "colspan", "committee", "committee", "committee", "committee", "commoditylogic", "company", "company", "company", "company", "competitive", "competitive", "conference", "conference", "conference", "congestion", "consumer", "consumer", "consumer", "consumer", "contact", "contact", "contact", "contact", "contact", "coords", "copyright", "copyright", "coral", "coral", "coral", "coronado", "corp", "corp", "corp", "corp", "corp", "court", "court", "credit", "credit", "credit", "credit", "ctr", "ctr", "curtailment", "customer", "customer", "customer", "customer", "customer", "darrell", "date", "date", "date", "date", "day", "day", "day", "day", "day", "dbcaps", "declared", "declared", "denver", "denver", "design", "design", "desk", "desk", "desk", "desk", "desk", "detected", "devastated", "devastated", "development", "development", "development", "development", "deviation", "devon", "dietz", "doc", "doc", "doc", "doc", "donate", "donate", "dun", "dynegy", "dynegy", "dynegy", "dynegy", "e", "e", "e", "e", "e", "ecar", "ecar", "eei", "eei", "eix", "emailed", "emailed", "emaillink", "emerging", "emerging", "employee", "employee", "employee", "employee", "employee", "employer", "energy", "energy", "energy", "energy", "energy", "enform", "enform", "enron", "enron", "enron", "enron", "enron", "enronxgate", "enronxgate", "enronxgate", "enronxgate", "enronxgate", "eol", "eol", "eol", "eol", "eol", "epenergy", "epenergy", "euci", "even", "even", "even", "even", "even", "executive", "executive", "executive", "exotica", "f", "f", "f", "f", "f", "fagan", "federal", "federal", "ffffff", "final", "final", "final", "final", "financial", "financial", "financial", "financial", "financially", "financially", "font", "font", "former", "former", "fran", "fri", "fri", "fri", "friend", "friend", "friend", "friend", "fund", "fund", "fund", "fund", "fw", "fw", "fw", "fw", "fx", "game", "game", "game", "gas", "gas", "gas", "gas", "gas", "generation", "generation", "generation", "generation", "get", "get", "get", "get", "get", "gif", "gif", "gif", "gift", "gift", "gift", "gift", "go", "go", "go", "go", "go", "going", "going", "going", "going", "government", "government", "government", "graf", "greg", "greg", "greg", "greg", "gregwhalley", "hanagriff", "headcount", "height", "height", "hou", "hou", "hou", "hou", "hou", "house", "house", "house", "hp", "hp", "hp", "hr", "hr", "hr", "hr", "href", "href", "hurt", "hurt", "image", "image", "image", "image", "image", "img", "img", "impacted", "impacted", "intended", "intended", "intended", "intended", "interchange", "investor", "investor", "investor", "itcapps", "january", "january", "january", "january", "january", "john", "john", "john", "john", "john", "kern", "kern", "kim", "kim", "kim", "kim", "kimberly", "kimberly", "kimberly", "know", "know", "know", "know", "know", "kowalke", "krishna", "krishnarao", "labour", "laurel", "laurel", "law", "law", "law", "law", "leg", "leg", "let", "let", "let", "let", "let", "like", "like", "like", "like", "like", "link", "link", "link", "link", "link", "linkbarseperator", "locate", "lokay", "london", "london", "london", "london", "london", "lover", "lover", "lynn", "lynn", "made", "made", "made", "made", "made", "mail", "mail", "mail", "mail", "mail", "mailto", "mailto", "mailto", "mailto", "mailto", "make", "make", "make", "make", "make", "management", "management", "management", "management", "management", "map", "map", "map", "mark", "mark", "mark", "mark", "mark", "market", "market", "market", "market", "market", "master", "master", "master", "master", "master", "matrix", "matrix", "matrix", "matrix", "matrix", "maximize", "maximize", "may", "may", "may", "may", "may", "member", "member", "member", "member", "message", "message", "message", "message", "message", "military", "million", "million", "million", "million", "money", "money", "money", "money", "mpc", "mr", "mr", "mr", "mws", "nbsp", "nbsp", "nc", "nc", "nc", "ne", "ne", "ne", "ne", "nerc", "nerc", "net", "net", "net", "net", "net", "netted", "netted", "neumin", "new", "new", "new", "new", "new", "news", "news", "news", "news", "november", "november", "november", "november", "november", "npcc", "npcc", "nytimes", "nytimes", "obtained", "obtained", "obtained", "one", "one", "one", "one", "one", "order", "order", "order", "order", "original", "original", "original", "original", "original", "outage", "outage", "outage", "pampa", "parsing", "parsing", "partnership", "partnership", "partnership", "pcg", "pdf", "pdf", "pdf", "pdf", "pension", "permian", "pinnamaneni", "plan", "plan", "plan", "plan", "plan", "please", "please", "please", "please", "please", "pm", "pm", "pm", "pm", "pm", "pnm", "pnm", "pnm", "pocketbook", "pocketbook", "power", "power", "power", "power", "power", "powersrc", "powersrc", "preferred", "price", "price", "price", "price", "price", "process", "process", "process", "process", "process", "product", "product", "product", "product", "product", "pseg", "pseg", "pseg", "pt", "pt", "pt", "raquel", "raquel", "reassigned", "recipient", "recipient", "recipient", "recipient", "rect", "redshirt", "ref", "ref", "ref", "reliantenergy", "reliantenergy", "reliantenergy", "relieve", "repair", "repair", "request", "request", "request", "request", "request", "retirement", "retirement", "review", "review", "review", "review", "review", "rfp", "rfp", "rfp", "richard", "richard", "richard", "richard", "right", "right", "right", "right", "right", "rk", "rk", "royalty", "said", "said", "said", "said", "sat", "sat", "saving", "saving", "saving", "savita", "say", "say", "say", "say", "schedule", "schedule", "schedule", "schedule", "scheduled", "scheduled", "scheduled", "scheduled", "school", "school", "school", "schoolcraft", "scoop", "script", "script", "se", "se", "se", "se", "se", "sec", "sec", "sec", "see", "see", "see", "see", "see", "seller", "seller", "seller", "seller", "sept", "sept", "server", "server", "service", "service", "service", "service", "service", "sfs", "share", "share", "share", "share", "shipping", "shipping", "shipping", "shipping", "simulation", "simulation", "simulation", "skadden", "sp", "sp", "sp", "src", "src", "srrs", "stack", "stack", "status", "status", "status", "status", "status", "std", "std", "stock", "stock", "stock", "stock", "stock", "stocklookup", "stsw", "stwbom", "svc", "sw", "sw", "sw", "sw", "sw", "syncrasy", "syncrasy", "t", "t", "table", "table", "table", "tagline", "tagline", "tape", "td", "td", "team", "team", "team", "team", "team", "tepc", "thanks", "thanks", "thanks", "thanks", "thanks", "thru", "thru", "thursday", "thursday", "thursday", "thursday", "thursday", "time", "time", "time", "time", "time", "todaysheadlines", "tonight", "tonight", "tr", "tr", "trading", "trading", "trading", "trading", "trading", "transition", "transition", "transmission", "transmission", "transmission", "transmission", "tw", "tw", "two", "two", "two", "two", "two", "txt", "txt", "tycholiz", "tycholiz", "tycholiz", "tyco", "tyrell", "tyrell", "u", "u", "u", "u", "u", "underhanded", "underhanded", "valign", "valign", "vegetarian", "veggie", "westdesk", "westdesk", "whalley", "whalley", "width", "width", "wiped", "wiped", "wj", "wolfe", "wolfe", "wolfe", "wolfe", "worker", "would", "would", "would", "would", "would", "wscc", "wscc", "wscc", "wyndham", "xll", "year", "year", "year", "year", "year"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 2, 3, 4, 5]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el61140356880911240231555628", ldavis_el61140356880911240231555628_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el61140356880911240231555628", ldavis_el61140356880911240231555628_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el61140356880911240231555628", ldavis_el61140356880911240231555628_data);
            })
         });
}
</script>



#### Assign topics to your original data

* One practical application of topic modeling is to determine what topic a given text is about
* To find that, find the topic number that has the highest percentage contribution in that text
* The function, `get_topic_details` shown here, nicely aggregates this information in a presentable table
* Combine the original text data with the output of the `get_topic_details` function
* Each row contains the dominant topic number, the probability score with that topic and the original text data

```python
def get_topic_details(ldamodel, corpus):
    topic_details_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_details_df = topic_details_df.append(pd.Series([topic_num, prop_topic]), ignore_index=True)
    topic_details_df.columns = ['Dominant_Topic', '% Score']
    return topic_details_df


contents = pd.DataFrame({'Original text':text_clean})
topic_details = pd.concat([get_topic_details(ldamodel,
                           corpus), contents], axis=1)
topic_details.head()


     Dominant_Topic    % Score     Original text
0    0.0              0.989108    [investools, advisory, free, ...
1    0.0              0.993513    [forwarded, richard, b, ...
2    1.0              0.964858    [hey, wearing, target, purple, ...
3    0.0              0.989241    [leslie, milosevich, santa, clara, ...
```

### Interpreting the topic model

* Use the visualization results from the pyLDAvis library shown in 4.4.0.2.
* Have a look at topic 1 and 3 from the LDA model on the Enron email data. Which one would you research further for fraud detection purposes?

**Possible Answers**

* __**Topic 1.**__
* ~~Topic 3.~~
* ~~None of these topics seem related to fraud.~~


**Topic 1 seems to discuss the employee share option program, and seems to point to internal conversation (with "please, may, know" etc), so this is more likely to be related to the internal accounting fraud and trading stock with insider knowledge. Topic 3 seems to be more related to general news around Enron.**

### Finding fraudsters based on topic

In this exercise you're going to **link the results** from the topic model **back to your original data**. You now learned that you want to **flag everything related to topic 3**. As you will see, this is actually not that straightforward. You'll be given the function `get_topic_details()` which takes the arguments `ldamodel` and `corpus`. It retrieves the details of the topics for each line of text. With that function, you can append the results back to your original data. If you want to learn more detail on how to work with the model results, which is beyond the scope of this course, you're highly encouraged to read this [article](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/).

Available for you are the `dictionary` and `corpus`, the text data `text_clean` as well as your model results `ldamodel`. Also defined is `get_topic_details()`.

**Explanation 1/3**

* Print and inspect the results from the `get_topic_details()` function by inserting your LDA model results and `corpus`.

#### def get_topic_details


```python
def get_topic_details(ldamodel, corpus):
    topic_details_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_details_df = topic_details_df.append(pd.Series([topic_num, prop_topic]), ignore_index=True)
    topic_details_df.columns = ['Dominant_Topic', '% Score']
    return topic_details_df
```


```python
# Run get_topic_details function and check the results
topic_details_df = get_topic_details(ldamodel, corpus)
```


```python
topic_details_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>0.759451</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>0.631745</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.529827</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.993497</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.993399</td>
    </tr>
  </tbody>
</table>
</div>




```python
topic_details_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2085</th>
      <td>0.0</td>
      <td>0.789644</td>
    </tr>
    <tr>
      <th>2086</th>
      <td>3.0</td>
      <td>0.599926</td>
    </tr>
    <tr>
      <th>2087</th>
      <td>0.0</td>
      <td>0.999322</td>
    </tr>
    <tr>
      <th>2088</th>
      <td>1.0</td>
      <td>0.998151</td>
    </tr>
    <tr>
      <th>2089</th>
      <td>3.0</td>
      <td>0.988429</td>
    </tr>
  </tbody>
</table>
</div>



**Explanation 2/3**

* Concatenate column-wise the results from the previously defined function `get_topic_details()` to the original text data contained under `contents` and inspect the results.


```python
# Add original text to topic details in a dataframe
contents = pd.DataFrame({'Original text': text_clean})
topic_details = pd.concat([get_topic_details(ldamodel, corpus), contents], axis=1)
```


```python
topic_details.sort_values(by=['% Score'], ascending=False).head(10).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
      <th>Original text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>442</th>
      <td>2.0</td>
      <td>0.999963</td>
      <td>[pleased, send, web, based, e, mail, alert, pr...</td>
    </tr>
    <tr>
      <th>849</th>
      <td>2.0</td>
      <td>0.999877</td>
      <td>[original, message, received, thu, aug, cdt, e...</td>
    </tr>
    <tr>
      <th>2081</th>
      <td>0.0</td>
      <td>0.999631</td>
      <td>[unsubscribe, mailing, please, go, money, net,...</td>
    </tr>
    <tr>
      <th>211</th>
      <td>2.0</td>
      <td>0.999622</td>
      <td>[opinionjournal, best, web, today, january, ja...</td>
    </tr>
    <tr>
      <th>2087</th>
      <td>0.0</td>
      <td>0.999322</td>
      <td>[image, image, image, image, image, image, ima...</td>
    </tr>
  </tbody>
</table>
</div>




```python
topic_details.sort_values(by=['% Score'], ascending=False).head(10).tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
      <th>Original text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>280</th>
      <td>2.0</td>
      <td>0.999224</td>
      <td>[financial, express, wednesday, october, anti,...</td>
    </tr>
    <tr>
      <th>161</th>
      <td>2.0</td>
      <td>0.999107</td>
      <td>[today, headline, new, york, time, web, thursd...</td>
    </tr>
    <tr>
      <th>1211</th>
      <td>4.0</td>
      <td>0.999057</td>
      <td>[start, date, hourahead, hour, hourahead, sche...</td>
    </tr>
    <tr>
      <th>974</th>
      <td>3.0</td>
      <td>0.998204</td>
      <td>[forwarded, vince, j, kaminski, hou, pm, shirl...</td>
    </tr>
    <tr>
      <th>2088</th>
      <td>1.0</td>
      <td>0.998151</td>
      <td>[transmission, expansion, system, transition, ...</td>
    </tr>
  </tbody>
</table>
</div>



**Explanation 3/3**

* Create a flag with the `np.where()` function to flag all content that has topic 3 as a dominant topic with a 1, and 0 otherwise


```python
# Create flag for text highest associated with topic 3
topic_details['flag'] = np.where((topic_details['Dominant_Topic'] == 3.0), 1, 0)
```


```python
topic_details_1 = topic_details[topic_details.flag == 1]
```


```python
topic_details_1.sort_values(by=['% Score'], ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>% Score</th>
      <th>Original text</th>
      <th>flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>974</th>
      <td>3.0</td>
      <td>0.998204</td>
      <td>[forwarded, vince, j, kaminski, hou, pm, shirl...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>3.0</td>
      <td>0.998096</td>
      <td>[meet, elevator, original, message, maggi, mik...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1769</th>
      <td>3.0</td>
      <td>0.998008</td>
      <td>[please, make, sure, know, asst, forwarded, je...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1044</th>
      <td>3.0</td>
      <td>0.997852</td>
      <td>[forwarded, steven, j, kean, na, enron, pm, ka...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1084</th>
      <td>3.0</td>
      <td>0.997852</td>
      <td>[forwarded, steven, j, kean, na, enron, pm, ka...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2076</th>
      <td>3.0</td>
      <td>0.997744</td>
      <td>[know, houston, suck, xo, j, original, message...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>69</th>
      <td>3.0</td>
      <td>0.997727</td>
      <td>[late, meet, house, original, message, erin, r...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>58</th>
      <td>3.0</td>
      <td>0.997692</td>
      <td>[calling, fat, as, serious, restaurant, better...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2039</th>
      <td>3.0</td>
      <td>0.997663</td>
      <td>[w, e, e, k, e, n, e, v, l, b, l, f, r, decemb...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>870</th>
      <td>3.0</td>
      <td>0.997151</td>
      <td>[w, e, e, k, e, n, e, v, l, b, l, f, r, march,...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**You have now flagged all data that is highest associated with topic 3, that seems to cover internal conversation about enron stock options. You are a true detective. With these exercises you have demonstrated that text mining and topic modeling can be a powerful tool for fraud detection.**

## InShort

### Working with imbalanced data

* Worked with highly imbalanced fraud data
* Learned how to resample your data
* Learned about different resampling methods

### Fraud detection with labeled data

* Refreshed supervised learning techniques to detect fraud
* Learned how to get reliable performance metrics and worked with the precision recall trade-off
* Explored how to optimize your model parameters to handle fraud data
* Applied ensemble methods to fraud detection

### Fraud detection without labels

* Learned about the importance of segmentation
* Refreshed your knowledge on clustering methods
* Learned how to detect fraud using outliers and small clusters with K-means clustering
* Applied a DB-scan clustering model for fraud detection

### Text mining for fraud detection

* Know how to augment fraud detection analysis with text mining techniques
* Applied word searches to flag use of certain words, and learned how to apply topic modeling for fraud detection
* Learned how to effectively clean messy text data

### Future Works for fraud detection

* Network analysis to detect fraud
* Different supervised and unsupervised learning techniques (e.g. Neural Networks)
* Working with very large data
