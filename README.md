<a href="https://pypi.org/project/minmlst"><img src="https://img.shields.io/pypi/v/minmlst"></a>
<a href="https://pypi.org/project/minmlst"><img src="https://img.shields.io/pypi/pyversions/minmlst"></a>
<a href="https://pypi.org/project/minmlst"><img src="https://img.shields.io/badge/platform-windows%20%7C%20linux-lightgrey"></a>
<a href="https://pypi.org/project/minmlst"><img src="https://img.shields.io/pypi/l/minmlst"></a>

# minMLST

minMLST is a machine-learning based methodology for identifying a minimal subset of genes
that preserves high discrimination among bacterial strains. It combines well known
machine-learning algorithms and approaches such as XGBoost, distance-based hierarchical
clustering, and SHAP. 

minMLST quantifies the importance level of each gene in an MLST scheme and allows the user 
to investigate the trade-off between minimizing the number of genes in the scheme vs preserving 
a high resolution among strains.


## Installation

minMLST can be easily installed from [PyPI](https://pypi.org/project/minmlst):

<pre>
pip install minmlst
</pre>

## Usage
### 1. Quantifying gene importance

This function provides gene importance values according to selected measures: shap, weight, gain,
cover, total gain or total cover.
* **shap** - the mean magnitude of the SHAP values, i.e. the mean absolute value of the 
           SHAP values of a given gene ([Lundberg and Lee, 2017](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)).
* **weight** - the number of times a given gene is used to split the data across all splits.
* **gain** (or **total gain**) - the average (or total) gain is the average (or total) reduction of 
                             Multiclass Log Loss contributed by a given gene across all splits.
* **cover** (or **total cover**) - the average (or total) quantity of observations concerned by a given 
                               gene across all splits.
                               
As a pre-step, CTs (cluster types) with a single representative isolate are filtered from the dataset.
Next, an XGBoost model is trained with parameters `max_depth`, `learning_rate`, `stopping_method` and `stopping_rounds`
([more information about XGBoost parameters](https://xgboost.readthedocs.io/en/latest/python/python_api.html)); 
model's performance is evaluated by Multi-class log loss over a validation set.
Finally, gene importance values are measured for the trained model and provided as a DataFrame output.

A data sample of Legionella pneumophila cgMLST scheme can be downloaded from 
[here](https://github.com/shanicohen33/minMLST/blob/master/data/Legionella_pneumophila_cgMLST_sample.csv). 


```python
import pandas as pd
from os.path import join, sep
from IPython.display import display
from minmlst.tools import gene_importance, gene_reduction_analysis

path = join('C:', sep, 'Users', 'user_name', 'Desktop', 'minMLST_data')
data_file = 'Legionella_pneumophila_cgMLST_sample'

# load scheme
data = pd.read_csv(join(path, data_file + '.csv'), encoding="cp1252")
# define measures
measures = ['shap', 'total_gain', 'total_cover', 'weight', 'gain', 'cover']
# quantify gene importance according to selected measures using minmlst 
gi_results = gene_importance(data, measures)
# save results to csv
gi_results.to_csv(join(path, 'gene_importance_Legionella_pneumophila' + '.csv'), index=False)
# display results
display(gi_results)
```

<p align="center">
  <img width="811" src="/docs/gene_importance_results.png" />
</p>

### minmlst.tools.gene_importance
##### Parameters:
* `data` (DataFrame): 
    
    DataFrame in the shape of (**m**,**n**):
    **n-1** columns of genes, last column **n** must contain the CT (cluster type).
    Each row **m** represents a profile of a single isolate.
    Data types should be integers only.
    Missing values should be represented as 0, no missing values are allowed for the CT (last column).
    
* `measures` (1-D array-like): 

    An array containing at least one of the following measures: 
    'shap', 'weight', 'gain', 'cover', 'total_gain' or 'total_cover'.
    
* `max_depth` (int, optional, default = 6): 

    Maximum tree depth for base learners. Must be greater equal to 0.
    
* `learning_rate` (float, optional, default = 0.3): 

    Boosting learning rate. Must be greater equal to 0.
    
* `stopping_method` (str, optional, default = 'num_boost_round'): 

    Stopping condition for the model's training process. Must be either 'num_boost_round' or 'early_stopping_rounds'.
    
* `stopping_rounds` (int, optional, default = 100): 

    Number of rounds for boosting or early stopping. Must be greater than 0.

##### Returns:
`importance_scores` (DataFrame):

Importance value per gene according to each of the input measures.
Higher scores are given to more important (informative) genes.
                 
<br>

### 2. Gene reduction analysis

This function analyzes how minimizing the number of genes in the MLST scheme impacts strain typing performance.
At each iteration, a reduced subset of most important genes is selected; and based on the allelic profile composed of these genes,
isolates are clustered into cluster types (CTs) using a distance-based hierarchical clustering.
The distance between every pair of isolates is measured by a normalized Hamming distance, which quantifies
the proportion of the genes which disagree on their allelic assignment. The distance between any two clusters 
is determined by a selected linkage method ([more information about the linkage methods supported by this tool](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)).

To obtain the partition into strains (i.e. the induced CTs), we apply a threshold on the distance between isolates belonging to the same cluster-type.
we use percentile-based thresholds instead of constant thresholds, i.e. for a given percentile of distances' distribution, we 
dynamically calculate the threshold value for each subset of genes.
The percentile (or percentiles) can be defined by the user in the `percentiles` input parameter, or alternatively being selected by the **<em>find recommended percentile</em>** procedure
that searches in the search space of `percentiles_to_check` input parameter (see 2.3).

Typing performance is measure by the Adjusted Rand Index (ARI), which quantifies similarity between the induced
partition into cluster types (that is based on a subset of genes) and the original partition in cluster types (that is based on
all genes, and was given as an input) ([Hubert and Arabie, 1985](https://link.springer.com/article/10.1007/BF01908075)).
p-value for the ARI is calculated using a permutation test based on Monte Carlo simulation study ([Qannari et al., 2014](https://www.sciencedirect.com/science/article/abs/pii/S0950329313000852)).
The analysis results are provided as a DataFrame output and are also plotted by default.


#### 2.1 Using default parameters
```python
# define a measure
measure = 'shap'
# reduce the number of genes in the scheme based on the selected measure
# and compute the ARI results per subset of genes  
analysis_results = gene_reduction_analysis(data, gi_results, measure)
# save results to csv
analysis_results.to_csv(join(path, f'analysis_Legionella_pneumophila_{measure}' + '.csv'), index=False)
# display results
display(analysis_results)
```

<p align="center">
  <img width="811" src="/docs/analysis_results.png" />
</p>

### minmlst.tools.gene_reduction_analysis
##### Parameters:
* `data` (DataFrame): 
    
    DataFrame in the shape of (**m**,**n**):
    **n-1** columns of genes, last column **n** must contain the CT (cluster type).
    Each row **m** represents a profile of a single isolate.
    Data types should be integers only.
    Missing values should be represented as 0, no missing values are allowed for the CT (last column).
    
* `gene_importance` (DataFrame): 

   Importance scores in the format returned by the **<em>gene_importance</em>** function.
    
* `measure` (str):

   A single measure according to which gene importance will be defined.
   It can be either 'shap', 'weight', 'gain', 'cover', 'total_gain' or 'total_cover'. 
   Note that the selected measure must be included in the `gene_importance` input.
    
* `reduction` (int or float, optional, default = 0.2): 
 
    The number (int) or percentage (0<float<1) of least important genes to be removed at each iteration.
    The first iteration includes all genes, the second iteration includes all informative genes (importance score > 0), 
    and the subsequent iterations include a reduced subset of genes according to the `reduction` parameter.

* `linkage_method` (str, default = 'complete'): 
 
    The the linkage method to compute the distance between clusters in the hierarchical clustering. 
    It can be either 'single', 'complete', 'average', 'weighted', 'centroid', 'median' or 'ward'.

* `percentiles` (float or 1-D array-like of floats, optional, default = [0.5, 1]):

    The percentile (or percentiles) of distances distribution to be used.
    Each percentile must be greater than 0 and smaller than 100. 
    For a given percentile, we dynamically calculate the threshold value for each subset of genes.
    The threshold value refers to the distance between isolates of the same CT, which is defined as the proportion of 
    genes that disagree on their allelic assignment. 
    
* `find_recommended_percentile` (boolean, optional, default = False): 

    if True, ignore parameter `percentiles` and run the **<em>find recommended percentile</em>** procedure 
    (see 2.3).
    
* `percentiles_to_check` (1-D array-like of floats, optional, default = numpy.arange(.5, 20.5, 0.5)): 

    The percentiles of distances distribution to be evaluated by the **<em>find recommended percentile</em>** procedure 
    (see 2.3). The array must contain at least 2 percentiles; each percentile 
    must be greater than 0 and smaller than 100.

* `simulated_samples` (int, optional, default = 0): 

    The number of samples (partitions) to simulate for the computation of the p-value of the observed ARI
    (see 2.2).
    For the significance of the p-values results, it's recommended to use ~1000 samples or more (see 
    [Qannari et al., 2014](https://www.sciencedirect.com/science/article/abs/pii/S0950329313000852)).
    In case `simulated_samples`=0, simulation won't run and p-values won't be calculated.

* `plot_results` (boolean, optional, default = True): 

    If True, plot the ARI and p-value results for each selected percentile as a function of the number of genes.

* `n_jobs` (int, optional, default = min(60, <em>number of CPUs in the system</em>)): 

    The maximum number of concurrently running jobs.

##### Returns:
`analysis_results` (DataFrame):

ARI and p-value (if `simulated_samples` > 0) computed for each subset of most important genes, and for each selected
percentile.
                 
<br>

#### 2.2 ARI simulation study for p.v calculation

P-value is calculated using a significance test of the Adjusted Rand Index (ARI), suggested by 
[Qannari et al., 2014](https://www.sciencedirect.com/science/article/abs/pii/S0950329313000852).
The suggested permutation test involves a simulation of a large number (~1000) of partitions with several constrains. 
The number of simulated partitions is set by the input parameter `simulated_samples`.

```python
measure = 'shap'
# define the percentiles of distances distribution to be used (optional)
percentiles = [0.1, 0.5, 1, 1.5]
# define the number of simulated samples
simulated_samples = 1000
analysis_results = gene_reduction_analysis(data, gi_results, measure, percentiles=percentiles, 
                                           simulated_samples=simulated_samples)
analysis_results.to_csv(join(path, f'analysis_Legionella_pneumophila_{measure}' + '.csv'), index=False)
display(analysis_results)
```

<p align="center">
  <img width="811" src="/docs/analysis_results_pv.png" />
</p>
      
<br>

#### 2.3 Find recommended percentile

To obtain the induced partition into cluster-types, we use percentile-based thresholds. For a given percentile of distances' 
distribution, we dynamically calculate the threshold value for each subset of genes.


The 'find recommended percentile' procedure uses an heuristic to find a recommended percentile. This is the percentile 
with the best overall predictive performance, which is equivalent to the ARI curve with the highest AUC, and is 
referred as ‘best’. At first, we initialize ‘best’ to the minimal percentile in search space (`percentiles_to_check`). 
Then we compare ‘best’ to the successor percentile in `percentiles_to_check`, referred as ‘next’, 
by computing the "non-absolute" L1 distance between their ARI vectors. 
This distance equals to the sum of the differences between the two vectors when subtracting the 'best' from the 'next'. 
In case the distance is not negative (i.e., 'next' performs better or the same), 'next' is defined as the new 'best'. 
Otherwise, the search is completed and 'best' is selected as the recommended percentile.

In case `find_recommended_percentile` is True, the outputs are the ARI and p-value results computed for each subset of 
genes when using the recommended percentile.

```python
measure = 'shap'
# define the percentiles of distances' distribution to be evaluated (optional)
search_space = [0.1, 0.3, 0.5, 1, 1.5, 2, 2.5, 3]
# set find_recommended_percentile to True
analysis_results = gene_reduction_analysis(data, gi_results, measure, find_recommended_percentile=True, 
                                           percentiles_to_check=search_space, simulated_samples=1000)
analysis_results.to_csv(join(path, f'analysis_Legionella_pneumophila_{measure}' + '.csv'), index=False)
display(analysis_results)
```

<p align="center">
  <img width="811" src="/docs/analysis_results_find_percentile.png" />
</p>


