# minMLST
Machine-learning based minimal MLST scheme for bacterial strain typing



## Install

minMLST can be easily installed from [PyPI](https://pypi.org/project/minmlst):

<pre>
pip install minmlst
</pre>

## Examples
### 1. Quantifying gene importance

TBD
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
gi_results = gene_importance(data=data, measures=measures)
# save results to csv
gi_results.to_csv(join(path, 'gene_importance_Legionella_pneumophila' + '.csv'), index=False)
# display results
display(gi_results)
```

<p align="center">
  <img width="811" src="/docs/gene_importance_results.png" />
</p>


### 2. Gene reduction analysis

TBD
#### 2.1 Using default parameters
```python
# define a measure
measure = 'shap'
# reduce the number of genes in the scheme based on the selected measure
# and compute the ARI results per subset of genes  
gra_gi_results = gene_reduction_analysis(data, gi_results, measure)
# save results to csv
gra_gi_results.to_csv(join(path, f'analysis_Legionella_pneumophila_{measure}' + '.csv'), index=False)
# display results
display(gra_gi_results)
```

<p align="center">
  <img width="811" src="/docs/analysis_results_default.png" />
</p>

<p align="center">
  <img width="811" src="/docs/analysis_plot_default.png" />
</p>

#### 2.2 Find recommended threshold
```python
measure = 'shap'
gra_gi_results = gene_reduction_analysis(data, gi_results, measure, find_recommended_thresh=True)
gra_gi_results.to_csv(join(path, f'analysis_Legionella_pneumophila_{measure}' + '.csv'), index=False)
```

#### 2.3 ARI simulation study for p.v calculation
```python
measure = 'shap'
gra_gi_results = gene_reduction_analysis(data, gi_results, measure, find_recommended_thresh=True,
                                         simulated_samples=1000)
gra_gi_results.to_csv(join(path, f'analysis_Legionella_pneumophila_{measure}' + '.csv'), index=False)
```
