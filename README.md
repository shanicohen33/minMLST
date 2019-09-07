# minMLST
Machine-learning based minimal MLST scheme for bacterial strain typing



## Install

minMLST can be easily installed from [PyPI](https://pypi.org/project/minmlst):

<pre>
pip install minmlst
</pre>

## Examples
### Quantifying gene importance

TBD
```python
import pandas as pd
from os.path import join, sep
from minmlst.tools import gene_importance, gene_reduction_analysis

path = join('C:', sep, 'Users', 'user_name', 'Desktop', 'minMLST_data')
data_file = 'Legionella_pneumophila_cgMLST_sample'

# load scheme
data = pd.read_csv(join(path, data_file + '.csv'), encoding="cp1252")
# define measures
measures = ['shap', 'total_gain']
# quantify gene importance according to selected measures using minmlst 
gi_results = gene_importance(data=data, measures=measures)
# save results to csv
gi_results.to_csv(join(path, 'gene_importance_Legionella_pneumophila' + '.csv'), index=False)
```

### Gene reduction analysis

TBD
##### Using default parameters
```python
measure = 'shap'
gra_gi_results = gene_reduction_analysis(data, gi_results, measure)
gra_gi_results.to_csv(join(path, f'gra_Legionella_pneumophila_{measure}' + '.csv'), index=False)
```

##### Find recommended threshold
```python
measure = 'shap'
gra_gi_results = gene_reduction_analysis(data, gi_results, measure, find_recommended_thresh=True)
gra_gi_results.to_csv(join(path, f'gra_Legionella_pneumophila_{measure}' + '.csv'), index=False)
```

##### ARI simulation study for p.v calculation
```python
measure = 'shap'
gra_gi_results = gene_reduction_analysis(data, gi_results, measure, find_recommended_thresh=True,
                                         simulated_samples=1000)
gra_gi_results.to_csv(join(path, f'gra_Legionella_pneumophila_{measure}' + '.csv'), index=False)
```