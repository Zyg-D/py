Flattening dict such as {k, iterable} to list of lists [[,], [,], [,], [,]]

```python
import itertools as itertools
dataset_dict = {1:(11,12),2:(21,22)}
x = list(itertools.chain.from_iterable([[[k, i] for i in v] for k, v in list(dataset_dict.items())]))
print(dataset_dict)
print(x)
```
```
{1: (11, 12), 2: (21, 22)}  
[[1, 11], [1, 12], [2, 21], [2, 22]]
```

-----------------------------------------------------------------------

