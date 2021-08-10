Patikrina visus DF numeric laukus. Prasisukus kodui reikia isitikinti, ar visur nuliai be decimal daliu. 
Jeigu 0, tai ta lauka galima laikyti integer be decimal dalies. Jeigu ne 0, tada lauko tipas privalomas decimal.

```python
from transforms.api import transform_df, Input, Output
from pyspark.sql import functions as F


def generate_transforms(names):
    transforms = []
    for name in names:

        @transform_df(
            Output(f'/target_path/clean/{name.lower()}'),
            inp=Input(f'/source_path/raw/{name}'),
        )
        def my_compute_function(ctx, inp):
            cols_to_drop = [k for k, v in dict(inp.dtypes).items() if v in ['string', 'date', 'timestamp']]
            df = inp.drop(*cols_to_drop)
            for c in df.columns:
                df = df.withColumn(f'__{c}', F.abs(F.col(c)) % 1).drop(c)
            df = df.withColumn('fake_col', F.lit(0))
            df = df.groupBy('fake_col').max()
            df = df.toDF(*[c.replace('(', '').replace(')', '') for c in df.columns])
            return df

        transforms.append(my_compute_function)
    return transforms


TRANSFORMS = generate_transforms([
    'failas1',
    'failas2',
    'failas3',
])
```
