reduce vs loop

```python
cols = ['c1', 'c2', 'c3']

schema = df.select(cols[0])
if len(cols) > 1:
    for i in cols[1:]:
        schema = schema.crossJoin(df.select(i)).distinct()

schema = reduce(lambda i, j: i.crossJoin(j), [df.select(c) for c in cols]).distinct()
```
