```python
from pyspark.sql import SparkSession, functions as F, Window as W
spark = SparkSession.builder.getOrCreate()
```

# PySpark

**conf**

```python
spark.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")

spark.conf.setAll([
    ("spark.sql.autoBroadcastJoinThreshold", "-1"),
    ("spark.driver.maxResultSize", "0"),
])
```

-------------------------------------------------------------------------------
**Create DF/RDD**

Create example DF:

```python
# Several cols:
df = spark.createDataFrame(
    [(1, 11),
     (2, 22)],
    ['c1', 'c2'])
# OR
df = spark.createDataFrame([(1, 11), (2, 22)], ['c1', 'c2'])
# OR
df = spark.createDataFrame([('1', 11), ('2', 22)], 'c1 string, c2 int')

# One col
df = spark.createDataFrame([(1, ), (2, )], ['c'])
df = spark.createDataFrame([1, 2], 'int').toDF('c')
df = spark.range(1, 4).toDF('c')
df = spark.range(4)

# One row
df = spark.createDataFrame([{'c1': 1, 'c2': 2}])

# Date type
df = spark.createDataFrame([(1, '2020-05-05'), (2, None)], ['c1', 'c2'])
df = df.select(*[F.col(c).cast('date') if c in {'c2'} else c for c in df.columns])
```

Create empty DF:

```python
# Take schema from other df
df2 = spark.createDataFrame([], df.schema)
df2 = df.limit(0)

# Provide schema
df = spark.createDataFrame([], 'c1 string, c2 long')
```

Create example RDD:

```python
sc = spark.sparkContext
dept = [('Marketing', 10),
        (  'Finance', 20),
        (    'Sales', 30)]
rdd = sc.parallelize(dept)
# print(rdd.collect())
```

RDD from Rows:

```python
from pyspark.sql import Row
rdd = spark.sparkContext.parallelize([
    Row('Key-001', 'B4', 42, 'K5', 19, 'C20', 20), 
    Row('Key-002', 'X16', 42),
    Row('Key-003', 'O14', 41, 'P13', 8),
])
# print(rdd.collect())
```


RDD to DF:

```python
# Auto-generated col names
df = rdd.toDF()
# Providing col names
df = rdd.toDF(['dept_name', 'dept_id'])
```

List of Rows to DF:

    spark.createDataFrame(data)

DF to RDD:

    rdd = df.rdd

RDD, DF from local txt:

```python
rdd = spark.sparkContext.textFile(r'C:\Temp\sample.txt')
header_str = rdd.first()
rddDataLines = rdd.filter(lambda line: line != header_str)
rddSplit = rddDataLines.map(lambda k: k.split(','))
df = rddSplit.toDF(header_str.split(','))
```
When encoding is needed:
```python
rdd = spark.sparkContext.textFile(
    r'C:\Temp\sample.csv', 
    use_unicode=False
).map(lambda x: x.decode('cp1257'))
```
Assert correct amount of separators in RDD lines (just after importing .txt or .csv):
```python
sep = ','
countRDD = rdd.map(lambda line: len(line.split(sep)))
sortedRDD = sorted(countRDD.collect())
print(sortedRDD[:3], sortedRDD[-3:])

# Show lines of specified length
sep = ','
filter_key = 25
lenRDD = rdd.map(lambda line: (len(line.split(sep)), line))
filtRDD = lenRDD.filter(lambda line: line[0] == filter_key)
print(rdd.take(1))
print(filtRDD.collect())
```

RDD, DF from example JSON

```python
json = \
{
  "metadata": {"Symbol": "IBM"},
  "data": {
    "2021-04-14": {
      "open": "131.305",
      "close": "132.630"
    },
    "2021-04-13": {
      "open": "133.000",
      "close": "131.180"
    }
  }
}
rdd = spark.sparkContext.parallelize([json])
df = spark.read.json(rdd)
```

RDD, DF from online JSON (more in drive)

```python
from urllib.request import urlopen
url = 'https://randomuser.me/api/0.8/?results=3'
jsonData = urlopen(url).read().decode('utf-8')
# NOTE SQUARE BRACKETS
rdd = spark.sparkContext.parallelize([jsonData])
df = spark.read.json(rdd)
# df.printSchema()
# root
#  |-- nationality: string (nullable = true)
#  |-- results: array (nullable = true)
#  |    |-- element: ...
# Navigating:
# F.explode is used to get inside 'results' array
# Dot notation is used to get inside structs 'user' and 'name'
df = df.withColumn('expl',F.explode('results')).select('expl.user.name.*')
df.show()
#|  first|  last|title|
#+-------+------+-----+
#| monica| marin| miss|
#|eduardo|lozano|   mr|
#|lorenzo|ferrer|   mr|
```

RDD, DF from local CSV

```python
df = spark.read.csv(r'C:\Temp\test.csv', header=True, sep=',')
rdd = spark.read.csv(r'C:\Temp\test.csv', header=True, sep=',').rdd
```

Missing:  
RDD, DF from online csv - searched; probably does not exist   


DF from dictionary list (list of dicts)

```python
# deprecated
df = spark.createDataFrame(data)
# OR
from pyspark.sql import Row
df = spark.createDataFrame([Row(**i) for i in data])
```


----------------------------------------------------------------------------------
**Conversions**


array of struct to map

    F.map_from_entries("c1")


binary to long/int (dec)

    F.conv(F.hex("c1"), 16, 10)


columns to map

```python
df = spark.createDataFrame(
    [(1, 'a', 10), (1, 'b', 20), (2, 'c', 30)],
    ['id', 'col_k', 'col_v'])

df = df.groupBy('id').agg(
    F.map_from_entries(F.collect_set(F.struct('col_k', 'col_v'))).alias('map')
)
df.show()
# +---+------------------+
# | id|               map|
# +---+------------------+
# |  1|{b -> 20, a -> 10}|
# |  2|         {c -> 30}|
# +---+------------------+
```


map to array of struct (fields: key, value)

    map_entries("c1")

map to columns (keys as col names)


- if col names are not known - reading the whole column in order to infer the new schema from all the keys (map only had 2 fields: key+value)

	- _PySpark_
		```python
		df = spark.createDataFrame(
		    [("x", {"a":1},),
		     ("y", {"a":2, "b":3},)],
		    ["c1", "c2"])

		df = df.withColumn("_c", F.to_json("c2"))
		json_schema = spark.read.json(df.rdd.map(lambda r: r._c)).schema
		df = df.withColumn("_c", F.from_json("_c", json_schema))
		df = df.select("*", "_c.*").drop("_c")

		df.show()
		# +---+----------------+---+----+
		# | c1|              c2|  a|   b|
		# +---+----------------+---+----+
		# |  x|        {a -> 1}|  1|null|
		# |  y|{a -> 2, b -> 3}|  2|   3|
		# +---+----------------+---+----+
		print(df.dtypes)
		# [('c1', 'string'), ('c2', 'map<string,bigint>'), ('a', 'bigint'), ('b', 'bigint')]
		```

	- _Scala_
		```scala
		val json_col = to_json($"c2")
		val json_schema = spark.read.json(df.select(json_col).as[String]).schema
		val df2 = df.withColumn("_c", from_json(json_col, json_schema))
		val df3 = df2.select("*", "_c.*").drop("_c")
		```

- if col names are known:

    ```python
    cols = ["a", "b"]
    df = df.select([F.col("c2")[c].alias(c) for c in cols])
    ```

map to string (of json/map/dict form)

    F.to_json('c1')


map to struct (keys as col names)

- if field names are not known - reading the whole column in order to infer the new schema from all the keys (map just had 2 fields: key+value)

	- _PySpark_
		```python
		df = spark.createDataFrame(
		    [("x", {"a":1},),
		     ("y", {"a":2, "b":3},)],
		    ["c1", "c2"])

		df = df.withColumn("c3", F.to_json("c2"))
		json_schema = spark.read.json(df.rdd.map(lambda r: r.c3)).schema
		df = df.withColumn("c3", F.from_json("c3", json_schema))

		df.show()
		# +---+----------------+---------+
		# | c1|              c2|       c3|
		# +---+----------------+---------+
		# |  x|        {a -> 1}|{1, null}|
		# |  y|{a -> 2, b -> 3}|   {2, 3}|
		# +---+----------------+---------+
		print(df.dtypes)
		# [('c1', 'string'), ('c2', 'map<string,bigint>'), ('c3', 'struct<a:bigint,b:bigint>')]
		```

	- _Scala_
		```scala
		val json_col = to_json($"c2")
		val json_schema = spark.read.json(df.select(json_col).as[String]).schema
		val df2 = df.withColumn("c3", from_json(json_col, json_schema))
		```

- if field names are known

    ```python
    cols = ["a", "b"]
    df = df.withColumn("c3", F.struct([F.col("c2")[c].alias(c) for c in cols]))
    ```


struct to string (of json/map/dict form)

    F.to_json('c1')


struct to map (field_name -> value)

    F.from_json(F.to_json("c1"), 'map<string, string>')



----------------------------------------------------------------------------------
**Other**

Foundry expectations

```python
from transforms.api import Check
from transforms import expectations as E
...
    Output(
        'rid',
        checks=[
            Check(E.__funkcija__, 'error_text', on_error='FAIL'),
        ],
    )
```
```python
# Examples:
cols_list = ['c1', 'c2']
nine_digits = r'^\d{9}$'
encrypted = r'^BELLASO::'
*[Check(E.col(c).non_null(), 'Check ' + c + ' not null', on_error='WARN') for c in cols_list],
Check(E.group_by('c1', 'c2').is_unique(), 'text', on_error='WARN'),
Check(E.any(E.col('c1').rlike(nine_digits), E.col('c1').rlike(encrypted)), 'text'),  # turi passinti bent viena salyga
Check(E.when(E.col('c1').rlike(encrypted), E.col('c2').rlike(encrypted)).otherwise(E.true()), 'text'),
Check(E.col('c1').rlike('^(\d*(\.0+)?)|(0E-10)$'),
      'Expectation, kad stulpelis C1 po kablelio turi tik nulius.',
      on_error='WARN'
)
```

Modify/ rename all columns in DF: 
```python
df = df.toDF(*[f'v_{c}' for c in tp_v.columns])
# or
new_column_name_list = [map(lambda x: x.replace(' ', '_'), df.columns)]
df = df.toDF(*new_column_name_list)
```

Get DF column datatype:

    dict(df.dtypes)['col_name']

Get DF col name:

```python
# These can return different results
F.col('c1')._jc.toString()
F.col('c1').__repr__()
F.col('c1').__str__()
str(col).replace("`", "").split("'")[-2].split(" AS ")[-1])
import re
re.search(r"'.*?`?(\w+)`?'", str(col)).group(1)
```

Filter DF rows:

```python
1  df = df.filter(df.distance > 2000) # repeated df
2  df = df.filter('d<5 and (col1 <> col3 or (col1 = col3 and col2 <> col4))')
# requires import pyspark.sql.functions as F
3  df = df.filter(F.col('distance') > 2000)
4  df = df.filter(
        ((F.col('col1') != F.col('col3')) | 
         (F.col('col2') != F.col('col4')) & (F.col('col1') == F.col('col3')) ) )
```

First row:
```python
df.head()  # Row
df.first()  # Row
```

First few rows:
```python
df.head(1)  # list of Rows
df.take(1)  # list of Rows
df.limit(1).collect()  # list of Rows
```

Keeping only top few rows in df:
```python
df.limit(5)
```

Referencing the value of a column:
```python
v = df.head().c1
v = df.head()['c1']
```

Keep only top rows for specified partitions:
```python
df_filt = df.withColumn('rn', F.row_number().over(
    W.partitionBy('asm_id')
    .orderBy(
        F.col('data').desc_nulls_last(),
    )
)).filter('rn = 1').drop('rn')
```

Length of strings (string length)

    F.length('col_name')

Order by

```python
df.orderBy(F.col('e_snf.snf_san_data').desc_nulls_last())
                                      .asc_nulls_first()
```
`.orderBy` = `.sort`  
`F.asc('col')` =` F.col('col').asc()` = `F.col('col').asc_nulls_first()`  
`F.desc('col')` = `F.col('col').desc()` = `F.col('col').desc_nulls_last()`

Group + aggregate

    df.groupby(F.col('e_cbi.cbi_id')).agg(F.max('e_cbi.ist_data').alias('data'))

Change col type

    new_df = df.withColumn("colx", df["colx"].cast('date'))

Window

```python
from pyspark.sql import Window as W
w = W.partitionBy('id').orderBy(F.asc_nulls_last('id2')) \
     .rowsBetween(W.unboundedPreceding, W.unboundedFollowing)
df = df.withColumn('last_su_betw', F.last('id2').over(w))
```

Unpivot

```python
df = spark.createDataFrame(
    [(101, 3, 520, 2001),
     (102, 29, 530, 2020)],
    ['ID', 'Col1', 'Col2', 'Col40'])
# Option1
df2 = df.select(
    "ID",
    F.expr("stack(3, Col1, 'Col1', Col2, 'Col2', Col40, 'Col40') as (ColVal, ColDescr)")
)
# Option2
cols_to_unpivot = [f"`{c}`, \'{c}\'" for c in df.columns if c != 'ID']
stack_string = ", ".join(cols_to_unpivot)
df2 = df.select(
    "ID",
    F.expr(f"stack({len(cols_to_unpivot)}, {stack_string}) as (ColVal, ColDescr)")
)
```

Median, quartiles

```python
df = (
    spark.range(1, 6)
    # accurate percentiles for given values
    .withColumn('percent_rank', F.percent_rank().over(W.orderBy('id')))
    # ACCURATE values for given percentiles
    .withColumn('lower_quartile_acc', F.expr('percentile(id, .25) over()'))
    .withColumn('median_acc', F.expr('percentile(id, .5) over()'))
    .withColumn('quartiles_acc', F.expr('percentile(id, array(.25, .5, .75)) over()'))
    # APPROX values for given percentiles
    .withColumn('median_approx', F.percentile_approx('id', .5).over(W.partitionBy(F.lit(1))))
    .withColumn('median_approx2', F.expr('percentile_approx(id, .5) over()'))
    .withColumn('quartiles_approx', F.percentile_approx('id', [.25, .5, .75]).over(W.partitionBy(F.lit(1))))
    .withColumn('quartiles_approx2', F.expr('percentile_approx(id, array(.25, .5, .75)) over()'))
)
df.show()
#+---+------------+------------------+----------+---------------+-------------+--------------+----------------+-----------------+
#| id|percent_rank|lower_quartile_acc|median_acc|  quartiles_acc|median_approx|median_approx2|quartiles_approx|quartiles_approx2|
#+---+------------+------------------+----------+---------------+-------------+--------------+----------------+-----------------+
#|  1|         0.0|               2.0|       3.0|[2.0, 3.0, 4.0]|            3|             3|       [2, 3, 4]|        [2, 3, 4]|
#|  2|        0.25|               2.0|       3.0|[2.0, 3.0, 4.0]|            3|             3|       [2, 3, 4]|        [2, 3, 4]|
#|  3|         0.5|               2.0|       3.0|[2.0, 3.0, 4.0]|            3|             3|       [2, 3, 4]|        [2, 3, 4]|
#|  4|        0.75|               2.0|       3.0|[2.0, 3.0, 4.0]|            3|             3|       [2, 3, 4]|        [2, 3, 4]|
#|  5|         1.0|               2.0|       3.0|[2.0, 3.0, 4.0]|            3|             3|       [2, 3, 4]|        [2, 3, 4]|
#+---+------------+------------------+----------+---------------+-------------+--------------+----------------+-----------------+
```

Change deeply nested structure

```python
schema = (
    T.StructType([
        T.StructField('x', T.ArrayType(T.StructType([
            T.StructField('y', T.LongType()),
            T.StructField('z', T.ArrayType(T.StructType([
                T.StructField('log', T.StringType())
            ]))),
        ])))
    ])
)
df = spark.createDataFrame([
    [
        [[
            9,
            [[
                'text'
            ]]
        ]]
    ]
], schema)
df.printSchema()
#  root
#   |-- x: array (nullable = true)
#   |    |-- element: struct (containsNull = true)
#   |    |    |-- y: long (nullable = true)
#   |    |    |-- z: array (nullable = true)
#   |    |    |    |-- element: struct (containsNull = true)
#   |    |    |    |    |-- log: string (nullable = true)

df = df.withColumn('x', F.expr('transform(x, e -> struct(e.y as y, array(struct(struct(e.z.log[0] as b, e.z.log[0] as c) as log)) as z))'))
df.printSchema()
#  root
#   |-- x: array (nullable = true)
#   |    |-- element: struct (containsNull = false)
#   |    |    |-- y: long (nullable = true)
#   |    |    |-- z: array (nullable = false)
#   |    |    |    |-- element: struct (containsNull = false)
#   |    |    |    |    |-- log: struct (nullable = false)
#   |    |    |    |    |    |-- b: string (nullable = true)
#   |    |    |    |    |    |-- c: string (nullable = true)
```

Recursive function to standardize struct field names in the provided struct:

```python
def standardize_fields(struct):
    ''' Standardizes struct field names. '''
    if struct == None:
        return T.StructType()
    updated = []
    for f in struct.fields:
        new_name = f.name.replace(" ", "_").replace(":", "_").replace("-", "_")
        if isinstance(f.dataType, T.StructType):
            updated.append(T.StructField(new_name, standardize_fields(f.dataType)))
        elif isinstance(f.dataType, T.ArrayType):
            updated.append(T.StructField(new_name, T.ArrayType(
                (standardize_fields(f.dataType.elementType)))))
        else:
            # Else handle all the other types except for struct and array
            updated.append(T.StructField(new_name, f.dataType, f.nullable))   
    return T.StructType(updated)
```

Join DFs (more in drive)

```py
DF_joined = DF1.join(DF2, DF1.id == DF2.id, "inner")
# Possible complex criteria: 
DF_joined = empDF.join(deptDF,[(empDF.emp_id < deptDF.dept_id/10)|(empDF.salary==deptDF.dept_id/-10)],"inner")
```

_Spark's regex = PCRE (not PCRE2) ([docs](https://www.pcre.org/original/doc/html/pcrepattern.html))_

Regex check if match exists

```python
data = [('@@',), ('coo',),]
df=spark.createDataFrame(data, ['col'])
df = df.withColumn('col2', F.when(F.col('col').rlike('(\w+)'), 'match'))
df.show()
#    |col| col2|
#    +---+-----+
#    | @@| null|
#    |coo|match|
```

Regex return specific group or whole match from the 1st match

```python
df = df.withColumn('new_col', F.regexp_extract('c1', '\d+ ', 0))  # 0=full 1st match, 1...=groups
```

Regex return all matches (3.1)

```python
data = [('one two',), ('I am',), ('coo',),]
df=spark.createDataFrame(data, ['col'])
df = df.withColumn('col2', F.expr(r"regexp_extract_all(col, '(\\w+)', 1)"))
df.show()
#    |    col|      col2|
#    +-------+----------+
#    |one two|[one, two]|
#    |   I am|   [I, am]|
#    |    coo|     [coo]|
```

Regex replace

If match isn't found, col value is returned
```python
df = df.withColumn('new_col', F.regexp_replace('c1', '\d+ (\w+)', '$1'))
```


Foundry - many transforms with one script

```python
from transforms.api import transform_df, Input, Output


def generate_transforms(names):
    transforms = []
    for name in names:
        @transform_df(
            Output(f"/path/clean/{name}"),
            inp=Input(f"/path/raw/{name}"),
        )
        def compute(inp):
            return inp
        transforms.append(compute)
    return transforms


TRANSFORMS = generate_transforms([
    'file_name1',
    'file_name2',
    'file_name3',
])
```

**Datetime**

Configuration in Spark 3 to use Spark 2 datetime patterns:
```python
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
```

Specified format to date:
```python
import pandas as pd
@F.pandas_udf('date')
def _to_date(year_week: pd.Series) -> pd.Series:
    return pd.to_datetime(year_week + '-1', format='%G-%V-%u')

spark.createDataFrame([('2020-01',)]).withColumn('c2', _to_date('_1')).collect()
# [Row(_1='2020-01', c2=datetime.date(2019, 12, 30))]
```

Date to specified format
```python
import pandas as pd
@F.pandas_udf('string')
def _to_str(date: pd.Series) -> pd.Series:
    date = pd.to_datetime(date)
    return date.dt.strftime('%m/%d/%Y')

spark.createDataFrame([('2020-01-01',)]).withColumn('c2', _to_str('_1')).collect()
# [Row(_1='2020-01-01', c2='01/01/2020')]
```



-------------------------------------------------------------------------------
**Spark ML**

Create vector type cols:

```python
from pyspark.ml.linalg import Vectors
df = spark.createDataFrame(
    [(Vectors.dense(0.9, 0.5, 0.2), Vectors.dense(0.1, 0.3, 0.2)),
     (Vectors.dense(0.8, 0.7, 0.1), Vectors.dense(0.8, 0.4, 0.2)),
     (Vectors.dense(0.9, 0.2, 0.8), Vectors.dense(0.3, 0.1, 0.8))],
    ['vector1', 'vector2']
)
```

Convert vector to array:

```python
from pyspark.ml.functions import vector_to_array
df = df.select(
    vector_to_array('vector1').alias('vector1'),
    vector_to_array('vector2').alias('vector2'),
)
```

Convert array to vector (array must contain doubles):

```python
from pyspark.ml.functions import array_to_vector
df = spark.createDataFrame([([-0.02, 0.1, 0.0],)], ['col_arr_dbl'])
df = df.withColumn('vect1', array_to_vector('col_arr_dbl'))
```





-------------------------------------------------------------------------------
# Python

**Pandas**

Create an empty df:

    df = pd.DataFrame()

Create a sample df: 

```python
df = pd.DataFrame.from_records(
			[{'col_1': 'a', 'col_2': 1},
			 {'col_1': 'b', 'col_2': 2}])
df = pd.DataFrame({'c1': [2, 4, 8, 0],
                   'c2': [2, 0, 0, 0],
                   'c3': [10, 2, 1, 8]})
df = pd.DataFrame({'c1_num_legs': [2, 4, 8, 0],
                   'c2_num_wings': [2, 0, 0, 0],
                   'c3_num_specimen_seen': [10, 2, 1, 8] },
                  index=['falcon', 'dog', 'spider', 'fish'])
pandasDF = pysparkDF.toPandas()
```

Put all Excel sheets into a dict { "sheetName" : df , ... }

    dfs = pd.read_excel(r'C:\Temp\file.xlsx', sheet_name=None)

Referencing individual items - by indexes:  
<sup>[row index, col index]</sup>

    df.iat[0,0]
    
Referencing individual items - by row index, col name:

    df.at[0,'lecture_hours']
    
Adding a new empty col:

    df['new_col_name'] = ""
    
Adding a new col with NaN values:

    df['new_col_name'] = numpy.nan
    
Deleting a col by col name  
<sup>1 is the axis number (0 for rows and 1 for columns.)</sup>

    df = df.drop('column_name', 1)

Count of cols (w/o index):

    len(df.columns)

Loop through cols: 

    for col in df.columns:

Sorting

    df = df.sort_values(by=['col_by'], ascending=False)

Keeping only top 5 rows:

    df = df.sort_values(by=['col_by'], ascending=False).head(5)

MultiIndex.from_tuples  and slicing:

```py
import pandas as pd
import numpy as np

tuples = [(1, 'red'), (1, 'blue'),
          (2, 'red'), (2, 'blue')]
header = pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))
df = pd.DataFrame(np.random.randn(3, 4),
                  index=['a','b','c'],
                  columns=header)
print(df)
#   number         1                   2          
#   color        red      blue       red      blue
#   a       1.429644 -1.456978 -0.724916 -1.287699
#   b       1.707773 -1.108309  0.529229 -1.601489
#   c       1.159339 -1.395915 -1.426026  0.798999

print(df.xs('red', level='color', axis=1))
#   number         1         2
#   a      -1.315971 -0.829633
#   b       0.251705  1.201666
#   c       0.136187  0.092231
```

Transform df header from flat to multiindex:

```py
import pandas as pd
df = pd.DataFrame({'ABCBase_CIP00': [1, 1, 1],
                   'ABCBase_CIP02': [3, 3, 3],
                   'ABC2_CIP00': [7, 7, 7],
                   'ABC2_CIP02': [9, 9, 9] },
                  index=['X', 'Y', 'Z'] )
lt = []
for col in df.columns:
    cut = col.find('_')
    lt.append((col[:cut], col[cut+1:]))
df.columns = pd.MultiIndex.from_tuples(lt, names=('name', 'code'))
print(df)
#   name ABCBase        ABC2      
#   code   CIP00 CIP02 CIP00 CIP02
#   X          1     3     7     9
#   Y          1     3     7     9
#   Z          1     3     7     9
```

----------------------------------------------------------------------------
**lists**

Adding an empty list:

    lt = []

Adding items:

    lt.append("a")

Referencing elements by index:

    lt[0]

Number of items:

    len(lt)

Finding the (first) index given item name:

    ["foo", "bar", "baz"].index("bar")

Replacing items:

    lt[1], lt[2] = lt[2], lt[1]

----------------------------------------------------------------------------
**tuples**  
<sup>Tuples are unchangeable, or immutable</sup>

    tpl = ("apple", "banana", "cherry")

Referencing elements by index:

    tpl[0]

Number of items:
    
    len(tpl)

Changing values:

    x = ("apple", "banana", "cherry")
    y = list(x)
    y[1] = "kiwi"
    x = tuple(y)
    print(x)   # ('apple', 'kiwi', 'cherry')

----------------------------------------------------------------------------
**dicts**

Adding an empty dict: 

    mydict = {}
    mydict = dict()

Adding values: 

    mydict[key_name] = a_value

Deleting keys:

    del mydict[key_name]

Referencing the 1st element (order is only guaranteed in dicts from 3.7):

    list(mydict.keys())[0]
    
Referencing the value of a specified key:  
<sup>default is returned shen the key is not found</sup>

    mydict.get('key_name'[, default])

Dict comprehension

    dict_comp = {x:y for x, y in [('a', 1), ('b', 2)]}

----------------------------------------------------------------------------
**sets**

Adding an empty set:

    myset = set()

Adding item:

    myset.add(item)

Deleting item:

    myset.remove(item)

----------------------------------------------------------------------------
**importing .json**

    df=pandas.read_json("file_name.json", encoding = 'utf8')
    
----------------------------------------------------------------------------
**exporting .xlsx**

    with pandas.ExcelWriter('out_file.xlsx') as writer:
        df.to_excel(writer)

----------------------------------------------------------------------------
**Activating a window**

Option 1
```py
import win32com.client as comclt
wsh= comclt.Dispatch("WScript.Shell")
wsh.AppActivate("the_name_of_window")
wsh.SendKeys("{ENTER}")
```
<sup>It doesn't need any extra package to be installed! And most importantly, it can be compiled to EXE with 'py2exe' w/o problem, whereas 'pynput' and 'pyautogui' produce problems. </sup>


Option 2
```py
import win32gui
def myF(hwnd, lParam):
    if win32gui.IsWindowVisible(hwnd):
        if win32gui.GetWindowText(hwnd) == 'FRANKONAS':
            win32gui.SetForegroundWindow(hwnd)
# Passing the handle of each window, to an application-defined callback function (in this case: myF)
win32gui.EnumWindows(myF, None)
```

----------------------------------------------------------------------------
**Read txt, csv file contents**
```py
file = open(r'C:\Temp\test.txt', 'r', encoding='cp1257')

# String of full contents
print(file.read())

# List of all lines
print(file.readlines())

# 1st line, without loading the whole file:
print(file.readline())

# Only one specified line, without loading the whole file
line = 19
[next(file) for x in range(line-1)]
print(next(file))

# Only first n lines, without loading the whole file:
n = 5
head = [next(file) for x in range(n)]
print(*head, sep='')

# Specified lines in the specified order, without loading the whole file:
lines = [1,3,2]
dict = {}
for x in range(max(lines)):
    dict[x+1] = next(file) if x+1 in lines else next(file)
print(*[dict.get(i) for i in lines], sep='')

# Specified lines ascending (not according to the original order), without loading the whole file:
lines = [1,3,2]
for x in range(max(lines)):
    print(next(file)) if x+1 in lines else next(file)
```
