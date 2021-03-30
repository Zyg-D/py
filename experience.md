-------------------------------------------------------------------------------
**DF spark**

Create example DF:
```python

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Several cols:
deptCols = [ 'dept_name', 'dept_id']
dept =     [('Marketing', 10),
            (  'Finance', 20),
            (    'Sales', 30),
            (       'IT', 40)]
df = spark.createDataFrame(data=dept, schema=deptCols)
# OR
df = spark.createDataFrame([(1, 11), (2, 22)], ['c1', 'c2'])

# One col
df = spark.createDataFrame([(1,),(2,)], ['c'])
df = spark.createDataFrame([1,2], 'int').toDF('c')
df = spark.range(1, 4).toDF('c')
```

(Foundry)
```python
def my_compute_function(ctx, ...):
    ...
    answer = 2
    df = ctx.spark_session.createDataFrame([(answer,)], ['c1'])
```

Create example RDD:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

sc = spark.sparkContext
dept = [('Marketing', 10),
        (  'Finance', 20),
        (    'Sales', 30)]
rdd = sc.parallelize(dept)
# print(rdd.collect())
```

RDD to DF:

```python
# Auto-generated col names
df = rdd.toDF()
# Providing col names
df = rdd.toDF(['dept_name', 'dept_id'])
```

DF to RDD:

    rdd = df.rdd

RDD, DF from local txt:

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
rdd = spark.sparkContext.textFile(r'C:\Temp\sample.txt')
header_str = rdd.first()
rddDataLines = rdd.filter(lambda line: line != header_str)
rddSplit = rddDataLines.map(lambda k: k.split(','))
df = rddSplit.toDF(header_str.split(','))
```
Assert correct amount of separators in RDD lines (just after importing .txt or .csv):
```python
sep = ','
countRDD = rdd.map(lambda line: len(line.split(sep)))
print(sorted(countRDD.collect())[:3])
```

RDD, DF from online json (more in drive)

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from urllib.request import urlopen

spark = SparkSession.builder.getOrCreate()

url = 'https://randomuser.me/api/0.8/?results=10'
jsonData = urlopen(url).read().decode('utf-8')
rdd = spark.sparkContext.parallelize([jsonData])
df = spark.read.json(rdd)
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

Modify/ rename all columns in DF: 
```python
df = df.toDF(*[f'v_{c}' for c in tp_v.columns])
# or
new_column_name_list = [map(lambda x: x.replace(' ', '_'), df.columns)]
df = df.toDF(*new_column_name_list)
```

Get DF column datatype:

    dict(df.dtypes)['colName']

Get DF col name:

```python
# These can return different results
F.col('c1')._jc.toString()
F.col('c1').__repr__()
F.col('c1').__str__()
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

Keeping only top 5 rows:
```python
df.limit(5)
```

Keep only top rows:
```python
df_filt = df.withColumn('rn', F.row_number().over(
    W.partitionBy(F.col('asm_id'))
    .orderBy(
        F.col('data').desc_nulls_last(),
    )
)).filter('rn = 1').drop('rn')
```

Order by

```python
df.orderBy(F.col('e_snf.snf_san_data').desc_nulls_last())
                                      .asc_nulls_first()
```

Group + aggregate

    df.groupby(F.col('e_cbi.cbi_id')).agg(F.max('e_cbi.ist_data').alias('data'))

Change col type

    new_df = df.withColumn("colx", df["colx"].cast('date'))

Window

```python
from pyspark.sql import Window as W
df03 = df02.withColumn('rn', F.row_number().over(
    W.partitionBy(F.col('e_snf.bucket')).orderBy(F.col('e_snf.date').desc_nulls_last())
))
```

Coalesce

```python
df = df.selectExpr('coalesce(m.ari_asm_id, e_snf.snf_ji_asm_nr) as m_id')
df = df.select(F.coalesce('m.ari_asm_id', 'e_snf.snf_ji_asm_nr').alias('m_id'))
df = df.select(F.coalesce(F.col('m.ari_asm_id'), F.col('e_snf.snf_ji_asm_nr')).alias('m_id'))
```

Select specified cols from DF

    DF2 = DF1.select(['col1','col2'])

Show DF

    DF.show()
    DF.show(truncate=False)

Join DFs (more in drive)

```py
DF_joined = DF1.join(DF2, DF1.id == DF2.id, "inner")
# Possible complex criteria: 
DF_joined = empDF.join(deptDF,[(empDF.emp_id < deptDF.dept_id/10)|(empDF.salary==deptDF.dept_id/-10)],"inner")
```

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
df = df.withColumn('col2', F.expr("regexp_extract_all(col, '(\\\\w+)', 1)"))
df.show()
#    |    col|      col2|
#    +-------+----------+
#    |one two|[one, two]|
#    |   I am|   [I, am]|
#    |    coo|     [coo]|
```

Regex replace

```python
df = df.withColumn('new_col', F.regexp_replace('c1', '\d+', 'replaced_with'))
```

Other

```py
F.current_date()
F.current_timestamp()
F.year(F.current_date())

def days(num):
  return F.expr(f'interval {num} days')
F.col('last_positive_date') + days(37)

df_group.where(F.col('test_type_id').isin({'1477', '1537', '1557'}))

df.withColumn('gmp_indication', F.when(F.col('gmp_promo_code').isNull(), F.lit('profilaktika'))
			     .when(F.col('gmp_promo_code').contains('SIMPT'), F.lit('simptomai')) )

from pyspark.sql import functions as F, Window
w = Window.partitionBy('col1', 'col2').orderBy(  # false comes before true when ordering in sql
  F.col('gmp_patient_municipality_name').isNull(),
  F.col('sender_completion_status') != 'Užbaigta vesti',
  F.desc('sender_form_id'),
)
df = df.withColumn('sender_form_rank', F.row_number().over(w))

```

Foundry - many transforms with one script

```python
from transforms.api import transform_df, Input, Output
from .osp_json_parser import parse_json # Custom function in another file

def generate_transforms(names):
    transforms = []
    for name in names:
        @transform_df(
            Output(f"/path/clean/{name}"),
            raw_json=Input(f"/path/raw/{name}"),
        )
        def my_compute_function(ctx, raw_json):
            return parse_json(raw_json, ctx.spark_session)
        transforms.append(my_compute_function)
    return transforms

TRANSFORMS = generate_transforms([
    'file_name1',
    'file_name2',
    'file_name3',
])
```

-------------------------------------------------------------------------------
**df (pandas)**

Create an empty df:

    df = pd.DataFrame()

Create a sample df (1): 

    df = pd.DataFrame.from_records(
			    [{'col_1': 'a', 'col_2': 1},
			     {'col_1': 'b', 'col_2': 2} ] )

Create a sample df (2): 

    df = pd.DataFrame({'num_legs': [2, 4, 8, 0],
                       'num_wings': [2, 0, 0, 0],
                       'num_specimen_seen': [10, 2, 1, 8] },
                      index=['falcon', 'dog', 'spider', 'fish'] )

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

Finding the index given item name:

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
wsh.AppActivate("FRANKONAS")
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
# List of lines
print(file.readlines())
# 1st line
print(file.readline())
```

**Read specified lines in txt, csv**
```python
file = open(r'C:\Temp\test.txt', 'r', encoding='cp1257')
lines = file.readlines()
for i in list(range(2)):
    print(lines[i])
```
