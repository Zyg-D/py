-------------------------------------------------------------------------------
**dataframes (pandas)**

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

    dfs = pd.read_excel('C:\\Temp\\file.xlsx', sheet_name=None)

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

Loop through cols: 

    for col in df.columns:

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
        if win32gui.GetWindowText(hwnd) == "FRANKONAS":
            win32gui.SetForegroundWindow(hwnd)
# Passing the handle of each window, to an application-defined callback function (in this case: myF)
win32gui.EnumWindows(myF, None)
```

----------------------------------------------------------------------------
**Read file contents**
```py
a = open("C:\\Temp\\test.txt", "r")
a.read()
```

