-------------------------------------------------------------------------------
**dataframes (pandas)**

Create an empty df:

    df = pd.DataFrame()

Create a sample df: 

    df = pd.DataFrame.from_records(
			    [{'col_1': 'a', 'col_2': 1},
			     {'col_1': 'b', 'col_2': 2} ] )

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

    mydict[key] = value

Referencing the 1st element (but no order is guaranteed in dicts):

    list(my_dict.keys())[0]
    
Referencing the value of a specified key:  
<sup>default is returned shen the key is not found</sup>

    my_dict.get('key_name'[, default])

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

