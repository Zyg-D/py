<h3><!---------------------------------------------------------------------------->
    dataframes (pandas) 
</h3><!--------------------------------------------------------------------------->

Referencing individual items - by indexes:

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

<h3><!---------------------------------------------------------------------------->
    lists
</h3><!--------------------------------------------------------------------------->

Referencing elements by index:

    list[0]

Finding the index given item name:

    ["foo", "bar", "baz"].index("bar")

<h3><!---------------------------------------------------------------------------->
    dicts
</h3><!---------------------------------------------------------------------------> 

Referencing the 1st element (but no order is guarranteed in dicts):

    list(my_dict.keys())[0]
    
Referencing the value of a specified key:  
<sup>default is returned shen the key is not found</sup>

    my_dict.get('key_name'[, default])

<h3><!---------------------------------------------------------------------------->
    importing .json
</h3><!--------------------------------------------------------------------------->

    df=pandas.read_json("file_name.json", encoding = 'utf8')
    
<h3><!---------------------------------------------------------------------------->
    exporting .xlsx
</h3><!--------------------------------------------------------------------------->

    with pandas.ExcelWriter('out_file.xlsx') as writer:
        df.to_excel(writer)

