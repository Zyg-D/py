----------------------------------------
Funkcijos gali grazinti daugiau negu viena rezultata.
```python
def convert_secs(secs):
    hours = secs // 3600
    mins = (secs - hours * 3600) // 60
    remaining_secs = secs - hours * 3600 - mins * 60
    return hours, mins, remaining_secs
# calling:
h, m, s = convert_secs(5000)
print(h, m, s)
```
----------------------------------------------
Jeigu `if` blokuose yra `return`'ai, tada `else` galima visiskai pakeisti `return`'u. 
```py
def comp(x):
    if x == 2:
        return print("Equals")
    else:
        return print("Not equals")
```
yra tas pats kas
```py
def comp(x):
    if x == 2:
        return print("Equals")
    return print("Not equals")
```
------------------------------------------------
