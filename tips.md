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
