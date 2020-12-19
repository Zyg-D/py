----------------------------------------
Funkcijos gali grazinti daugiau negu viena rezultata.
```python
def convert_secs(secs):
    hours = secs // 3600
    mins = (secs - hours * 3600) // 60
    remaining_secs = secs - hours * 3600 - mins * 60
    return hours, mins, remaining_secs
# calling:
result = convert_secs(5000)   # writes values to one varb as a tuple
# OR
h, m, s = convert_secs(5000)  # implicit tuple unpacking
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

**Asterisks**

Iterable unpacking:
```py
numbers = [2, 1, 3, 4, 7]
print(*numbers, sep=',')
```
```py
print([*'asterisk'])
```
Packing into list/ dict (funkcijos viduje bus dazniausiai loopinama per elementus):
```py
from random import randint
def roll(*dice):
    return sum(randint(1, die) for die in dice)
roll(6, 6) # rollinami du 6-sieniai kauliukai 
```
```py
def tag(tag_name, **attributes):
    attribute_list = [
        f'{name}="{value}"'
        for name, value in attributes.items()
    ]
    return f"<{tag_name} {' '.join(attribute_list)}>"
tag('img', height=20, width=40, src="face.jpg") # '<img height="20" width="40" src="face.jpg">'
```

Funkcijoje `def get_multiple(*keys, dictionary, default=None)` argumentai `dictionary` ir `default` gali buti irasyti tik kaip kayword arguments (named arguments), nes pries tai einantis nelimituotas skaicius positional argumentu yra supackinamas i `keys` lista. 

Funkcijos `def with_previous(iterable, *, fillvalue=None)` sintakse rodo, kad yra su argumentai: `iterable` yra pozicinis, o `fillvalue` - keyword-only. Supackinimas buvo atliktas pries passinant `iterable` i funkcija. Jeigu jis nebutu atliktas, galima naudoti pries tai aprasyta sintakse. 
```py
def with_previous(iterable, *, fillvalue=None):
    previous = fillvalue
    for item in iterable:
        yield previous, item
        previous = item
```

For some reason this is called *tuple* unpacking: 
```py
fruits = ['lemon', 'pear', 'watermelon', 'tomato']
first, second, *remaining = fruits
remaining # ['watermelon', 'tomato']
```

Merging dicts (each subsequent dict overrides values in keys which duplicate): 
```py
{**dict1, **dict2}
```
---------------------------------------------------------------------------------------------------

