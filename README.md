
**Jupyter notebooks**  
Example:  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rmartin977/mnist_classification/master?filepath=mnist_classification.ipynb)  
My own example file:  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Zyg-D/py/master?filepath=jupyter%2F201210.ipynb)

--------------------------------------------------------------------------
**Type hints**
```py
def double(number: float) -> float:
    return 2 * number
```
`number` should be a float and `double()` should return a float. These annotations are treated as type hints - they are not enforced. Type hints allow static type checkers to do type checking of your Python code, without actually running your scripts.  

Static type checkers:  
- Pyright 
- Pytype 
- Pyre
- Mypy

Using `def draw_line(direction: Literal["horizontal", "vertical"]) -> None` instead of `def draw_line(direction: str) -> None` will inform type checkers that other literals are not allowed. (`from typing import Literal` is required.)

--------------------------------------------------------------------------
**Tuple *unpacking***
```py
t = (1,23,20)
h, m, s = t
```
also this (implicit):
```py
lst = ["q","w","e"]
for i, val in enumerate(lst):
  print(f"{i} - {val}")
```

-------------------------------------------------------------------------
**List comprehension**: 
```py
print( [ x * 7 for x in range(1,11)] )
```

----------------------------------------------------------------------------------
**2018-09 py frameworks** (tinklalapio kūrimui berods čia ieškojau)
- Django - full-stack, most popular
- webapp2 - full-stack, less popular, compatible with Google App Engine’s webapp
- Flask - non full-stack, popular, micro, but good scalability, extensions/ plugins, used by LinkedIn, Pinterest
- Web2py - full-stack, popular, pasirinktas to, kuris norėjo daugiau negu php
- Pyramid - non full-stack, popular, powerful, harder to learn, flexible
- Pylons  - full-stack, popular, powerful, harder to learn

----------------------------------------------------------------------------------
**IDE**
- Atom - feature-rich, fits professionals, better than sublime, developed by GitHub
- PyCharm - professionals' choice
