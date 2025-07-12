## Table of Contents
1. [What is Python, and its key features?](#1-what-is-python-and-its-key-features)
2. [What are Python's built-in data types?](#2-what-are-pythons-built-in-data-types)
3. [How do you declare and use variables?](#3-how-do-you-declare-and-use-variables)
4. [Explain the difference between a list, tuple, and set.](#4-explain-the-difference-between-a-list-tuple-and-set)
5. [How do you iterate over a list in Python?](#5-how-do-you-iterate-over-a-list-in-python)
6. [What are Python's conditional statements, and how are they used?](#6-what-are-pythons-conditional-statements-and-how-are-they-used)
7. [How does Python handle memory management?](#7-how-does-python-handle-memory-management)
8. [Explain the use of the `len()` function.](#8-explain-the-use-of-the-len-function)
9. [What is the difference between `is` and `==` in Python?](#9-what-is-the-difference-between-is-and-in-python)
10. [How do you handle exceptions in Python?](#10-how-do-you-handle-exceptions-in-python)
11. [What are Python functions, and how do you define them?](#11-what-are-python-functions-and-how-do-you-define-them)
12. [What is the difference between `*args` and `**kwargs`?](#12-what-is-the-difference-between-args-and-kwargs)
13. [How is Python’s `for` loop different from other programming languages?](#13-how-is-pythons-for-loop-different-from-other-programming-languages)
14. [Explain the purpose of the `range()` function.](#14-explain-the-purpose-of-the-range-function)
15. [How do you import and use modules?](#15-how-do-you-import-and-use-modules)
16. [What are Python decorators, and how do they work?](#16-what-are-python-decorators-and-how-do-they-work)
17. [How do you reverse a string in Python?](#17-how-do-you-reverse-a-string-in-python)
18. [How do you check if an element exists in a list?](#18-how-do-you-check-if-an-element-exists-in-a-list)
19. [What is a lambda function? Provide an example.](#19-what-is-a-lambda-function-provide-an-example)
20. [Explain the difference between shallow copy and deep copy in Python.](#20-explain-the-difference-between-shallow-copy-and-deep-copy-in-python)
21. [What are Python comprehensions, and how are they used?](#21-what-are-python-comprehensions-and-how-are-they-used)
22. [How does Python’s garbage collection work?](#22-how-does-pythons-garbage-collection-work)
23. [Explain Python’s Global Interpreter Lock (GIL).](#23-explain-pythons-global-interpreter-lock-gil)
24. [What is the difference between mutable and immutable objects?](#24-what-is-the-difference-between-mutable-and-immutable-objects)
25. [How do you use the `zip()` function?](#25-how-do-you-use-the-zip-function)
26. [Explain the difference between `@staticmethod` and `@classmethod`.](#26-explain-the-difference-between-staticmethod-and-classmethod)
27. [How do you merge two dictionaries?](#27-how-do-you-merge-two-dictionaries)
28. [What is the difference between `sort()` and `sorted()`?](#28-what-is-the-difference-between-sort-and-sorted)
29. [How do you handle file operations?](#29-how-do-you-handle-file-operations)
30. [What are Python’s iterators and generators?](#30-what-are-pythons-iterators-and-generators)
31. [How do you use the `with` statement?](#31-how-do-you-use-the-with-statement)
32. [What is Python’s `itertools` module, and when would you use it?](#32-what-is-pythons-itertools-module-and-when-would-you-use-it)
33. [Explain the difference between positional and keyword arguments.](#33-explain-the-difference-between-positional-and-keyword-arguments)
34. [How do you perform matrix operations in Python?](#34-how-do-you-perform-matrix-operations-in-python)
35. [What are Python’s metaclasses, and how are they used?](#35-what-are-pythons-metaclasses-and-how-are-they-used)
36. [How do you perform unit testing?](#36-how-do-you-perform-unit-testing)
37. [Explain how Python’s `os` module is used.](#37-explain-how-pythons-os-module-is-used)
38. [What are `argsort()` and `argmax()` functions?](#38-what-are-argsort-and-argmax-functions)
39. [How do you optimize code performance?](#39-how-do-you-optimize-code-performance)
40. [How does Python’s multiprocessing differ from threading?](#40-how-does-pythons-multiprocessing-differ-from-threading)
41. [Explain how to implement a custom Python metaclass.](#41-explain-how-to-implement-a-custom-python-metaclass)
42. [How do you implement memoization?](#42-how-do-you-implement-memoization)
43. [What is Python’s asyncio, and how does it handle concurrency?](#43-what-is-pythons-asyncio-and-how-does-it-handle-concurrency)
44. [How do you profile Python code to identify performance bottlenecks?](#44-how-do-you-profile-python-code-to-identify-performance-bottlenecks)
45. [How do you handle circular imports in Python projects?](#45-how-do-you-handle-circular-imports-in-python-projects)
46. [What are Python’s weak references, and when would you use them?](#46-what-are-pythons-weak-references-and-when-would-you-use-them)
47. [How does Python’s dataclasses module work, and when should you use it?](#47-how-does-pythons-dataclasses-module-work-and-when-should-you-use-it)
48. [What are Python’s context managers, and how do you create a custom one?](#48-what-are-pythons-context-managers-and-how-do-you-create-a-custom-one)

# Basic: Laying the Foundation
### 1. What is Python, and its key features?
Python is a high-level, interpreted, general-purpose programming language known for its readability and simplicity. Created by Guido van Rossum and first released in 1991, it supports multiple programming paradigms, including procedural, object-oriented, and functional programming.
**Key Features:**
- **Readability:** Python’s syntax is clear and concise, using indentation for block structures.
- **Interpreted:** Runs code line-by-line, making debugging easier.
- **Dynamically Typed:** No need to declare variable types explicitly.
- **Extensive Standard Library:** Provides modules for tasks like file I/O, networking, and data processing.
- **Cross-Platform:** Runs on Windows, macOS, Linux, etc.
- **Community Support:** Large ecosystem with frameworks like Django, Flask, and libraries like NumPy, Pandas.
- **Versatility:** Used in web development, data science, automation, AI, and more.
### 2. What are Python's built-in data types?
Python’s built-in data types include:
- **Numeric:** `int` (e.g., 5), `float` (e.g., 3.14), `complex` (e.g., 3+4j).
- **Sequence:** `str` (e.g., "hello"), `list` (e.g., [1, 2, 3]), `tuple` (e.g., (1, 2, 3)).
- **Mapping:** `dict` (e.g., {"key": "value"}).
- **Set:** `set` (e.g., {1, 2, 3}), `frozenset` (immutable set).
- **Boolean:** `bool` (`True`, `False`).
- **NoneType:** `None` (represents absence of a value).
### 3. How do you declare and use variables?
In Python, variables are declared by assigning a value using the `=` operator. No explicit type declaration is needed due to dynamic typing.
**Example:**
```python
x = 10          # Integer
name = "Alice"  # String
is_active = True  # Boolean
print(x, name, is_active)  # Output: 10 Alice True
# Variables can be reassigned to different types:
x = "Now a string"  # Reassign x to a string
```
### 4. Explain the difference between a list, tuple, and set.
- **List:** Ordered, mutable collection. Allow duplicates.
- **Example:** `[1, 2, 2, 3]`
- **Use:** When you need a modifiable sequence.
- **Tuple:** Ordered, immutable collection. Allow duplicates.
- **Example:** `(1, 2, 2, 3)`
- **Use:** When data should not change (e.g., fixed records).
- **Set:** Unordered, mutable collection. No duplicates.
- **Example:** `{1, 2, 3}`
- **Use:** For unique elements and set operations (union, intersection).
### 5. How do you iterate over a list in Python?
Use a `for` loop or `while` loop to iterate over a list.
**Example (for loop):**
```python
my_list = [1, 2, 3]
for item in my_list:
print(item)
```
### 6. What are Python's conditional statements, and how are they used?
Python’s conditional statements are `if`, `elif`, and `else`.
**Example:**
```python
x = 10
if x > 5:
print("x is greater than 5")
elif x == 5:
print("x is 5")
else:
print("x is less than 5")
```
### 7. How does Python handle memory management?
Python uses:
- **Reference Counting:** Tracks the number of references to an object. When the count reaches zero, the object is deallocated.
- **Garbage Collection:** Handles cyclic references (objects referencing each other) using a generational garbage collector.
- **Memory Allocator:** Python’s `pymalloc` manages memory for small objects efficiently.
### 8. Explain the use of the `len()` function.
The `len()` function returns the number of items in an object (e.g., list, string, tuple, dictionary).
**Example:**
```python
my_list = [1, 2, 3]
print(len(my_list))  # Output: 3
```
### 9. What is the difference between `is` and `==` in Python?
- `==`: Checks for value equality (compares content).
- `is`: Checks for identity (same memory location).
**Example:**
```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # Output: True
print(a is b)  # Output: False
```
### 10. How do you handle exceptions in Python?
Use `try`, `except`, `else`, and `finally` blocks.
**Example:**
```python
try:
result = 10 / 0
except ZeroDivisionError:
print("Cannot divide by zero!")
else:
print("Division successful")
finally:
print("This always runs")
# Output: Cannot divide by zero! This always runs
```
### 11. What are Python functions, and how do you define them?
Functions are reusable blocks of code that perform a specific task, defined using the `def` keyword.
**Example:**
```python
def greet(name):
return f"Hello, {name}!"
print(greet("Alice"))  # Output: Hello, Alice!
```
### 12. What is the difference between `*args` and `**kwargs`?
- `*args`: Allows a function to accept any number of positional arguments as a tuple.
- `**kwargs`: Allows a function to accept any number of keyword arguments as a dictionary.
**Example:**
```python
def example(*args, **kwargs):
print("Positional args:", args)
print("Keyword args:", kwargs)
example(1, 2, 3, name="Alice", age=30)
# Output: Positional args: (1, 2, 3)
#         Keyword args: {'name': 'Alice', 'age': 30}
```
### 13. How is Python’s `for` loop different from other programming languages?
Python’s `for` loop is designed for iteration over sequences (lists, strings, etc.) and is more concise than traditional C-style loops. It uses an iterator protocol, making it flexible for custom objects.
**Example:**
```python
for i in range(5):  # Iterates over 0, 1, 2, 3, 4
print(i)
```
In contrast, languages like C use index-based loops with explicit counters.
### 14. Explain the purpose of the `range()` function.
The `range()` function generates a sequence of numbers, often used in loops.
**Syntax:** `range(start, stop, step)`
- **start:** Starting number (inclusive, default 0).
- **stop:** End number (exclusive).
- **step:** Increment (default 1).
**Example:**
```python
for i in range(1, 6, 2):  # Outputs: 1, 3, 5
print(i)
```
### 15. How do you import and use modules?
Modules are imported using the `import` statement.
**Example:**
```python
import math
print(math.sqrt(16))  # Output: 4.0
```
You can use aliases (`import math as m`) or import specific functions (`from math import sqrt`).
### 16. What are Python decorators, and how do they work?
Decorators are functions that modify the behavior of another function or method. They are applied using the `@` symbol.
**Example:**
```python
def my_decorator(func):
def wrapper():
print("Before function")
func()
print("After function")
return wrapper
@my_decorator
def say_hello():
print("Hello!")
say_hello()
# Output: Before function
#         Hello!
#         After function
```
### 17. How do you reverse a string in Python?
Use slicing with a step of -1.
**Example:**
```python
text = "hello"
reversed_text = text[::-1]
print(reversed_text)  # Output: olleh
```
### 18. How do you check if an element exists in a list?
Use the `in` operator.
**Example:**
```python
my_list = [1, 2, 3]
print(2 in my_list)  # Output: True
print(4 in my_list)  # Output: False
```
### 19. What is a lambda function? Provide an example.
A lambda function is an anonymous, single-line function defined using the `lambda` keyword.
**Example:**
```python
square = lambda x: x * x
print(square(5))  # Output: 25
```
## Intermediate: Keep Practicing
### 20. Explain the difference between shallow copy and deep copy in Python.
- **Shallow Copy:** Copies the outer object but references nested objects. Use `copy.copy()`.
- **Deep Copy:** Recursively copies all objects, including nested ones. Use `copy.deepcopy()`.
**Example:**
```python
import copy
lst = [[1, 2], 3]
shallow = copy.copy(lst)
deep = copy.deepcopy(lst)
lst[0][0] = 99
print(shallow)  # Output: [[99, 2], 3]
print(deep)     # Output: [[1, 2], 3]
```
### 21. What are Python comprehensions, and how are they used?
Comprehensions provide a concise way to create lists, dictionaries, or sets.
**Examples:**
- **List:** `[x**2 for x in range(5)]` → `[0, 1, 4, 9, 16]`
- **Dictionary:** `{x: x**2 for x in range(5)}` → `{0: 0, 1: 1, 2: 4, 3: 9, 4: 16}`
- **Set:** `{x for x in [1, 1, 2, 3]}` → `{1, 2, 3}`
### 22. How does Python’s garbage collection work?
Python’s garbage collector:
- Uses reference counting to deallocate objects with zero references.
- Employs a generational garbage collector to detect and clean up cyclic references.
- Divides objects into generations (0, 1, 2) to optimize collection frequency.
### 23. Explain Python’s Global Interpreter Lock (GIL).
The GIL is a mutex that allows only one native thread to execute Python bytecode at a time, even on multi-core systems. It simplifies memory management but limits true parallelism in CPU-bound tasks. For I/O-bound tasks, threading is still effective.
### 24. What is the difference between mutable and immutable objects?
- **Mutable:** Can be modified after creation (e.g., list, dict, set).
- **Immutable:** Cannot be changed (e.g., int, str, tuple).
**Example:**
```python
lst = [1, 2]
lst.append(3)  # Mutable: [1, 2, 3]
text = "hello"
text += " world"  # Creates new string, original unchanged
```
### 25. How do you use the `zip()` function?
The `zip()` function combines multiple iterables into tuples.
**Example:**
```python
names = ["Alice", "Bob"]
ages = [25, 30]
for name, age in zip(names, ages):
print(f"{name} is {age}")
# Output: Alice is 25
#         Bob is 30
```
### 26. Explain the difference between `@staticmethod` and `@classmethod`.
- `@staticmethod:` A method that doesn’t access instance (self) or class (cls) data. Behaves like a regular function in a class.
- `@classmethod:` A method that takes the class (cls) as its first argument. Useful for alternative constructors.
**Example:**
```python
class MyClass:
@staticmethod
def static_func():
return "Static"
@classmethod
def class_func(cls):
return f"Class: {cls.__name__}"
print(MyClass.static_func())  # Output: Static
print(MyClass.class_func())   # Output: Class: MyClass
```
### 27. How do you merge two dictionaries?
- Using `|=` (Python 3.9+): `dict1 |= dict2`
- Using `update()`: `dict1.update(dict2)`
- Using merge operator `|`: `new_dict = dict1 | dict2`
**Example:**
```python
dict1 = {"a": 1}
dict2 = {"b": 2}
merged = dict1 | dict2
print(merged)  # Output: {'a': 1, 'b': 2}
```
### 28. What is the difference between `sort()` and `sorted()`?
- `sort()`: Modifies a list in place.
- `sorted()`: Returns a new sorted list, leaving the original unchanged.
**Example:**
```python
lst = [3, 1, 2]
lst.sort()  # Modifies lst: [1, 2, 3]
new_lst = sorted([3, 1, 2])  # Returns [1, 2, 3], original unchanged
```
### 29. How do you handle file operations?
Use `open()` for file operations, preferably with a `with` statement.
**Example:**
```python
with open("file.txt", "w") as f:
f.write("Hello, World!")
with open("file.txt", "r") as f:
content = f.read()
print(content)  # Output: Hello, World!
```
### 30. What are Python’s iterators and generators?
- **Iterator:** An object with `__iter__` and `__next__` methods for sequential access.
- **Generator:** A function using `yield` to produce values lazily, saving memory.
**Example (Generator):**
```python
def my_gen():
for i in range(3):
yield i
for val in my_gen():
print(val)  # Output: 0, 1, 2
```
### 31. How do you use the `with` statement?
The `with` statement ensures proper resource management (e.g., closing files) using context managers.
**Example:**
```python
with open("file.txt", "r") as f:
print(f.read())
# File is automatically closed after the block
```
### 32. What is Python’s `itertools` module, and when would you use it?
The `itertools` module provides tools for efficient iteration (e.g., `chain`, `combinations`, `permutations`). Use it for combinatorial tasks or memory-efficient looping.
**Example:**
```python
from itertools import chain
result = list(chain([1, 2], [3, 4]))  # Output: [1, 2, 3, 4]
```
### 33. Explain the difference between positional and keyword arguments.
- **Positional Arguments:** Passed in order, matched by position.
- **Keyword Arguments:** Passed with a name, allowing flexible order.
**Example:**
```python
def func(a, b):
print(a, b)
func(1, 2)         # Positional: 1 2
func(b=2, a=1)     # Keyword: 1 2
```
### 34. How do you perform matrix operations in Python?
Use libraries like NumPy for efficient matrix operations.
**Example:**
```python
import numpy as np
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
result = matrix1 @ matrix2  # Matrix multiplication
print(result)  # Output: [[19 22], [43 50]]
```
### 35. What are Python’s metaclasses, and how are they used?
Metaclasses are classes of classes, controlling class creation. They inherit from `type`.
**Example:**
```python
class Meta(type):
def __new__(cls, name, bases, attrs):
attrs["custom"] = "value"
return super().__new__(cls, name, bases, attrs)
class MyClass(metaclass=Meta):
pass
print(MyClass.custom)  # Output: value
```
### 36. How do you perform unit testing?
Use the `unittest` module to write and run tests.
**Example:**
```python
import unittest
def add(a, b):
return a + b
class TestMath(unittest.TestCase):
def test_add(self):
self.assertEqual(add(2, 3), 5)
if __name__ == "__main__":
unittest.main()
```

### 37. Explain how Python’s `os` module is used.
The `os` module provides functions for interacting with the operating system (e.g., file handling, directories).
**Example:**
```python
import os
print(os.getcwd())  # Current directory
os.mkdir("new_folder")  # Create directory
```
### 38. What are `argsort()` and `argmax()` functions?
These are NumPy functions:
- `argsort()`: Returns indices that would sort an array.
- `argmax()`: Returns the index of the maximum value.
**Example:**
```python
import numpy as np
arr = np.array([3, 1, 2])
print(np.argsort(arr))  # Output: [1 2 0]
print(np.argmax(arr))   # Output: 0
```
### 39. How do you optimize code performance?
- Use efficient data structures (e.g., sets for lookups).
- Leverage built-in functions and libraries (e.g., NumPy).
- Use list comprehensions instead of loops where possible.
- Profile code with tools like `cProfile` to identify bottlenecks.
- Use multiprocessing for CPU-bound tasks.
### Advanced: Taking Your Skills to the Next Level
### 40. How does Python’s multiprocessing differ from threading?
- **Threading**: Uses threads within a single process, limited by the GIL for CPU-bound tasks. Suitable for I/O-bound tasks.
- **Multiprocessing**: Uses separate processes, bypassing the GIL for true parallelism. Ideal for CPU-bound tasks.
**Example (Multiprocessing):**
```python
from multiprocessing import Process
def task():
print("Running")
p = Process(target=task)
p.start()
p.join()
```
### 41. Explain how to implement a custom Python metaclass.
A metaclass defines how a class is created. Inherit from `type` and override `__new__` or `__init__`.
**Example:**
```python
class CustomMeta(type):
def __new__(cls, name, bases, attrs):
attrs["extra"] = "added"
return super().__new__(cls, name, bases, attrs)
class MyClass(metaclass=CustomMeta):
pass
print(MyClass.extra)  # Output: added
```
### 42. How do you implement memoization?
Memoization caches function results to avoid redundant computations. Use a dictionary or `functools.lru_cache`.
**Example:**
```python
from functools import lru_cache
@lru_cache(maxsize=None)
def fibonacci(n):
if n < 2:
return n
return fibonacci(n-1) + fibonacci(n-2)
print(fibonacci(10))  # Output: 55
```
### 43. What is Python’s asyncio, and how does it handle concurrency?
`asyncio` provides a framework for asynchronous programming using coroutines, allowing concurrent execution for I/O-bound tasks.
**Example:**
```python
import asyncio
async def say_hello():
await asyncio.sleep(1)
print("Hello")
asyncio.run(say_hello())
```
### 44. How do you profile Python code to identify performance bottlenecks?
Use `cProfile` or `line_profiler` to analyze execution time.
**Example:**
```python
import cProfile
def slow_function():
sum(range(1000000))
cProfile.run("slow_function()")
```
### 45. How do you handle circular imports in Python projects?
- Restructure code to avoid circular dependencies.
- Move imports inside functions or at the end of the module.
- Use lazy imports with `importlib.import_module`.
**Example:**
```python
# module_a.py
def func_a():
from module_b import func_b
func_b()
```
### 46. What are Python’s weak references, and when would you use them?
Weak references allow referencing an object without increasing its reference count, useful for caching or avoiding memory leaks.
**Example:**
```python
import weakref
obj = [1, 2, 3]
weak_ref = weakref.ref(obj)
print(weak_ref())  # Output: [1, 2, 3]
del obj
print(weak_ref())  # Output: None
```
### 47. How does Python’s dataclasses module work, and when should you use it?
The `dataclasses` module simplifies class creation for data storage by auto-generating methods like `__init__`, `__repr__`.
**Example:**
```python
from dataclasses import dataclass
@dataclass
class Person:
name: str
age: int
p = Person("Alice", 30)
print(p)  # Output: Person(name='Alice', age=30)
```
**Use Case**: When you need classes primarily for storing data with minimal boilerplate.
### 48. What are Python’s context managers, and how do you create a custom one?
Context managers handle resource setup and cleanup, used with the `with` statement. Create one by implementing `__enter__` and `__exit__`.
**Example:**
```python
class MyContext:
def __enter__(self):
print("Enter")
return self
def __exit__(self, exc_type, exc_val, exc_tb):
print("Exit")
with MyContext():
print("Inside")
# Output: Enter
#         Inside
#         Exit
```


