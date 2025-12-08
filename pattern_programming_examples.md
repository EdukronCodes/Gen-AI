# Pattern Programming Examples in Python

## Pattern 1: Right-Angled Triangle with Stars

```python
n = 5
for i in range(1, n + 1):
    print('*' * i)
```

**Output:**
```
*
**
***
****
*****
```

**Explanation:** This pattern prints stars in increasing order. Each row contains one more star than the previous row, creating a right-angled triangle shape.

---

## Pattern 2: Inverted Right-Angled Triangle with Stars

```python
n = 5
for i in range(n, 0, -1):
    print('*' * i)
```

**Output:**
```
*****
****
***
**
*
```

**Explanation:** This pattern prints stars in decreasing order. Each row contains one less star than the previous row, creating an inverted right-angled triangle.

---

## Pattern 3: Pyramid Pattern with Stars

```python
n = 5
for i in range(1, n + 1):
    spaces = ' ' * (n - i)
    stars = '*' * (2 * i - 1)
    print(spaces + stars)
```

**Output:**
```
    *
   ***
  *****
 *******
*********
```

**Explanation:** This pattern creates a pyramid by printing spaces before stars. The number of spaces decreases while the number of stars increases in each row, forming a centered pyramid shape.

---

## Pattern 4: Number Pattern - Right-Angled Triangle

```python
n = 5
for i in range(1, n + 1):
    for j in range(1, i + 1):
        print(j, end=' ')
    print()
```

**Output:**
```
1 
1 2 
1 2 3 
1 2 3 4 
1 2 3 4 5 
```

**Explanation:** This pattern prints numbers in each row starting from 1 up to the row number. Each row displays consecutive numbers from 1 to the current row index.

---

## Pattern 5: Diamond Pattern with Stars

```python
n = 5
# Upper half
for i in range(1, n + 1):
    spaces = ' ' * (n - i)
    stars = '*' * (2 * i - 1)
    print(spaces + stars)
# Lower half
for i in range(n - 1, 0, -1):
    spaces = ' ' * (n - i)
    stars = '*' * (2 * i - 1)
    print(spaces + stars)
```

**Output:**
```
    *
   ***
  *****
 *******
*********
 *******
  *****
   ***
    *
```

**Explanation:** This pattern creates a diamond shape by combining an upper pyramid and an inverted lower pyramid. The pattern is symmetric around the middle row.

---

## Pattern 6: Square Pattern with Numbers

```python
n = 5
for i in range(1, n + 1):
    for j in range(1, n + 1):
        print(i, end=' ')
    print()
```

**Output:**
```
1 1 1 1 1 
2 2 2 2 2 
3 3 3 3 3 
4 4 4 4 4 
5 5 5 5 5 
```

**Explanation:** This pattern prints a square where each row contains the same number repeated. The number corresponds to the row number.

---

## Pattern 7: Alphabet Pattern - Right-Angled Triangle

```python
n = 5
for i in range(1, n + 1):
    for j in range(i):
        print(chr(65 + j), end=' ')
    print()
```

**Output:**
```
A 
A B 
A B C 
A B C D 
A B C D E 
```

**Explanation:** This pattern prints alphabets in each row starting from 'A'. Each row displays consecutive letters from 'A' up to the current row's letter. ASCII value 65 represents 'A'.

---

## Pattern 8: Hollow Square Pattern

```python
n = 5
for i in range(1, n + 1):
    for j in range(1, n + 1):
        if i == 1 or i == n or j == 1 or j == n:
            print('*', end=' ')
        else:
            print(' ', end=' ')
    print()
```

**Output:**
```
* * * * * 
*       * 
*       * 
*       * 
* * * * * 
```

**Explanation:** This pattern creates a hollow square by printing stars only on the border (first row, last row, first column, last column) and spaces in the interior.

---

## Pattern 9: Number Pyramid Pattern

```python
n = 5
for i in range(1, n + 1):
    spaces = ' ' * (n - i)
    numbers = ''
    for j in range(1, i + 1):
        numbers += str(j) + ' '
    print(spaces + numbers)
```

**Output:**
```
    1 
   1 2 
  1 2 3 
 1 2 3 4 
1 2 3 4 5 
```

**Explanation:** This pattern creates a number pyramid where each row contains numbers from 1 to the row number, centered with spaces to form a pyramid structure.

---

## Pattern 10: Reverse Number Pattern

```python
n = 5
for i in range(n, 0, -1):
    for j in range(1, i + 1):
        print(j, end=' ')
    print()
```

**Output:**
```
1 2 3 4 5 
1 2 3 4 
1 2 3 
1 2 
1 
```

**Explanation:** This pattern prints numbers in decreasing rows. Each row starts from 1 and goes up to the current row number, with the number of rows decreasing from top to bottom.

