+++
title = 'Homework 2'
date = 2024-11-21T11:57:07-05:00
draft = false
summary = "My homework backup for Advance Algorithm subject."
series = ["Advance Algorithm",]
tags = ["Advance Algorithm", "homework", "university", "school"]
author= ["Me"]
[params]
  math = true
+++

## Answer 1

a. $$\Theta(n^5)$$

b. $$\Theta(n \lg n)$$

c. $$\Theta(n \cdot 2^n)$$

d. $$\Theta(n^3)$$

---

## Answer 2

1. This algorithm finds sums of square values of first n positive integers. That is **1<sup>2</sup> + 2<sup>2</sup> + 3<sup>2</sup> + 4<sup>2</sup>..... + n<sup>2</sup>**.
2. Basic operation is multiplication for finding square i.e. `i*i`
3. It is executed n times. Once per loop and loop runs from 1 to n (inclusive).
4. Efficiency of this algo is O(n).
5. Instead of running a loop to find this sum. We can use below mathematical formula for finding sum of square values of first n numbers.

    $$
    S = \frac{n(n+1)(2n+1)}{6} 
    $$

    The efficiency of this version of same algo is O(1).

---

## Answer 3

1. In current algo, we are executing basic operation in 3 nested loop. And outermost loop executes on order of n _(from 0 -> n-2 ~ (n -1) times)_, where n is size of matrix (Assuming a square matrix). There, efficiency is O($n^3$).
2. A simple improvement that can be made is that in operation `A[j, k]← A[j, k] − A[i, k] ∗ A[j, i] / A[i, i]` which inside 3rd loop _(k <- i to n>)_, the `A[j, i] / A[i, i]` does not depends on k, therefore it can be calculated once in 2nd loop and need not be calculated on every iteration in 3rd loop.
   
    Optimized version is
    ```
    for i ← 0 to n - 2 do
    for j ← i + 1 to n - 1 do
        factor ← A[j, i] / A[i, i]
        for k ← i to n do
            A[j, k] ← A[j, k] - factor * A[i, k]

    ```

    In this version even though time complexity is Still $O(n^3)$, division operations have been reduced from $O(n^3) to O(n^2)$. Which will be huge improvement in cases where size of input matrix is massive.

---

## Answer 4


a. 
$$
 x(n) = 5n - 5 
$$

b. 
$$
 x(n) = 4 \times 3^{n-1} 
$$

c. 
$$
 x(n) = \frac{n(n+1)}{2} 
$$

d. 
$$
 x(n) = 2n - 1 
$$

e. 
$$
 x(n) = \log_3(n) + 1 
$$


---

## Answer 5

1. Base Case `T(1)=1` which n = 1
    
    For n > 1, a recursion call is made with T(n - 1) time operations, Compute 3 multiplications for cube and adding result of recursion which is one additional operation. Therefore, `T(n)=T(n−1)+3+1=T(n−1)+4`

        i. T(n) = T(n-1) + 4
        ii. T(n-1) = T(n−2)+4

        Replacing step ii in i,

        iii. T(n)=(T(n−2)+4)+4=T(n−2)+8
        iv. T(n−2)=T(n−3)+4

        replacing step iv in iii,
        T(n)=(T(n−3)+4)+8=T(n−3)+12
        .
        . Iterating till k
        .

        T(n)=T(n−k)+4k

        When k=n−1,
        T(n)=T(1)+4(n−1).

        We know that T(1)=1,
        T(n)=1+4(n−1) = 4n−3

    So, resulting recursive relation is `4n−3`

2. In non-recursive approach, we will find sum of n cubes by iterating from from 1 -> n and adding the cubes.
   
    Time complexity in both approaches is same, O(n)
    Space complexity in non-recursive approach is O(1) where as in recursive approach, we use a call stack of size O(n).

    So, non-recursive approach is optimized in case of space complexity.

---

## Answer 6

1. This algorithm is finding smallest element in first n-1 elements.
2. For Recurrence Relation
    
    Base case is `T(1)=1` for n = 1
    For n > 0, it makes recursive call for `T(n - 1)` + 2 operations comparison and return. Therefore,

        i, T(n) = T(n-1) + 2
        ii, Also, T(n-1) = T(n-2) + 2

        Replacing values of step ii into i,
        iii T(n) = (T(n−2)+2)+2 = T(n−2)+4

        iv, T(n−2)=T(n−3)+2
        Replacing values of step iv into iii,
        T(n) = (T(n−3)+2)+4 = T(n−3)+6

        .
        . Iterating for k steps
        .

        T(n)=T(n−k)+2k

        When, k=n−1
        T(n) = T(1)+2(n−1), we know that T(1) = 1

        So, T(n)=1+2(n−1) = 1+2n−2 = 2n−1

    The final solution for T(n) is,

        T(n)=2n−1


---

## Answer 7

Below is the python code for worst case bruteforce approach for said algorithm returning index of pattern.

``` python
    def match_pattern_brut(text, pattern):
        len_t = len(text)
        len_p = len(pattern)

        for i in range(len_t - len_p + 1):
            is_matched = True

            for j in range(len_p):
                if text[i + j] != pattern[j]:
                    is_matched = False
                    break

            if matched:
                return i

        return -1

```

In the above algorithm, it can be observed that outer loop in worst case will executing `len_t - len_p + 1` times and inner loop is executing `len_p` times. So, resulting operations in worst case are,

```
    len_p * (len_t - len_p +1)
```

The, overall time complexity is O(len_t * len_p).

---

## Answer 8

Below is Python for exhaustive search algo for finding a clique of size `k` in a graph

``` python
from itertools import combinations

def check_complete_subgraph(graph, vertex_set):
    for x in range(len(vertex_set)):
        for y in range(x + 1, len(vertex_set)):
            if vertex_set[y] not in graph[vertex_set[x]]:
                return False
    return True


# graph is dict containing mapping of a node and list of vertices
# Ex: {1: [2,3], 2:[4]....}
def search_clique(graph, size):
    nodes = list(graph.keys())
    for subset in combinations(nodes, size):
        if check_complete_subgraph(graph, subset):
            return True
    return False


```

---

## Answer 9

**Adjacency graph**

```
    . | a | b | c | d | e | f | g
    -----------------------------
    a | 0 | 1 | 1 | 1 | 1 | 0 | 0
    -----------------------------
    b | 1 | 0 | 0 | 1 | 0 | 1 | 0
    -----------------------------
    c | 1 | 0 | 0 | 0 | 0 | 0 | 1
    -----------------------------
    d | 1 | 1 | 0 | 0 | 0 | 1 | 0
    -----------------------------
    e | 1 | 0 | 0 | 0 | 0 | 0 | 1
    -----------------------------
    f | 0 | 1 | 0 | 1 | 0 | 0 | 0
    -----------------------------
    g | 0 | 0 | 1 | 0 | 1 | 0 | 0

```

**Adjacency List**

```
    a: [b, c, d, e]
    b: [a, d, f]
    c: [a, g]
    d: [a, b, f]
    e: [a, g]
    f: [b, d]
    g: [c, e]
```

**DFS traversal**

- Order of push stack (traversal)

    `a, b, d, f, c, g, e`

- Order of pop stack (dead end)
    
    `f, d, b, e, g, c, a`


---

## Answer 10

**BFS Order of previous graph**

`a, b, c, d, e, f, g`

```
Root      : a
1st Level : b, c, d, e
2nd Level : f, g
```