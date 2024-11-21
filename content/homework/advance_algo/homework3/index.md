+++
title = 'Homework 3'
date = 2024-11-21T11:57:07-05:00
draft = false
summary = "My homework backup for Advance Algorithm subject."
series = ["Advance Algorithm",]
tags = ["Advance Algorithm", "homework", "university", "school"]
author= ["Me"]
+++

## Answer 1


```
Insertion Sort Pseducocode

ins_sort(arr):
    for p1 = 0 to len(arr) - 1:
        key = arr[p1]
        p2 = p1 - 1

        while p2 >= 0 and arr[p2] > key:
            arr[p2 + 1] = arr[p2]
            p2 -= 1
        
    arr[p2 + 1] = key

```

Initial list, E,X,A,M,P,L,E

1. Start with X (2nd element) and compare with E.
   
    - E,X - X > E. No Change

2. Compare A with previous elements
   
    - A,X: A < X, shift X to the right

    - A,E: A < E, shift E to the right

    - Insert A in the first position

3. Compare M with previous element

    - M,X: M < X, shift X to right

    - M,E: M > E, no change

4. Compare P with previous elements

    - P,X: P < X, shift X to right

    - P,M: P > M, no change

5. Compare L with previous elements

    - L,X: L < X, shift X to right

    - L,P: L < P, shift P to right

    - L,M: L < M, shift M to right

    - L,E: L > E, no change

6. Compare E with previous elements

    - E,X: E < X, shift x to right

    - E,P: E < P, shift P to right

    - E,M: E < M, shift M to right

    - E,L: E < L, shift L to right

    - E,E: E = E, no change

**Final Result:**
After applying insertion sort, the array is sorted alphabetically as:

```
A,E,E,L,M,P,X
```

---

## Answer 2

```python
Pseudocode

sort_topologically(digraph):
    visited_nodes = set()
    topo_order = []

    for each node in digraph:
        if node not in visited_nodes:
            depth_first_explore(node, digraph, visited_nodes, topo_order)

    return topo_order

depth_first_explore(current_node, digraph, visited_nodes, topo_order):
    visited_nodes.add(current_node)

    for each neighbor in digraph[current_node]:
        if neighbor not in visited_nodes:
            depth_first_explore(neighbor, digraph, visited_nodes, topo_order)

    topo_order.insert(0, current_node)  # Insert node in topological order after visiting all neighbors

```

Topological Sort: Ensures nodes appear before their dependencies in a DAG.

DFS Logic:

- Perform DFS on unvisited nodes.

- Insert nodes after visiting all neighbors (post-order).

Reverse Order: Nodes are added in reverse completion order to satisfy dependencies.


#### For Graph (a):

Adjacency List (For reference):

```

a: [b, c]
b: [e, g]
c: [f]
d: [a, f, g, b, c]
e: []
f: []
g: [e, f]

```

Start DFS from d (Because it has most dependencies):

* From a -> visit b -> visit e. e has no adjacent/unvisited nodes so, add in result list.
* Visit g -> e already visited so skip -> Insert f -> insert g -> insert b
* From a -> visit c -> f already visited so skip -> insert c -> insert a
* Insert d

**Final topological sort - `d,a,c,b,g,f,e`**


### For graph (b):

Adjacency List (For reference):

```

a: [b]
b: [c]
c: [d]
d: [g]
e: [a]
f: [b, c, e, g]
g: [e]

```

There is a cycle in this graph: `g -> e -> a -> b -> c -> d -> g`.


The presence of a cycle means that **topological sorting cannot be performed** on this graph.

---

## Answer 3

No, we can't use the order in which vertices are pushed because for a topological order. A vertex must come after all of it's pre-requisites in that order. 

That's the reason correct order is computed when vertices are popped off the stack - after all their pre-requisites have been fully explored.

---

## Answer 4 

The source-removal algorithm works like this:
 
1. **Identify sources** : Find nodes with no incoming edges.
 
2. **Remove source** : Eliminate the source node and its outgoing edges.
 
3. **Add to order** : Add the source to the topological order.
 
4. **Repeat** : Continue identifying and removing sources until all nodes are processed. If no sources are found and nodes remain, a cycle exists.

#### Graph a

1. **Identify source d**  (no incoming edges), remove d, and update the graph. 
  - Topological order: [d]
 
2. **Source a** , remove $a$, update graph. 
  - Topological order: [d, a]
 
3. **Source b** , remove b, update graph. 
  - Topological order: [d, a, b]
 
4. **Source c** , remove c, update graph. 
  - Topological order: [d, a, b, c]
 
5. **Source f** , remove f. 
  - Topological order: $[d, a, b, c, f]$
 
6. **Source g** , remove g. 
  - Topological order: $[d, a, b, c, f, g]$
 
7. **Source e** , remove $e$. 
  - Topological order: $[d, a, b, c, f, g, e]$

Final Topological Order: 
$$
 d, a, b, c, f, g, e 
$$

#### Graph b


1. **Source $f$** , remove $f$, and update the graph. 
  - Topological order: $[f]$
 
2. **Source $e$** , remove $e$, and update the graph. 
  - Topological order: $[f, e]$

3. **Source $a$** , remove $a$, and update the graph. 
  - Topological order: $[f, e, a]$
 
4. **Source $b$** , remove $b$, and update the graph. 
  - Topological order: $[f, e, a, b]$
 
5. **Source $c$** , remove $c$, and update the graph. 
  - Topological order: $[f, e, a, b, c]$
 
6. **Source $d$** , remove $d$, and update the graph. 
  - Topological order: $[f, e, a, b, c, d]$
 
7. **Source $g$** , remove $g$. 
  - Topological order: $[f, e, a, b, c, d, g]$

### Final Topological Order: 
$$
 f, e, a, b, c, d, g 
$$


---

## Answer 5

$log_2(n)$ means number of times (x) to which 2 should raised until we get 2. Example: 

if n = 8, we know $2^3$=8. So, $log_2(8)$=3. x in this case is 3.

We can use this property. Divide n by 2 and count how many division are needed to reach 1.


```python
def log2_decrease_by_half(n):
    if n == 1:
        return 0  # Base case: log2(1) = 0
    else:
        return 1 + log2_decrease_by_half(n // 2)  # Recursive case: each step halves n
```


#### Recurrence Relation: 

$$
 T(n) = T\left(\frac{n}{2}\right) + 1 
$$
In this case, each recursive step does exactly **1**  unit of work.

### Solving with Substitution: 
 
1. Start with the recurrence:
$$
 T(n) = T\left(\frac{n}{2}\right) + 1 
$$
 
2. Expand for $T\left(\frac{n}{2}\right)$:
    $$
    T\left(\frac{n}{2}\right) = T\left(\frac{n}{4}\right) + 1 
    $$

    Substituting this back into the original equation:
    $$
    T(n) = T\left(\frac{n}{4}\right) + 1 + 1 = T\left(\frac{n}{4}\right) + 2 
    $$
 
3. Expand again:
    $$
    T\left(\frac{n}{4}\right) = T\left(\frac{n}{8}\right) + 1 
    $$

    Substituting:
    $$
    T(n) = T\left(\frac{n}{8}\right) + 3 
    $$
 
4. Continue this until the base case is reached:
    $$
    T(n) = T\left(\frac{n}{2^k}\right) + k 
    $$

    The base case is when $\frac{n}{2^k} = 1$, or $k = \log_2 n$. Therefore:
    
    $$
    T(n) = T(1) + \log_2 n 
    $$
 
5. Since $T(1) = 1$, we get:
    $$
    T(n) = 1 + \log_2 n 
    $$


The final time complexity of this algo is `1 + $log_2(n)$`. Or, **O($log(n)$)**