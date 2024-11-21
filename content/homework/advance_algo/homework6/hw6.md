+++
title = 'Homework 6'
date = 2024-11-21T11:57:07-05:00
draft = false
summary = "My homework backup for Advance Algorithm subject."
series = ["Advance Algorithm",]
tags = ["Advance Algorithm", "homework", "university", "school"]
author= ["Me"]
+++

## Answer 1
```
                a b c d
Frequency       2 3 2 1
Distribution    2 5 7 8

```

| Original Array (Reversed) | Distribution Array | New Index (Distribution -1) |
| -------------- | ------------------ | --------- |
| b              | [2 5 7 8]          | 4         |
| a              | [2 4 7 8]          | 1         |
| a              | [1 4 7 8]          | 0         |
| b              | [0 4 7 8]          | 3         |
| c              | [0 3 7 8]          | 6         |
| d              | [0 3 6 8]          | 7         |
| c              | [0 3 6 7]          | 5         |
| b              | [0 3 5 7]          | 2         |

Updated Array

`['a', 'a', 'b', 'b', 'b', 'c', 'c', 'd']`


   
## Answer 2

Yes, because same elements are placed in same order as they are seen in input array.

## Answer 3

a.

If m is length of search string. Shift of j'th character is calculated by m - j - 1. Using this algo, shift table is

```
    A C  G T
    5 2 10 1
```

b. 


```
    Text        : TTATAGATCTCGTATTCTTTTATAGATCTCCTATTCTT
    Pattern T 1 : TCCTATTCTT
    Pattern C 2 :  TCCTATTCTT
    Pattern T 1 :    TCCTATTCTT
    Pattern A 5 :     TCCTATTCTT
    Pattern T 1 :          TCCTATTCTT
    Pattern T 1 :           TCCTATTCTT
    Pattern T 1 :            TCCTATTCTT
    Pattern A 5 :             TCCTATTCTT
    Pattern T 1 :                  TCCTATTCTT
    Pattern C 2 :                   TCCTATTCTT
    Pattern C 2 :                     TCCTATTCTT
    Pattern T 1 :                       TCCTATTCTT
    Pattern A 5 :                        TCCTATTCTT
    Pattern T   :                             TCCTATTCTT [FOUND]
```
Pattern will be found after 10 searches

## Answer 4


a.


Hashes
| K  | k(h) |
| -- | --   |
| 30 | 8    |
| 20 | 9    |
| 56 | 1    |
| 75 | 9    |
| 31 | 9    |
| 19 | 8    |

Hash Table

| idx| Values |
| -- | --   |
| 0 |     |
| 1 |   56  |
| 2 |     |
| 3 |     |
| 4 |     |
| 5 |     |
| 6 |     |
| 7 |     |
| 8 |  30->19   |
| 9 |  20->75->31   |
| 10 |     |

b. 


Most key comparisons = 3 (For 31)

c. 


Average can be calculated as following

$$
    \dfrac{1}{6} + \dfrac{1}{6} + \dfrac{1}{6} + \dfrac{2}{6} + \dfrac{3}{6} + \dfrac{2}{6}=1.6666 \approx 1.7
$$

## Answer 5

a. 


Hashes are same as previous question

Closed Hash table.

| 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|    |    |    |    |    |    |    |    | 30 |    |    |
|    |    |    |    |    |    |    |    | 30 | 20 |    |
|    | 56 |    |    |    |    |    |    | 30 | 20 |    |
|    | 56 |    |    |    |    |    |    | 30 | 20 | 75 |
| 31 | 56 |    |    |    |    |    |    | 30 | 20 | 75 |
| 31 | 56 | 19 |    |    |    |    |    | 30 | 20 | 75 |

b.


Most key comparisons = 6 (For 19)


c. 


Average can be calculated as following

$$
    \dfrac{1}{6} + \dfrac{1}{6} + \dfrac{1}{6} + \dfrac{2}{6} + \dfrac{3}{6} + \dfrac{6}{6} \approx 2.3
$$

## Answer 6


| Operation | unordered array | ordered array | binary search tree | balanced search tree | hashing |
| --    | --    | --    | --    | --    | --     |
| Search | Θ(n),Θ(n) | Θ(log n),Θ(log n) | Θ(log n),Θ(n) | Θ(log n),Θ(log n)  | Θ(1),Θ(n)  |
| Insertion | Θ(1),Θ(1)  | Θ(n),Θ(n) | Θ(log n),Θ(n) | Θ(log n)Θ(log n)  | Θ(1),Θ(n)  |
| Deletion | Θ(1),Θ(1)  | Θ(n),Θ(n) | Θ(log n),Θ(n) | Θ(log n)Θ(log n)  | Θ(1),Θ(n)  |