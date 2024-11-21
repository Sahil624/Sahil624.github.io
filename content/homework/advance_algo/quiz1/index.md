+++
title = 'Quiz 1'
date = 2024-11-21T11:57:07-05:00
draft = false
summary = "My homework backup for Advance Algorithm subject."
series = ["Advance Algorithm",]
tags = ["Advance Algorithm", "midterm", "university", "school"]
author= ["Me"]
+++



## Answer 1

1. This algorithm sorts the given array in ascending order. The exact algorithm followed here is Selection Sort, where an element is replaced with smallest element on unsorted (right) side of the array.
2. There are 2 loops involved here. Outer loop L1[0 -> n-2] and inner loop L2[i+1 -> n-1]. Both of these loops have to execute to their extreme (n-2 and n-1 respectively) regardless weather input array is sorted or not. Therefore, the best time complexity is O(n$^2$).
3. There are 2 loops involved here. Outer loop L1[0 -> n-2] and inner loop L2[i+1 -> n-1].

    $$
    T_{\text{worst}}(n) = \sum_{i=0}^{n-2} \sum_{j=i+1}^{n-1} 1 
    $$
    
    For each iteration of outer loop i, the inner loop will iterate over i+1 -> n-1 elements. Which means for each i'th loop inner loop executes (n - 1 - i) times.

    i.
    $$
    T_{\text{worst}}(n) = \sum_{i=0}^{n-2} (n-1-i) 
    $$

    Dissecting it,

    ii. 
    $$
    T_{\text{worst}}(n) = \sum_{i=0}^{n-2} (n-1) - \sum_{i=0}^{n-2} i 
    $$

    We know that in
     $$
    \sum_{i=0}^{n-2} (n-1)
    $$
    (n-1) is added (n-1) times. So,

    iii.

     $$
    \sum_{i=0}^{n-2} (n-1) = (n-1)(n-1)
    $$

    We also know

    iv.

    $$
    \sum_{i=0}^{k} i = \frac{k(k+1)}{2} 
    $$

    we can use iii. and iv. for k=n-2 in step ii.

    v.
    $$
    T_{\text{worst}}(n) = (n-1)(n-1) - \frac{(n-2)(n-1)}{2} 
    $$

    $$
     = (n-1) \left[ (n-1) - \frac{n-2}{2} \right] 
    $$

    After simplification the final complexity is,

    $$
    T_{\text{worst}}(n) = \frac{n(n-1)}{2} 
    $$

## Answer 2

1. The given algorithm recursively prints n^2 values trice except a base case (n=1) for each recursion call.

    For n=3, the the algo will print values in following order.

    `1, 4, 1, 9, 1, 4, 1`

    This will be more understandable in a visual tree format. Each recursion call can be represented as an branch.

    ```
          9
         / \
        4   4
       / \ / \
      1  1 1  1

    ```

2. We need to set up the recurrence relation for the algorithm’s basic operation (the print statement) and solve it using backward substitution.

    The recurrence calls for the total number of print operations 

    $$
    T(n) = 2T(n-1) + 1 \quad \text{for} \quad n > 1 
    $$
    and
    $$
    T(1) = 1 
    $$

    Now, we perform substitutions to find a general solution.
    $$
    T(n) = 2T(n-1) + 1 
    $$

    Substitute  $T(n-1)$ :

    $$
    T(n) = 2(2T(n-2) + 1) + 1 = 2^2T(n-2) + 2 + 1 
    $$

    Substitute $T(n-2)$:

    $$
    T(n) = 2^2(2T(n-3) + 1) + 2 + 1 = 2^3T(n-3) + 2^2 + 2 + 1 
    $$

    Continue this substitution process till k:
    $$
    T(n) = 2^kT(n-k) + 2^{k-1} + 2^{k-2} + \dots + 2 + 1 
    $$

    When $k = n-1$, we reach the base case $T(1) = 1$:

    $$
    T(n) = 2^{n-1}T(1) + 2^{n-2} + 2^{n-3} + \dots + 2 + 1 
    $$
    Since $T(1) = 1$,  we have:

    $$
    T(n) = 2^{n-1} + 2^{n-2} + \dots + 2 + 1 
    $$

    This is a geometric series with a sum of:
    $$
    T(n) = 2^n - 1 
    $$

    Thus, the number of print operations grows as $T(n) = 2^n - 1$, which gives us the time complexity:
    $$
    T(n) \in \Theta(2^n) 
    $$

    ### Conclusion: 
    The time complexity of the `Do_Something` algorithm is **exponential** , i.e., ${O(2^n)}$.


## Answer 3



Proving $\frac{n(n-1)}{2} \in \Theta(n^2)$. We need to demonstrate that $$\frac{n(n-1)}{2}$$ 
belongs to the asymptotic complexity class $\Theta(n^2)$. To do this, we will apply the formal definition of $\Theta$-notation.Formal Definition of $\Theta$-notation: A function $f(n)$ is said to be in $\Theta(g(n))$ if there exist positive constants $c_1, c_2,$ and $n_0$ such that for all $n \geq n_0$:
$$
 c_1 g(n) \leq f(n) \leq c_2 g(n) 
$$

In our case, we want to prove that:
$$
 f(n) = \frac{n(n-1)}{2} \quad \text{is in} \quad \Theta(n^2) 
$$
This requires us to find constants $c_1$, $c_2$, and $n_0$ such that:
$$
 c_1 n^2 \leq \frac{n(n-1)}{2} \leq c_2 n^2 \quad \text{for large enough} \ n 
$$

Step 1: Simplifying $f(n)$
The function we are working with is:

$$
 f(n) = \frac{n(n-1)}{2} 
$$

Expanding this expression gives:
$$
 f(n) = \frac{n^2 - n}{2} 
$$

We can rewrite this as:
$$
 f(n) = \frac{n^2}{2} - \frac{n}{2} 
$$

### Step 2: Establishing the Lower Bound 
To show the lower bound, we need to find a constant $c_1$ such that for large $n$:

$$
 c_1 n^2 \leq \frac{n^2}{2} - \frac{n}{2} 
$$

Since $\frac{n^2}{2} - \frac{n}{2}$ is dominated by $\frac{n^2}{2}$ as $n$ increases, we can focus on the $\frac{n^2}{2}$ term. For sufficiently large $n$, the $\frac{n}{2}$ term becomes negligible. Therefore, we can say:

$$
 \frac{n^2}{2} - \frac{n}{2} \geq \frac{n^2}{4} 
$$

Thus, we can choose $c_1 = \frac{1}{4}$, and for large $n$:

$$
 \frac{1}{4} n^2 \leq \frac{n^2}{2} - \frac{n}{2} 
$$

### Step 3: Establishing the Upper Bound 
Now we find a constant $c_2$ to show the upper bound for large $n$:

$$
 \frac{n^2}{2} - \frac{n}{2} \leq c_2 n^2 
$$

As $n$ grows, the $\frac{n}{2}$ term becomes insignificant compared to $\frac{n^2}{2}$. Therefore, we can approximate the upper bound as:

$$
 \frac{n^2}{2} - \frac{n}{2} \leq \frac{n^2}{2} 
$$

Hence, we can choose $c_2 = \frac{1}{2}$. This gives us the inequality:
$$
 \frac{n^2}{2} - \frac{n}{2} \leq \frac{1}{2} n^2 
$$

### Step 4: Conclusion 

By establishing both the lower and upper bounds, we have shown that:
$$
 \frac{1}{4}n^2 \leq \frac{n^2}{2} - \frac{n}{2} \leq \frac{1}{2}n^2 
$$

Thus, we conclude that:
$$
 \frac{n(n-1)}{2} \in \Theta(n^2) 
$$
This completes the proof using the formal definition of $\Theta$-notation.

## Answer 4

Increasing Order of Growth: 
 
1. $6$ (constant)
 
2. $\log n$, $\ln(n)$, $\log(n+1)^{100}$ (logarithmic functions)
 
3. $\sqrt[3]{n}$, $\sqrt[2]{n}$ (root functions)
 
4. $n$ (linear)
 
5. $n \log n$ (log-linear)
 
6. $n^2$, $n^2 + \log n$ (quadratic)
 
7. $n^3$ (cubic)
 
8. $n - n^3 + 7n^5$ (behaves as $n^5$, quintic)
 
9. $(1/3)^n$ (exponentially decreasing)
 
10. $(3/2)^n$, $2^n$, $2^{n-1}$, $3^n$ (exponential)
 
11. $n!$, $(n-1)!$ (factorial)


## Answer 5



a) 
- $a = 4$, $b = 2$, $f(n) = n$
 
- $\log_b a = \log 4 = 2$
 
- Since $f(n) = O(n^2)$, by **Case 1** :
$$
 T(n) = \Theta(n^2) 
$$


---

b)
- $a = 4$, $b = 2$, $f(n) = n^2$
 
- $\log_b a = \log 2 4 = 2$
 
- Since $f(n) = \Theta(n^2)$, by **Case 2** :
$$
 T(n) = \Theta(n^2 \log n) 
$$


---

c)
- $a = 4$, $b = 2$, $f(n) = n^3$
 
- $\log_b a = \log 4 = 2$
 
- Since $f(n) = \Omega(n^2)$ and regularity holds, by **Case 3** :
$$
 T(n) = \Theta(n^3) 
$$


---

d) 
- $a = 4$, $b = 2$, $f(n) = 1$
 
- $\log_b a = \log 4 = 2$
 
- Since $f(n) = O(n^2)$, by **Case 1** :
$$
 T(n) = \Theta(n^2) 
$$

