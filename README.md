## Hidden Markov Model Probabilities


The reference haplotype file `refpanel.txt` and target haplotype file `samples.txt` are based on real 1000 Genomes Project human data. 
Both files have both row and column names, representing the sample population and name (row name) and genetic variant ID (column name). 
Now, the reference haplotypes have labels which indicate which populations they are from, which are either CEU = Northern and Western European ancestry, 
or YRI = Yoruba in Ibadan, Nigeria. Similarly, the target haplotypes have labels ASW = African Ancestry in Southwest US. 

We model the haplotypes using hidden Markov model, and paint the first 5 target haplotypes one a time using the reference panel. **The aim is to find which one(s)
have an entirely African genetic background over the investigated chromosome.**


We follow the probabilities: 

1. **Initial Probabilities**

The initial probabilities, denoted as \(\pi_k\), represent the probability of the hidden state \(Q_1\) being \(k\). It can be calculated as:

$$
\pi_k = P(Q_1=k) = \frac{1}{K}
$$

2. **Transition Probabilities**

The transition probabilities, denoted as \(A_{j, k}\), represent the probability of transitioning from hidden state \(j\) to hidden state \(k\) (\(Q_{t+1}=k\) given \(Q_t=j\)). It can be calculated as:

$$
A_{j, k} = P(Q_{t+1}=k \mid Q_t=j) = \begin{cases}
\frac{1-0.999}{K}+0.999, & \text{if } j=k \\
\frac{1-0.999}{K}, & \text{if } j \neq k
\end{cases}
$$

3. **Emission Probabilities**

The emission probabilities, denoted as \(b_{k, t}\), represent the probability of observing output \(o_t\) given hidden state \(Q_t=k\). It can be calculated as:

$$
b_{k, t} = P(O_t=o_t \mid Q_t=k) = \begin{cases}
1-e, & \text{if } o_t=H_{k, t} \\
e, & \text{if } o_t \neq H_{k, t}
\end{cases}
$$

where \(e\) represents the error rate (default value: 0.1).
