# Lecture Log

## Lecture: Responsible Modelling
**Speaker:** Erica Thompson  
**Link:** https://youtu.be/aS6R3epK3ms  

### Key Notes
1. All models are simplifications, 'All models are wrong but some aree useful'
2. Model is useful but can be judged by data, but depends on whether the past is a good measure of the future
3. Assumption driven vs data driven regime.
4. What have we excluded from the models
5. Compare models that all model similar things to try to determine things about reality, cat metaphor
6. Useful even if wrong
7. Models dependent on value judgements for our models and what WE value in terms of society
8. Error term is a function of those making it
9. Model can work very well in interpolation space but vary drastically in extrapolation space


### Citation
Oxford Maths Public Lecture | Responsible Modelling (Erica Thompson) | Insight: Models are at best bad approximations but the error and bias is useful to understanding


## Lecture: Networks: Part 1 - Oxford Mathematics 4th Year Student Lecture  
**Link:** https://www.youtube.com/watch?v=TQKgB0RnjeY

### Key Notes
1. Focus on Connectivity removes very many details that are irrelevant
2. μ dx/dt = L x relation between structure and dynamics
3. Randomness vs Order in social networks
4. Poisson Processes Random sequnce events happening in time
5. Has constant rate of events and independent events
6. In  Δt goes to 0 either 1 or 0 events. Therefore P(1,Δt) is just λ Δt with that being the rate and 1- that for P(0, Δt)
7. Following:
P(m, t + Δt) = P(m, t) P(0, Δt) + P(m-1, t) P(1, Δt)  
                (m and 0)             (m-1 and 1)

lim Δt → 0

{ dP(m, t)/dt = λ P(m-1, t) - λ P(m, t)     m ≥ 1  
  dP(0, t)/dt = -λ P(0, t) }

8. Solving P(0,t) = e^-λt and Probability that you wait a certain time ε before an event is exponential as λe^-λt
9. As independent Past is irrevelant for ε
10. In random walk porbability at posttion as time t, is porbability that at t-1 you are at X' and that you jump directly to X for all infinite X' positions
11. Use a fourier transofrm to analyse the integral and use that at t = 0 all probabilities on X = 0
12. Power law distributions on log-log scale appears when graphing frequency against degree of each node (looks like y= -x + top of y axis) 
13. y = cx^-α
It is scale invariant


### Citation
Networks: Part 2 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics |

## Lecture: Networks: Part 2 - Oxford Mathematics 4th Year Student Lecture  
**Link:** https://www.youtube.com/watch?v=zF5nVMG-Big

### Key Notes
1. Networks are a map of nodes connected by links, set of nodes and a set of edges (V,E)
2. Any edge decsribed as pair of nodes (V,V')
3. Weight funtion for weighting edges generally positive
4. N x N adjacency matrix for encoding a network (inefficient for very large networks though)

Aij = {
  1 if i is adjacent to j
  0 is otherwise
}
Symmetric if undirected
5. Link list, encodes graph through all edges:
{(m1,v1),(m2,v2)...(M M, V M)}
6. Degree of node is number of links = sum of adjacency matrix row i for node i, regular network if all have same degree 
7. <img src="https://latex.codecogs.com/svg.latex?\color{White}\sum_{i=1}^{N}k_i=\sum_{i=1}^{N}\sum_{j=1}^{N}A_{ij}=2M" />


(Sum of degrees is twice number of edges)
8. Friendship paradox, ur friends will most likely have more friends than you as its not a regular network
9. A walk is a succession of adjacent nodes, path when each ndoe is visited only once, and shortest walk is a path
10. Distance is shortest path between two nodes, satisfies triangle inequality, use Dijikstra for finding this
11. average distance L = <img src="https://latex.codecogs.com/svg.latex?\color{White}L=\frac{2}{N(N-1)}\sum_{i=1}^{N}\sum_{j=1}^{i-1}d(i,j)" />



12. Diameter is farthest distance between two nodes
13. Weakly conected is for a directed graph underlying undirected graph there is a path between every two vertices, strongly connected, if directed and still directed path exists
14. Link prediction, simple method for predicted future link is two unconnected nodes connected to mnay open triangles, will probably connect soon
15. C_i = adbundance of traingles in neighbourhood of i, 
(number of traingles including node i/(k_i(k_i-1)/2))
*between 0 and 1*

16. Centrality, simplest method is just using the degree, also closeness centrality where is 1/the average distance, also betweeness centrality, is this nod part of many shortest paths between many nodes in the system

<img src="https://latex.codecogs.com/svg.latex?\color{White}\sum_{i=1}^{N}k_i=\sum_{i=1}^{N}\sum_{j=1}^{N}A_{ij}=2M" />


Kats Centrality considers all walks not just shortest but weighst towards shorter ones
17. Can also represent matrix in Laplacian matrix and Normalised Laplacian matrix

Laplacian = Degree matrix (degrees on diagonal) - A
Normalised Laplacian = Identity - D^-1/2 A D^-1/2

All eigen valiues are postitve of 0
 for Adjacency λ_1 > λ_2 > λ_u
 Reverse for laplacian

### Citation
Networks: Part 2 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics |

## Lecture: Networks: Part 3 - Oxford Mathematics 4th Year Student Lecture  
**Link:** https://www.youtube.com/watch?v=cq-rAUwQoaM&

### Key Notes
1. Erdos Reyni Model (ER), 1959, random model of a graph g(N,q) (q is porbbaly that a pair of nodes is connected), eg binary process for each set of nodes whether you generate or not
2. Probability of M edges is a binomial based of probability q
3. Average distance is around log N/log k (average degree)
4. Clusering coefficient eg how much nodes cluster is k/ N-1 (eg for large n and small k its very low as they arnt connecting alot)
5. AS average degree increases size of giant compoinbts increases logarithmically looking? at perpolation transition it shifts from liens to giant balls and trees
6. define u as probability that node is not part of giant component, eg if no giant component u = 1 else smaller:
if i does not belong to giant compoent cannot be adjacent to j if j is connected to GC as otherwise it would be aprt of it, by follwoing through we create a realtion forming u in terms of q, taking limit, 
perpolation is that s that node belongs to GCC is 
<img src="https://latex.codecogs.com/svg.latex?\color{White}s=1-e^{-\langle k\rangle S}" />

7. ER degree distribution is binomial, which is unrealistic so one way to generalise is configuration model
8. Configuration model you fix degrees of nodes and therefore fix degree distribtion 
9. Question arises then for node i degree k_i and node j degree k_j what is expected number of links between them whioch becoimes 

<img src="https://latex.codecogs.com/svg.latex?\color{White}\frac{k_i\,k_j}{2M}" />


10. Motif analysis uses random sampling vs empirical dataset to find which motifs are imprtant 
11. Babasi and Albert for growing networks, new nodes connect proportionally to the degree of the nodes currently in the networks, eg more likely to be friends with someone withba lot of friends, also follows power law distributions

### Citation
Networks: Part 3 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics |

## Lecture: Networks: Part 4 - Oxford Mathematics 4th Year Student Lecture  
**Link:** https://www.youtube.com/watch?v=W_A6NbqpTW8

### Key Notes
1. Community detection, so many graphs eg papers or social networks have dense parts thta are weakly connected with other dense paths
2. This can be hierarchical, connected to ER moidels before, this is extended to Stochastic Block Models (SBM) this defines communtiies, not nodes and then have probabilty that a node of community i is connected to a node of community j in a matrix, after assignin nodes to communties
3. Community detection by using a coarser resuolution so tis easier to see communities
4. Graph partitioning where one specifies number and size of groups and tried ot minimise connections between groups
5. Community Detection number of size of groups determined by network and trying to find natural clusters, can sometimes not divide networks
6. Example: partision network into two groups by minimising the cut, eg number of edges cut in half, random chocie grows exponentially so cannot work
7. Use spectral methods eg assigning nodes a spin -1 or 1, you can therefore write the groups in the laplacian of the graph (Laplacian is all eigen values can be written λ_1 < λ_2 < λ_u all greater than 0)
8. Approximation: If one wants a split into n1 and n2 = n − n1 vertices, one orders the components of the Fiedler vector from the largest positive to the smallest negative and picks the n1 largest (smallest) components of the Fiedler vector
9. Newman Girvan Modularity very common, defined as fraction of edges within communities comparing against expected fraction of such edges in an appropriate model, maximising links within communities
10. Modularity, lest say you have a parition with a communtiy that ahs two bits dicsonnected then it is always better to split this group into those two bits
11. Optimisation of modularity is NP complete, and modularity matrix can be negative or positive, then if s_i is the dominant eigenvector s_i = 1 if u_N,i > 0 and the opposite for negative 1, we iterate trying to find optimal partition then repeating again and again however this is slow
12. Greedy can be faster e.g Louvain method (GUY GIVING LECTURE WAS AUTHOR R.Lambiotte)  
13. 

The algorithm is based on two steps that are repeated iteratively. First phase: Find a local maximum

Give an order to the nodes (0, 1, 2, 3, …, N–1)
Initially, each node belongs to its own community (N nodes and N communities)

One looks through all the nodes (from 0 to N–1) in an ordered way. The selected node looks among its neighbours and adopts the community of the neighbour for which the increase of modularity is maximum (and positive).

This step is performed iteratively until a local maximum of modularity is reached (each node may be considered several times)

14. Then you do another phase where u make the nodes the communities that were previously made, with weights summed from the node weights, then you repeat these two phases until no better 
15. Test by seeing whether partitions creates by you match the oens predicated by model, and then graoh conformity
16. Limits of modularity, many partition combinations can have the same level of modularity (landscape very rugged)
17. when graph is too large may merge fine structires of communities, therefore dont use to compare different graphs
20. Modularity definition:

Modularity measures the strength of division of a network into communities.  
It compares the actual density of edges inside communities with the expected density if edges were placed at random.

---

The formula for modularity is

<img src="https://latex.codecogs.com/svg.latex?\color{White}Q=\frac{1}{2m}\sum_{i=1}^{N}\sum_{j=1}^{N}\Big(A_{ij}-\frac{k_i\,k_j}{2m}\Big)\,\delta(c_i,c_j)" />


---

Adjacency matrix

<img src="https://latex.codecogs.com/svg.latex?\color{White}A_{ij}" />


A_{ij} =
1 if there is an edge between nodes i and j
0 otherwise

Node degree

<img src="https://latex.codecogs.com/svg.latex?\color{White}k_i=\sum_{j=1}^{N}A_{ij}" />


This is the number of edges connected to node i.

Total number of edges

<img src="https://latex.codecogs.com/svg.latex?\color{White}m=\tfrac{1}{2}\sum_{i=1}^{N}k_i" />


This counts each edge once (since the sum of degrees counts each edge twice).

Expected edges in random network

<img src="https://latex.codecogs.com/svg.latex?\color{White}\frac{k_i\,k_j}{2m}" />


This is the expected number of edges between nodes i and j in a random network with the same degree distribution.

Community indicator

<img src="https://latex.codecogs.com/svg.latex?\color{White}\delta(c_i,c_j)" />


δ(c_i, c_j) =
1 if nodes i and j are in the same community
0 otherwise

---

Interpretation

Q ≈ 0 : No significant community structure.  
Q > 0.3 : Presence of community structure.  
Q ≈ 0.7 : Strong community structure (rare to go much higher).  




### Citation
Networks: Part 4 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics |



## Lecture: Networks: Part 5 - Oxford Mathematics 4th Year Student Lecture  
**Link:** https://www.youtube.com/watch?v=aS6R3epK3ms

### Key Notes
1. Laplacian is Degree matrix minus Adjancency (refer to previous notes)
2. A = A^T (Transpose)
3. diag(X) is Matrix X such that X_ii is equal top x_i and 0 otherwise
4. Consensus Dyanmics:
Each node starting with a state 
X = -LX
X_i = sum j (A_ij* X_j - δ_ij * x_i)
global consensus is that
x_i converges to x_*
eg reaches the average of all the initial states 
5. Another methos is for a markov chain random walk for diffusion of information, at equilibrium probability to find walker on a node is propertional to degree of that node 
6. time scale seperate eg dy/dt * ε with ε <<1 = f(x,y) and  dx/dt = f(x,y) theerfore time seperation as y chnages sklowere with time, this decouples the system into two regimes 
7. U can neglect many terms as you can seperate the time scale and assume all other regimes have recahed asymptitic behaviour before evaluating
8. If communities with weak connections subspace is close to the eigenspace that encode the communities 
9. Structural equivalence:
2 nodes equivalnec if have same neighbours
10 EEP (Externally Equivalent Partitions ) Partition of netork into Cells C_j such that for any node in C_i is isn connected to a fixed number of nodes in other partitons/group

10. If C is a matrix just saying which nodes are in which partitions it is EEP is Laplacian C = C L^π
L^π being k x k laplacian of quotient graph, k being number of cells
11. By focusing on analysing cells but not nodes using the symmetry it can be a very useful trick
12. With modules you are interested in the timescale seperataion, but with EEPs instead of the eingen valeus you are more concerned with the eigenvectors


### Citation
Networks: Part 5 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics |