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
7. ![equation](https://latex.codecogs.com/svg.latex?\bg_white%20\sum_{i=1}^{N}k_i=\sum_{i=1}^{N}\sum_{j=1}^{N}A_{ij}=2M)

(Sum of degrees is twice number of edges)
8. Friendship paradox, ur friends will most likely have more friends than you as its not a regular network
9. A walk is a succession of adjacent nodes, path when each ndoe is visited only once, and shortest walk is a path
10. Distance is shortest path between two nodes, satisfies triangle inequality, use Dijikstra for finding this
11. average distance L = ![equation](https://latex.codecogs.com/svg.latex?\bg_white%20L=\frac{2}{N(N-1)}\sum_{i=1}^{N}\sum_{j=1}^{i-1}d(i,j))


12. Diameter is farthest distance between two nodes
13. Weakly conected is for a directed graph underlying undirected graph there is a path between every two vertices, strongly connected, if directed and still directed path exists
14. Link prediction, simple method for predicted future link is two unconnected nodes connected to mnay open triangles, will probably connect soon
15. C_i = adbundance of traingles in neighbourhood of i, 
(number of traingles including node i/(k_i(k_i-1)/2))
*between 0 and 1*

16. Centrality, simplest method is just using the degree, also closeness centrality where is 1/the average distance, also betweeness centrality, is this nod part of many shortest paths between many nodes in the system

![equation](https://latex.codecogs.com/svg.latex?\bg_white\LARGE\sum_{i=1}^{N}k_i=\sum_{i=1}^{N}\sum_{j=1}^{N}A_{ij}=2M)

Kats Centrality considers all walks not just shortest but weighst towards shorter ones
17. Can also represent matrix in Laplacian matrix and Normalised Laplacian matrix

Laplacian = Degree matrix (degrees on diagonal) - A
Normalised Laplacian = Identity - D^-1/2 A D^-1/2

All eigen valiues are postituve of 0
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
$ s = 1 - e^{- \langle k \rangle S} $
7. ER degree distribution is binomial, which is unrealistic so one way to generalise is configuration model
8. Configuration model you fix degrees of nodes and therefore fix degree distribtion 
9. Question arises then for node i degree k_i and node j degree k_j what is expected number of links between them whioch becoimes 

$ {k_i}*{k_j}/2M $

10. Motif analysis uses random sampling vs empirical dataset to find which motifs are imprtant 
11. Babasi and Albert for growing networks, new nodes connect proportionally to the degree of the nodes currently in the networks, eg more likely to be friends with someone withba lot of friends, also follows power law distributions


### Citation
Networks: Part 3 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics |