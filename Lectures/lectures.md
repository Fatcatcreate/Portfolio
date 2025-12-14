# Lecture Log

## Lecture: Responsible Modelling

**Speaker:** Erica Thompson  
**Link:** <https://youtu.be/aS6R3epK3ms>  

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

**Link:** <https://www.youtube.com/watch?v=TQKgB0RnjeY>

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

**Link:** <https://www.youtube.com/watch?v=zF5nVMG-Big>

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

**Link:** <https://www.youtube.com/watch?v=cq-rAUwQoaM&>

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

**Link:** <https://www.youtube.com/watch?v=W_A6NbqpTW8>

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

**Link:** <https://www.youtube.com/watch?v=aS6R3epK3ms>

### Key Notes

1. Laplacian is Degree matrix minus Adjancency (refer to previous notes)
2. A = A^T (Transpose)
3. diag(X) is Matrix X such that X_ii is equal top x_i and 0 otherwise
4. Consensus Dyanmics:
Each node starting with a state
X = -LX
<img src="https://latex.codecogs.com/svg.latex?X_i%20=%20\sum_j%20(A_{ij}X_j%20-%20\delta(c_i,c_j)X_i)" />

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

## Lecture: How Does Mathematics Last? Heritage and Heritage-making in Mathematics - Caroline Ehrhardt

**Link:** <https://www.youtube.com/watch?v=yVbRNxNnHwE&t=1s>

### Key Notes

1. Library Mathematician's Labaratory, passing knowdlege on and acting a repository for mathematetical knowledge.
2. 19th century, bulletin of new sciences to circulate new science ideas, refers to Louis Francois Antoine Arbogast, was said to contain a full history of maths to reconstruct science if lost, yet is no longer unable to be found. 
3. Few libraries of 19th century mathematicians remain in physical form
4. Mathematicians bound together separate papers into a single volume
5. Referencial books also last
6. Euclid Elements has lasted very very long 
7. New editions help them not be forgotten not due to the text itself 
8. Mathematcis saved throuhg the curriculum not just through academia
9. Hopeful as the investmebnt in libraries using historical collections shows interest 

### Citation

How Does Mathematics Last? Heritage and Heritage-making in Mathematics - Caroline Ehrhardt |


## Lecture: Networks: Part 6 - Oxford Mathematics 4th Year Student Lecture  

**Link:** <https://www.youtube.com/watch?v=cctHyGe5D_k>

### Key Notes

1. randok walker on a nod, proabbitlties that ti goes to another node joiend tio it, sum to 1
2. Same as a markov chain
3. porpbailtity is T_i_j= A_i_j/k_i eg adjancency over degree
4. In diercted network, is an absoirbign state if no outgoing links, also ergodic set, ehre eu cant leave tyhat set of nodes adn Transient nodes which is a node not part of an ergodic set
5. the trasnoition matrix 
<img src="https://latex.codecogs.com/svg.latex?\color{White}T_{ij}%20=%20\frac{A_{ij}}{R_i}" />
<img src="https://latex.codecogs.com/svg.latex?\color{White}\tilde{A}_{ij}%20=%20\frac{A_{ij}}{\sqrt{R_i%20R_j}}" />
<img src="https://latex.codecogs.com/svg.latex?\color{White}\tilde{A}_{ij}%20=%20\sum_{e=1}^{N}%20\lambda_e%20\bar{u}_e%20\bar{u}_e^T" />
<img src="https://latex.codecogs.com/svg.latex?\color{White}\langle%20\bar{u}_e,%20\bar{u}_{e'}%20\rangle%20=%20\delta_{ee'}" />
<img src="https://latex.codecogs.com/svg.latex?\color{White}T_{ij}%20=%20\frac{\sqrt{R_j}%20\tilde{A}_{ij}}{\sqrt{R_i}}" />
<img src="https://latex.codecogs.com/svg.latex?\color{White}T%20=%20D^{-1/2}%20\tilde{A}%20D^{1/2}" />
<img src="https://latex.codecogs.com/svg.latex?\color{White}\text{The%20same%20eigenvalues}%20\Rightarrow%20\text{real}" />
<img src="https://latex.codecogs.com/svg.latex?\color{White}\mathbf{u}_e^L%20=%20((u_e)_1\sqrt{R_1},%20\dots,%20(u_e)_N\sqrt{R_N})" />
<img src="https://latex.codecogs.com/svg.latex?\color{White}\mathbf{u}_e^R%20=%20((u_e)_1/%20\sqrt{R_1},%20\dots,%20(u_e)_N/%20\sqrt{R_N})" />

6. Probabiltiyt to be b on a node at a certain time , is the sum of whast going on on all egein direction 
is 
<img src="https://latex.codecogs.com/svg.latex?\color{White}P_i(t)=\sum_{\ell=1}^{N}a_{\ell}(t)\,(\mathbf{u}_{\ell}^R)_i\sqrt{R_i}" />
<img src="https://latex.codecogs.com/svg.latex?\color{White}a_{\ell}(t)=\lambda_{\ell}^{t}a_{\ell}(0)" />

all eigen values between [-1,1]
=1 when connected and -1 when bipartite

as 
t-> <img src="https://latex.codecogs.com/svg.latex?\color{White}\infty" />

becomes the left eigenvecxtor of the trasnition matrix 

7.as t gets large 
<img src="https://latex.codecogs.com/svg.latex?\color{White}|\lambda_{\ell}|^{t}\ll|\lambda_{2}|^{t}" />

8.when 1-λ_2 is small then slow relaxation when large then converges very fast

9. conductance for a graph is the min of (number of edges that conenct the group of nodes S to the remining group/ min of the volume of S very S complement )
e.g eimagine circle around haf the graph and two links to the rest 

10. populaation entwrok ecah with some value Y reprenstaing some stat, eg age

Use *Respondent driven sampling (RDS)*, you choose one perosn, and get them to refer their friends and they try to get their friends for the survery, giving youn a sequnce of the y sampled from the set of y.    


Each person refers one friend and you can resample people, sampling done by a discrete time random walk, this is biased as this samples the high degree nodes more, so we must counterbalance 
 
11. We counterbalance in *RDS 2* we can give less importance to high degree nodes to try to counterbalance 

12. However if cannot backtrack then can use non backtracking random walk  

13. Continuous time random walks,  in Node cenrtic continuosu time random walks, time intervals distributed by an expoenntial, when more neihbours lower rate, 

d/dt P(t) = P(t) (-I +T)

so P_i = k_i/2m

P_i being stationary distribution 

In edge centric conitnous random walk, if node has mroe edges moves more quickly, so number or edges propertional to rate  of walk 

d/dt P(t) = P(t) (-D+A)

P_i = 1/N 

### Citation

Networks: Part 6 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics |

## Lecture: Going for Gold: the Mathematics of Sporting Glory - Amandine Aftalion

**Link:** <https://www.youtube.com/watch?v=UeGXwvXqUNA>

### Key Notes

1. Centripetal is mv^2/r
2. Effort is mental, eg depending if your brain works more or less gives you more or less enegry for other processes
3. a single extra metre of radius can make or break a record
4. For a bike COM above area and when falling mv^2/r
5. For basketball as the COM changes as they raise arms as the COM follows a parabola they stay in air for longer
6. Better to have someon in front for biking as vortices replaced by person
7. Swimming slightly under water is least water resistance as less pressure drag
8. When Paris olympics had lower pool depth the waves woud; relfect faster agaisnt them slowing them down comapred to 3m

### Citation

Going for Gold: the Mathematics of Sporting Glory - Amandine Aftalion | Oxford Mathematics |


## Lecture: Networks: Part 7 - Oxford Mathematics 4th Year Student Lecture  

**Link:** <https://www.youtube.com/watch?v=---gzhcMEHA>

### Key Notes

1. Retuirn to Newmann Girvan Modularity lecture 4, without forum,la here but more generally epxected links between i and j from their degree minus adjacence. Qualitity of a partition, is based on actual edges vs expected edges between 2 nodes based on a null graph, eg random edges but preservong node degrees * the delta eg 1 if in same community 0 if not. This can be optimised.
2. We could have multi level modualrities eg sum communtiies, and optimising a modularity will only optimsie for one level of community.
we can add a parmater gamma * the expected, large gamma is high reoslition so more like single nodes small gama is oppisite.
This is Riechtardt and Bornholdt.
3. Another way is setting a random walker on the network and hell prpobably spend mroe time in comuntiies befroe leaving
4. Dynamcis to find communtiies. We use coding theory. We label nodes with binary numbers, and try reduce numebrings. 
5. we write down an ordered list of nodes visited, but we wirte themn in binary such that ecah si not a prefix of another and we concatnate these.
We use huffman code to label the nodes, shorter onees given to more often visited nodes and longer ones for less visited nodes.  
6. When wlaker neters community, we yuse a code word to reprsent entry, rhen we contacentae the sequnce of visited nodes and then we reprsent he exuit of the sommuntiy withba code word, eg each community gets its own dictioinary of huffman codes, when ti exists iut enters other communitya dn we preat this process.
7. We have quality fucntion of the theroretical limit of how considlry we can specify a wlak ins aid way with a given aprtition. Thsi is the map equation. 
8. Partition defined as betetr if wlakers dtays long time in each patition befroe switching, eg ones nextc to each toher same parittion so doesnt switch quickly, this can be measured with Markov Stability, of probabiltity of wlaker to be at same comunity at time $$t_0$$ and time $$t_0+t$$ when system is in equilibrium - probabloity for two ranomw alkers to be in c or porobaboltity of wlaker to be in the same community at $$t_0$$ and at infinity.
9. In weighted directed networks moduelairty will oput strong wieghts togetehr comapred to Markov will priortise flows base cycles.
10. Tiome can be a rsuoltion aprmaters for poission witing times, at t = 0 we get 1- sum of (average degree of comunntiy)^2/2m with sommuntiies as singlw nodes
11. Stabiltoty at time t is miodualirty of a time dependent grapoh of adjacency matric X(t) = is the pflow of proabbiltoty that goes from node j to i intime t

### Citation

Networks: Part 7 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics |

## Lecture: Networks: Part 8 - Oxford Mathematics 4th Year Student Lecture  

**Link:** <https://www.youtube.com/watch?v=67xLOcA5Qbs>

### Key Notes

1. COntact networks and contact tracing, we trace contacts before they can infect. 
2. networks may be better than spatial modellign for seeing how diseases spread
3. Comparmental mdoels have imnteracting agents with different states corresponding to whther they are susceptible, infected and recovered for example. Mean field assumes random interactions between any pair of agents. $$S(t)$$ is portion that are sucsveotible and $$I(t)$$ is proprtion infected. chnage in $$S(t)$$ over time is proprtions to rovery rate μ and a β rate which detrmines how suceptiple people become infected
4. $$I^*$$ is $$1-μ/β$$ if the ratio is less thna one hjwoever it euals 0 when greater than one, eg whther mnor epeople recover than get infected or more get infected that recover.
5. SIR has recovered people which cannot be reinfected, but s otherwise simialr, this leads to s astate with no infected people only recoevered or suceptible.
6. IF $$S(t)$$ near one then rtae of chnage of $$I(t)$$ is $$β*S(t)I(t)$$ therefore $$I(t) \propto e^{βt}$$
7. Network with number of people going from one to another node is a wieghjted edge, eg nodes represent places. Therefore you do mean field over a lcoal area 
8. Metapopautoon networkm, agents random walk. This has been shown above. For interactions for SIS. 


Rules for the interactions:

$$
N_{S,i}, \quad N_{I,i}
$$

The differential equations are given by:

$$
\frac{d N_{S,i}}{dt} = -\beta N_{S,i} N_{I,i} + \mu N_{I,i} - D_S \sum_{j} N_{S,j} \tilde{L}_{ij}
$$

$$
\frac{d N_{I,i}}{dt} = \beta N_{S,i} N_{I,i} - \mu N_{I,i} - D_I \sum_{j} N_{I,j} \tilde{L}_{ij}
$$

As \( N_S \to N_{I,i} \), we have:

$$
N_S^* = \frac{r_i}{\langle k \rangle} \tilde{N}
$$

Therefore,

$$
\frac{\partial N_{I,i}}{\partial t} = \frac{\beta N}{\langle R \rangle} \tilde{r}_i N_{I,i} - \mu N_{I,i} - D_I \sum_{j} N_{I,j} \tilde{L}_{ij}
$$

# SIS Model - Additional Equations

For all \(i \in \{1, \dots\}, \tilde{N}\) is a solution:

$$
\frac{d \tilde{N}}{dt} \approx B \tilde{N}_B
$$

The relationship for \( B_{ij} \):

$$
B_{ij} = S_{ij} \left( \frac{\beta N}{\langle 2k \rangle} \tilde{N}_i - \mu \right) - D_i \tilde{L}_{ij}
$$

When \( D_I \to \infty \), we have:

$$
N_{I,i} = \frac{\tilde{R}_i N_I}{\langle \tilde{L} \rangle} \tilde{N}
$$

Thus, the total \( N_I \) is:

$$
N_I = \sum_{i \in S} N_{I,i}
$$

The equation for the time evolution of \( N_I \) is:

$$
\frac{d N_I}{dt} = \left( \frac{\beta N \langle \tilde{R}^2 \rangle}{\langle \tilde{L} \rangle^2} - \mu \right) N_I
$$

Finally, the condition for the system is:

$$
\frac{\beta}{\mu} > \frac{\langle \tilde{R} \rangle}{\langle \tilde{L} \rangle^2} \frac{\tilde{N}}{N}
$$



 if avarega of k tilde suare is much larger than square of average  k tilde menaingtopoogy of grpah plays massive role menaing it hs a broad distribution, it implies that disease will spread quickly 

9. Contact networks, we assume nodes are people not places and edges are contacts, hard to measure adn edges are not static irl, but we assume they are. each ndoe in state $$x_i$$ which is 1 if infected 0 if susceptible. we see how $$<x_i(t)>$$ chnages over time.

$$
\frac{d x_I}{dt}
$$ 
= β * sum of adjacency for i and j times probabiltity i is sucseptible and j is infected.

10. Use hypergraph to represent non binary interactions

### Citation

Networks: Part 8 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics |