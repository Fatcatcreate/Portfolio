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
Networks: Part 1 – Oxford Mathematics 4th Year Student Lecture | Oxford Mathematics | Insight: 