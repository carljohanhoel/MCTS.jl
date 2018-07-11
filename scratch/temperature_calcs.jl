#Calculates theoretical and practical visit counts of an example where two different actions are possible and there is a dq difference in expected q-value between them.
#The theoretical visit counts are valid after the network has been trained, when p_i = N_i^(1/tau)/sum(N_i^(1/tau))

using Roots
dq = .01 #Normalized difference in q-values
c_puct = 10
N = 2000
tau = 1.1
# c_puct=10 and tau=1.1 seems like a reasonable fit.
# With dq=10%, N1=1967, p1=0.976
#      dq=5%, N1=1812, p1=0.887
#      dq=1%, N1=1240, p1=0.609
#      dq=0.1%, N1=1025, p1=0.511
# p1 will scale with N, which is reasonable, since with more searches, the result should be more accurate.

if tau > 1.0 #Formula only valid for tau>1.0. For tau<1.0, N1_theoretical=N, N2_theoretical=0.
    f(N1) = dq/(c_puct*sqrt(N)) - ((N-N1)^(1/tau-1) - N1^(1/tau-1)) / ((N-N1)^(1/tau)+N1^(1/tau))
    N1_theoretical = fzero(f,0,N)
    N2_theoretical = N-N1_theoretical
else
    N1_theoretical = N
    N2_theoretical = 0
end

p1_theoretical = N1_theoretical^(1/tau) / (N1_theoretical^(1/tau) + N2_theoretical^(1/tau))
p2_theoretical = N2_theoretical^(1/tau) / (N1_theoretical^(1/tau) + N2_theoretical^(1/tau))

## Practical example
p1 = 0.5 #Starting probs
p2 = 1-p1

f(N1) = dq/(c_puct*sqrt(N)) - (p2/(N-N1)-p1/N1)
N1 = fzero(f,0,N)
# N1= round(N1)
N2 = N-N1

for i in 1:100 #Simulated learning loop
    p1 = N1^(1/tau)/(N1^(1/tau)+N2^(1/tau))
    p2 = 1-p1
    f(N1) = dq/(c_puct*sqrt(N)) - (p2/(N-N1)-p1/N1)
    N1 = fzero(f,0,N)
    # N1 = round(N1)
    N2 = N-N1
end
