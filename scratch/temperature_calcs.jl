#Calculates theoretical and practical visit counts of an example where two differetna ctions are possible and there is a dq difference in expected q-value between them.

using Roots
dq = .01 #Normalized difference in q-values
c_puct = 5
N = 2000
tau = 1.5

if tau > 1.0 #Formula only valid for tau>1.0. For tau<1.0, N1_theoretical=N, N2_theoretical=0.
    f(N1) = dq/(c_puct*sqrt(N)) - ((N-N1)^(1/tau-1) - N1^(1/tau-1)) / ((N-N1)^(1/tau)+N1^(1/tau))
    N1_theoretical = fzero(f,0,N)
    N2_theoretical = N-N1_theoretical
else
    N1_theoretical = N
    N2_theoretical = 0
end

##
p1 = 0.5
p2 = 1-p1

f(N1) = dq/(c_puct*sqrt(N)) - (p2/(N-N1)-p1/N1)
N1 = fzero(f,0,N)
# N1= round(N1)
N2 = N-N1

##
for i in 1:100
    p1 = N1^(1/tau)/(N1^(1/tau)+N2^(1/tau))
    p2 = 1-p1
    f(N1) = dq/(c_puct*sqrt(N)) - (p2/(N-N1)-p1/N1)
    N1 = fzero(f,0,N)
    # N1 = round(N1)
    N2 = N-N1
end
