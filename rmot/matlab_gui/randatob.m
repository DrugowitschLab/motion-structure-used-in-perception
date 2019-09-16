% generates a column vector of n random integers from the interval [a,b]
function r = randatob(a,b,n)
r = ceil(a + (b-a+1).*rand(n,1) - 1);