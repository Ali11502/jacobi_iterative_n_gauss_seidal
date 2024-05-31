import numpy as np
from numpy.linalg import *
import time
def gauss_seidel(A, b,x, tolerance, max_iterations):
    #x is the initial condition
    iter1 = 0
    #Iterate
    for k in range(max_iterations):
        iter1 = iter1 + 1
        x_old  = x.copy()
        
        #Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]
            
        #Stop condition 
        #LnormInf corresponds to the absolute value of the greatest element of the vector.
        
        LnormInf = max(abs((x - x_old)))/max(abs(x_old))   
        if  LnormInf < tolerance:
            break
    
           
    return x,iter1
def jacobi(A, b, x0, tol, maxiter):
   
    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    k = 0
    rel_diff = tol * 2

    while (rel_diff > tol) and (k < maxiter):

        for i in range(0, n):
            subs = 0.0
            for j in range(0, n):
                if i != j:
                    subs += A[i,j] * x_prev[j]

            x[i] = (b[i] - subs ) / A[i,i]
        k += 1

        rel_diff = norm(x - x_prev) / norm(x)
      
        x_prev = x.copy()

    return x,  k






n=400
x0 = np.zeros(n);
b = np.zeros(n)
for i in range(0,n):
    b[i]=np.random.randint(n)
tol = 10E-20
maxiter = 2000
A = np.random.randint(10, size=(n,n)) + np.eye(n,n)*(n*n)

start = time.time()
x,  k = jacobi(A, b, x0, tol, maxiter)
totaltime = time.time() - start

start2=time.time()
y,iterations=gauss_seidel(A,b,x0,tol,maxiter)
totaltime2=time.time()-start2

if iterations==maxiter:
    print(('WARNING: the Gauss-Seidal iterations did not '
            'converge within the required tolerance.'))
if k == maxiter:
 print(('WARNING: the Jacobi iterations did not '
            'converge within the required tolerance.'))
print("----Array-----")
for i in A:
     print(i)

print("-----Solution of Jacobi------")
for i in x:
    print(round (i,4))
print(f"iterations by Jacobi= {k}")
print("time taken by Jacobi: ",totaltime)

print("-----Solution of Gauss-Seidal------")
for i in y:
    print(round (i,4))
print(f"iterations by Gauss-Seidal= {iterations}")
print("time taken by Gauss-Seidal: ",totaltime2)

