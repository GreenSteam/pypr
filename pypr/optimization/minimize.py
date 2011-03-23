
# This code is translated from Carl Edward Rasmussens matlab minimize 
# matlab code. The goal was to make it as identical to Rasmussen's code.

# Matlab code copyright text:
# --- BEGIN
#Copyright Carl Edward Rasmussen, 2006-04-06.
#
#(C) Copyright 1999 - 2006, Carl Edward Rasmussen
#
#Permission is granted for anyone to copy, use, or modify these
#programs and accompanying documents for purposes of research or
#education, provided this copyright notice is retained, and note is
#made of any changes that have been made.
#
#These programs and documents are distributed without any warranty,
#express or implied.  As the programs were written for research
#purposes only, they have not been tested to the degree that would be
#advisable in any important application.  All use of these programs is
#entirely at the user's own risk.
# --- END

import sys
import numpy as np

def rosenbrock(x):
    """Returns the rosenbrock function value for x.
    Inputs x1, x2, .., xn are given column wise. Input vetors as rows.
    """
    r, c = np.shape(x)
    return np.c_[ np.sum(100 * pow((x[:,1:] - pow(x[:,0:c-1], 2)), 2) + \
            pow((1 - x[:,0:c-1]), 2), axis=1) ]
            
    # r(0,0) = 1
    # r(1,1) = 0
    # r(2,2) = 401

def rosenbrock_d(x):
    """Returns the values of the partial derivative of the rosenbrock function.
    Inputs x1, x2, .., xn are given column wise. Input vetors as rows.
    """
    r, c = np.shape(x)
    rd = np.zeros((r, c), dtype = x.dtype)
    rd[:,0:c-1] = -400 * x[:,0:c-1] * (x[:,1:c] - pow(x[:,0:c-1], 2)) - \
                2 * (1 - x[:,0:c-1])
    rd[:,1:c] = rd[:,1:c] + 200 * (x[:,1:c] - pow(x[:,0:c-1], 2))
    return rd

def minimize(X, f, df, length, verbose = False, return_result = False,
             callback = None):
    """This is Carl Edward Rasmussen's minimize method rewritten in Python.
    
    Inputs:
    
    X: Input for the function f. 
    
    f: The function to be minimized. f is a method that takes X as 
    input and provides af function evaluation.    
    
    df: The function gradient, evaluated at X.
    
    length: The maximum number of line searches.
    
    callback: An optional user-supplied function to call after each iteration.
              Called as callback(Xk), where Xk is the current parameter vector.
    
    Returns: By default the function value for each iteration, but if
             return_result is set to True, then it returns a tuple
             containing (function value, final function inputs).
    """
    INT = 0.1; # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0; # extrapolate maximum 3 times the current step-size
    MAX = 20;  # max 20 function evaluations per line search
    RATIO = 10.0; # maximum allowed slope ratio
    SIG = 0.1; RHO = SIG/2; # SIG and RHO are the constants controlling the
                            # Wolfe-Powell conditions.

    RUNL_LINESEARCH = 0;
    RUNL_FUNCEVAL = 1;

    if length>0:
        S = 'Line search'
    else:
        S = 'Function evaluation'
        
    i = 0; # zero the run length counter
    ls_failed = 0; # no previous line search has failed
    f0 = f(X)
    df0 = df(X)
    fX = f0
    if length<0: i = i + 1
    s = -df0
    d0 = np.dot(-s, s.T)
    x3 = 1 / (1 - d0)
    
    while i < abs(length):
        if length>0: i = i + 1
        X0 = X
        F0 = f0
        dF0 = df0
        if length>0: M = MAX
        else: M = min(MAX, -length - i)
        
        while True:
            x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
            success = False
            while (success==False) and (M>0):
                try:
                    M = M - 1
                    if (length<0): i = i + 1
                    f3 = f(X + x3 * s)
                    df3 = df(X + x3 * s)
                    # TODO: MAKE CHECKS
                    if (np.isnan(f3)) or (np.isinf(f3)) or \
                       (np.any(np.isnan(df3) + np.isinf(df3))):
                        raise Exception("isnan, isinf")
                    success = True
                except:
                    print "Sorry:", sys.exc_type, ":", sys.exc_value 
                    x3 = (x2 + x3)/2

            if (f3 < F0):
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3
            d3 = np.dot(df3, s.T)
            if (d3 > SIG * d0) or (f3 > f0+x3*RHO*d0) or (M==0):
                break
            x1 = x2; f1 = f2; d1 = d2;               # move point 2 to point 1
            x2 = x3; f2 = f3; d2 = d3;               # move point 3 to point 2
            A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);         # make cubic extrapolation
            B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
            x3 = x1-d1*(x2-x1)**2/(B+np.sqrt(B*B-A*d1*(x2-x1)));  # num. error possible, ok!

            if not(np.isreal(x3)) or np.isnan(x3) or np.isinf(x3) or x3 < 0:
                x3 = x2*EXT;
            elif x3 == x2*EXT:
                x3 = x2*EXT
            elif x3 < x2+INT*(x2-x1):
                x3 == x2+INT*(x2-x1);
                
        while ((abs(d3) > -SIG*d0) or (f3 > f0+x3*RHO*d0)) and (M>0):
            if (d3 > 0) or (f3 > f0+x3*RHO*d0):            # choose subinterval
                x4 = x3; f4 = f3; d4 = d3                # move point 3 to point 4
            else:
                x2 = x3; f2 = f3; d2 = d3                # move point 3 to point 2
            if f4 > f0:           
                x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2));  # quadratic interpolation
            else:
                A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);              # cubic interpolation
                B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
                x3 = x2+(np.sqrt(B*B-A*d2*(x4-x2)**2)-B)/A;        # num. error possible, ok!
            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2+x4)/2;               # if we had a numerical problem then bisect
            x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  # don't accept too close
            f3 = f(X + x3 * s)
            df3 = df(X + x3 * s)
            if f3 < F0: 
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3                                      #keep best values
            M = M - 1
            if (length<0): i = i + 1                            # count epochs?!
            d3 = np.dot(df3, s.T)

        if (abs(d3) < -SIG*d0) and (f3 < f0+x3*RHO*d0):  # if line search succeeded
            X = X + x3 * s                      # update variables
            f0 = f3; 
            fX = np.concatenate((fX, f0), axis=0);
            if callback!=None:
                callback(X)
            if (verbose):
                print S, " ", i, " Value: ", f0
            s = (np.dot(df3, df3.T) - np.dot(df0,df3.T)) / \
                np.dot(df0,df0.T)*s - df3;   # Polack-Ribiere CG direction
            df0 = df3;                                  # swap derivatives
            d3 = d0; 
            d0 = np.dot(df0, s.T)
            if d0 > 0:                            # new slope must be negative
                s = -df0;
                d0 = np.dot(-s, s.T)         # otherwise use steepest direction
            realmin = realmin = np.finfo(np.double).tiny
            x3 = x3 * min(RATIO, d3/(d0-realmin)) # slope ratio but max RATIO
            ls_failed = 0                         # this line search did not fail
        else:
            X = X0; f0 = F0; df0 = dF0;       # restore best point so far
            if (ls_failed==1) or (i > abs(length)):   # line search failed twice in a row
                break;                        # or we ran out of time, so we give up
            s = -df0;
            d0 = np.dot(-s, s.T)                 # try steepest
            x3 = 1/(1-d0)                     
            ls_failed = 1                        # this line search failed
    if return_result:
        return (fX, X)
    else:
        return fX
