# -*- coding: utf-8 -*-
"""
Ömer Oruç
Created on May 2024
python codes of problem 1 
generates results of Table 2 and 3, fro Rr = 3.5
"""

import matplotlib.pyplot  as plt
import numpy as np
from numba import prange, jit
import math as mt
from matplotlib import path
from scipy.stats import qmc
import scipy.io
       
def poly(x,y,degree):
    # polynomial augmentation
    o = np.zeros_like(x); I = np.ones_like(x)
    P =   np.column_stack([I, x, y, x**2, x*y, y**2, x**3,   x**2*y, x*y**2, y**3])
    Dxx = np.column_stack([o, o, o, 2*I,  o,   o,    6*x,    2*y,    o,      o])
    Dyy = np.column_stack([o, o, o, o,    o,   2*I,  o,      o,      2*x,    6*y]) 
    Dxy =  np.column_stack([o, o, o, o,  I,   o,    o, 2*x,  y*2,   o])
    return P,Dxx,Dyy,Dxy 

@jit(nopython=True,parallel=True)  
def kernel(points,centers,c):
    xp = points[:,0];  yp = points[:,1]; 
    xc = centers[:,0]; yc = centers[:,1]; 
    n=len(xp);  m=len(xc);
    M,Dxx,Dyy,Dxy = np.zeros((n,m)),np.zeros((n,m)),np.zeros((n,m)),np.zeros((n,m))
    coef1 = 3*c**4; coef2 = c**2;
    for i in prange(n):
        for j in range(m):
            x2 = (xp[i]-xc[j])**2; y2=(yp[i]-yc[j])**2; xx = (xp[i]-xc[j]); yy = (yp[i]-yc[j]);
            M[i,j] = 1/np.sqrt((x2+y2)*coef2+1);
            kup = M[i,j]**3; bes = M[i,j]**5
            Dxx[i,j] = coef1*x2*bes-coef2*kup
            Dyy[i,j] = coef1*y2*bes-coef2*kup
            Dxy[i,j] = (coef1*xx*yy)*bes
                        
    return M,Dxx,Dyy,Dxy


def radius(k,n,b):
    # adopted from the article https://doi.org/10.1016/j.enganabound.2021.05.006
    if k>n-b:
        r = 1;
    else:
        r = np.sqrt(k-1/2)/np.sqrt(n-(b+1)/2);
        
    return r

def fabric_pattern(n, alpha):
    # adopted from the article  https://doi.org/10.1016/j.enganabound.2021.05.006
    b = round(alpha*np.sqrt(n));
    phi = (np.sqrt(5)+1)/2;
    x=np.zeros((n,)); y=np.zeros((n,));
    for k in range (1,n+1):
        r = radius(k,n,b);
        theta = 2*np.pi*k/phi**2;
        x[k-1]=r*np.sin(theta); 
        y[k-1]=r*np.cos(theta);
    return x,y

def modified_Franke(n,D):
    return 0.8*n**0.25/D 


from scipy.linalg import svd

def solve_svd(A,b):
    U,s,Vh = svd(A) 
    c = np.dot(U.T,b)
    w = np.dot(np.diag(1/s),c)
    x = np.dot(Vh.T,w)
    return x

#exact soln
exact = lambda X, Y: np.sin(Y/np.sqrt(2.0))*np.cosh(2.0*np.sqrt(30.0)/15.0*X-np.sqrt(30.0)/10.0*Y) 

def main(Nx): 
    H11, H12, H21, H22 = 3.0, 1.5, 1.5, 2.0
    Ny = Nx; xl,xr,yl,yr = 0,1,0,1; xs,ys = np.linspace(xl,xr,Nx),np.linspace(yl,yr,Ny);
    x,y = np.meshgrid(xs,ys); x,y = x.transpose(),y.transpose();
    pts=np.column_stack((np.ravel(x),np.ravel(y)))   
    ip = np.squeeze( np.where((pts[:,0]>xl) & (pts[:,0]<xr) & (pts[:,1]>yl) & (pts[:,1]<yr)))
    bp = np.squeeze( np.where((pts[:,0]==xl) | (pts[:,0]==xr) | (pts[:,1]==yl) | (pts[:,1]==yr)))
    bpts = np.squeeze(pts[bp,:])
    ipts = np.squeeze(pts[ip,:])
    N = len(pts[:,0])
    [xc, yc] = fabric_pattern(N,1);    
    Rr = 3.5;  deg = 3; q = int((deg+1)*(deg+2)/2);
    epts = np.vstack([ipts,bpts]); ua = exact(epts[:,0], epts[:,1]) 
    F = 0*ipts[:,0]; bc = exact(bpts[:,0], bpts[:,1]);   
    rhs = np.hstack([F,bc]);c = modified_Franke(N, 2*Rr); centers = np.vstack([Rr*xc, Rr*yc]).T;
    uD,_,_,_ = kernel(bpts,centers,c) 
    _,Dxx,Dyy,Dxy = kernel(ipts,centers,c)    
    Pb,Pxxb,Pyyb,Pxyb = poly(bpts[:,0],bpts[:,1],deg)
    Pi,Pxxi,Pyyi,Pxyi = poly(ipts[:,0],ipts[:,1],deg)
    
    A11 = H11*Dxx+H22*Dyy+2*H12*Dxy; A12 = H11*Pxxi+H22*Pyyi+2*H12*Pxyi;
    A21 = uD; A22 = Pb; A31 = np.vstack([A12,A22]).T; A32 = np.zeros((q,q));
    solaug = np.vstack([np.hstack([A11,A12]),np.hstack([A21,A22]),np.hstack([A31,A32])]); 
    rhsaug = np.hstack([rhs,np.zeros((q,))]);    
    U,_,_,_ = kernel(epts,centers,c)
    
    lmdaug = solve_svd(solaug,rhsaug)
    P = np.vstack([Pi,Pb])
    uaug = np.hstack([U,P])@lmdaug
    diff2 = ua - uaug;
    Linf2 = np.linalg.norm(diff2,np.inf);
    L2 = np.linalg.norm(diff2,None);
    RMS = L2/np.sqrt(N)
    print('Number of nodes = %d, Linf =%.4e, L2 =%.4e,  RMS=%.4e\n' %(N,Linf2, L2,RMS));
    
    
N = [11,21,31,41,51]
for i in N:    
    main(i)
