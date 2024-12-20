import math

import numpy as np
import scipy.fftpack


def nsdual(*args):
# %NSDUAL  Canonical dual NSG frame (for painless systems)
# %   Usage: gd = nsdual(g,shift,M)
# %
# %   Input parameters:
# %         g         : Cell array of window functions/filters
# %         shift     : Vector of time/frequency shifts
# %         M         : Number of frequency channels (vector/scalar)
# %   Output parameters:
# %         gd        : Dual window functions 
# %
# %   Given a non-stationary Gabor frame specified by the windows g, shift 
# %   parameters shift, and channel numbers M, NSDUAL computes the
# %   canonical dual frame windows/filters gd by inverting the diagonal of 
# %   the frame operator and applying the inverse to g. More explicitly,
# %
# %      gd{n} = g{n} / ( sum M(l) |g{l}|^2 ), 
# %                        l  
# %
# %   If g, shift, M specify a painless frame, i.e. 
# %   SUPP(G{N})  <= M(n)~forall~n and 
# %
# %      A <= sum ( M(n) |g{n}|^2 ) <= B, for some 0 < A <= B < infty
# %            n  
# %
# %   the computation will result in the canonical dual frame. If  g, 
# %   shift, M specify a frame, but the first condition is violated, the 
# %   result can be interpreted as a first approximation of the corresponding 
# %   canonical dual frame.
# % 
# %   Note, the time shifts corresponding to the dual window sequence is the
# %   same as the original shift sequence and as such already given.
# %
# %   If g, shift, M is a painless frame, the output can be used for 
# %   perfect reconstruction of a signal using the inverse nonstationary 
# %   Gabor transform NSIGT.
# % 
# %   See also:  nsgt, nsigt, nsgt_real, nsigt_real, nsgtf, nsigtf
# % 
# %   References:
# %     P. Balazs, M. Dörfler, F. Jaillet, N. Holighaus, and G. A. Velasco.
# %     Theory, implementation and applications of nonstationary Gabor Frames.
# %     J. Comput. Appl. Math., 236(6):1481-1496, 2011.
# %     
# %
# %   Url: http://nsg.sourceforge.net/doc/core_routines/nsdual.php

# % Copyright (C) 2013 Nicki Holighaus.
# % This file is part of NSGToolbox version 0.1.0
# % 
# % This work is licensed under the Creative Commons 
# % Attribution-NonCommercial-ShareAlike 3.0 Unported 
# % License. To view a copy of this license, visit 
# % http://creativecommons.org/licenses/by-nc-sa/3.0/ 
# % or send a letter to 
# % Creative Commons, 444 Castro Street, Suite 900, 
# % Mountain View, California, 94041, USA.

# % Author: Nicki Holighaus, Gino Velasco
# % Date: 23.04.13

# Check input arguments
nargin = len(args)
assert nargin < 2, 'Not enough input arguments'

g = args[0]     # Cell array of window functions/filters 
shift = args[1] # Vector of time/frequency shifts

if nargin < 3:
    for kk in range(len(shift)):
        M[kk] = len(g[kk])
        M = M.T
else:
    M = args[2]  # Number of frequency channels (vector/scalar)

if max(M.shape) == 1:
    M = M[0] * np.ones((len(shift),1))

# Setup the necessary parameters
N = len(shift)

posit = np.cumsum(shift)
Ls = posit[N-1]
posit = posit-shift[0]

diagonal = np.zeros((Ls,1))
win_range = np.empty((N,1),dtype=object)  # Create cell array 

# Construct the diagonal of the frame operator matrix explicitly

for ii in range(N):
    Lg = len(g[ii])
    win_range[ii] = (posit[ii]+[-math.floor(Lg/2):math.ceil(Lg/2)]) % Ls+1
    np.diag(win_range[ii]) = np.diag(win_range[ii]) + (np.fft.fftshift(g[ii])**2)*M[ii]
end

# Using the frame operator and the original window sequence, compute
# the dual window sequence

gd = g

for ii in range(N):
    gd[ii] = np.fft.ifftshift(np.fft.fftshift(gd[ii]) / np.diag(win_range[ii]))

return gd # Dual window functions 
