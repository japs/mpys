#!/usr/bin/python3
# *********************************************************************        
# * Copyright (C) 2014 Jacopo Nespolo <j.nespolo@gmail.com>           *        
# *                                                                   *
# * For the license terms see the file LICENCE, distributed           *
# * along with this software.                                         *
# *********************************************************************
#
# This file is part of mpys.
# 
# mpys is free software: you can redistribute it and/or modify it under the 
# terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# mpys is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along 
# with mpys.  If not, see <http://www.gnu.org/licenses/>
#
# It goes without saying that scientific/academic ethics should apply to 
# this piece of software. If you use this code to produce a publication,
# it would be nice that you cite the author and provide a link to the web 
# page where you fuond the code. 
#

import numpy as np
import scipy.linalg as la

__version__ = "0.1.0"

class MPState:
    '''
    Comprehensive MPS class.
    '''

    def __init__(self, d, L=1, cutoff=None):
        '''
        Constructor

        Parameters
        ----------
        d: Single site Hilbert space dimension
        L: Chain length
        cutoff: Limit matrix dimensions to cutoff.

        Members
        -------
        mps: The list containing the actual matrices.
        left_normalised, right_normalised: Site up to which the MPS is 
            left (right) normalised. 
            left_normalised = 0 means no left-normalisation;
            right_normalised = L+1 means no right_normalisation.
        S: Singular values
        '''
        self.d = d
        self.L = L
        self.cutoff = cutoff
        self.mps = [None]
        self.left_normalised = 0
        self.right_normalised = 0
        self.S = None

    def random(self):
        '''
        Create random MPS.
        '''
        self.mps = [None]
        rows = 1
        for site in range(1, self.L + 1):
            if site < self.L - site:
                cols = min(self.d**(site), self.cutoff)
            else:
                s = self.L - site
                cols = min(self.d**(s), self.cutoff)
            self.mps.append(np.random.uniform(0, 1, size=(rows, 
                                                          self.d, 
                                                          cols)))
            rows = cols

    def normalise_site_svd(self, site, left=True):
        '''
        Normalise the MP at a given site. Use the SVD method,
        arXiv:1008.3477, sec. 4.1.3
        '''
        if (left and site == L) or (not left and site == 1):
            self.normalise_extremal_site(site, left=left)
            return
        
        M_asb = self.mps[site] # the matrix we have to work on, M_site
        r, s, c = M_asb.shape  # row, spin, column

        if left:
            M_aab = M_asb.reshape((r * s, c)) # merge first two indices

            U, S, Vh = la.svd(M_aab, full_matrices=False)
            self.mps[site] = U.reshape((r, s, U.size // (r * s))) # new A_site

            # Contract over index b: (S @ Vh)_ab (X) (M_site+1)_bsc
            SVh = np.dot(S, Vh)
            self.mps[site+1] = np.tensordot(SVh, self.mps[site+1], ((1),(0)))
        else:
            M_aab = M_asb.reshape((r, s * c)) # merge last two indices

            U, S, Vh = la.svd(M_aab, full_matrices=False)
            self.mps[site] = Vh.reshape((Vh.size // (s * c), s, c)) # new B
            # Contract over index b: (M_site-1)_asb (X) (U @ S)_bc 
            US = np.dot(U, S)
            self.mps[site-1] = np.tensordot(self.mps[site-1], US, ((2),(0)))
        return
    #enddef

    def normalise_site_qr(self, site, left=True):
        '''
        Normalise the MP at a given site. Use the QR method,
        arXiv:1008.3477, sec. 4.1.3
        '''
        if (left and site == L) or (not left and site == 1):
            self.normalise_extremal_site(site, left=left)
            return
        
        M_asb = self.mps[site] # the matrix we have to work on, M_site
        r, s, c = M_asb.shape  # row, spin, column

        if left:
            M_aab = M_asb.reshape((r * s, c)) # merge first two indices
            

            Q, R = la.qr(M_aab, mode='economic') # economic = thin QR in
                                                     # 1008.3477
            self.mps[site] = Q.reshape((r, s, Q.size // (r * s))) # new A_site

            # Contract R with matrix at site + 1
            self.mps[site+1] = np.tensordot(R, self.mps[site+1], ((1),(0)))
        else:
            M_aab = M_asb.reshape((r, s * c)) # merge last two indices

            # QR = M^+ ==> M = R^+ Q^+
            Q, R = la.qr(M_aab.transpose())
            Q = Q.transpose()
            R = R.transpose()
            self.mps[site] = Q.reshape((Q.size // (s * c), s, c)) # new B_site

            # Contract R with matrix at site - 1
            self.mps[site-1] = np.tensordot(self.mps[site-1], R, ((2),(0)))
        return
    #enddef

    def normalise_extremal_site(self, site, left):
        '''
        Normalise the first or last matrix in the MPS.
        '''
        M  = self.mps[site] # just an alias
        Mh = M.transpose((2, 1, 0)) # dagger on matrix indices,
                                    # spin index remains untouched

        # compute norm: fully contract - matrix indices firts, then spin
        if left:
            norm = np.tensordot(Mh, M, ((2, 0, 1),(0, 2, 1)))
        else:
            norm = np.tensordot(M, Mh, ((2, 0, 1),(0, 2, 1)))
        
        self.mps[site] /= np.sqrt(norm)


    def canonicalise(self, left=True, svd=True):
        '''
        Run through sites to put the MPS in left or right canonical form.
        '''
        if left:
            start, stop, step = 1, L+1, 1
            self.left_normalised = L
            self.right_normalised = L + 1
        else:
            start, stop, step = L, 0, -1
            self.left_normalised = 0
            self.right_normalised = 1

        for site in range(start, stop, step):
            if svd:
                self.normalise_site_svd(site, left=left)
            else: # QR
                self.normalise_site_qr(site, left=left)

    def mixed_canonicalise(site):
        '''
        Put the MPS in mixed canonical form.

        Sites 1...site are left-normalised, sites site...L are 
        right-normalised. The singular values are stored in self.S and 
        returned.
        '''
        if site > self.left_normalised:
            # gotta left-normalise sites left_normalised...site
            # left qr up to site-1
            for _s in range(self.left_normalised+1, site):
                self.normalise_site_qr(_s, left=True)
            # svd at site
            Msite = self.mps[site]
            r, s, c = Msite.shape
            Msite = Msite.reshape((r * s, c))
            U, self.S, Vh = la.svd(Msite, full_matrices=False)
            self.mps[site] = U.reshape((r, s, U.size // (r * s))) # new A_site
            self.mps[site+1] = np.tensordot(Vh, self.mps[site+1], ((1),(0))) 
        else:
            # gotta right-normalise sites right_normalised...site+1
            # right qr up to site+2
            for _s in range(self.right_normalised, site+1):
                self.normalise_site_qr(_s, left=False)
            # svd at site+1
            Msite = self.mps[site+1]
            r, s, c = Msite.shape
            Msite = Msite.reshape((r, s * c))
            U, self.S, Vh = la.svd(Msite, full_matrices=False)
            self.mps[site+1] = Vh.reshape((Vh // (s * c), s,c)) #new B
            self.mps[site] = np.tensordot(self.mps[site], U, ((1),(0))) 

        return self.S

if __name__ == "__main__":
    mps = MPState(2, 64, 50)

    mps.random()
