#################################################################################
#
#   File:       ImageScienceToolBox.py
#
#   Summary:    A collection of functions/methods that can be used to perform image
#               analysis.  DFT, filters, convolutions, covariance analysis and etc
#
#
#   Author:     Robert Upton (Sigma Space Corporation)
#
#   Date:       Feb 17, 2012 - April 12, 2012
#               June 29, 2012 - added structure function and encircled energy
#
#
#################################################################################

import numpy as num 
import pylab as py
from scipy.optimize import optimize
import pdb

class ImageScienceClass():
    '''This instance of ImageScienceClass requires something like: self = ImageScienceClass()
       then the attributes of ImageScienceClass get assigned to self, then those attributes 
       are accessed by self.padFactor(self.wfmap,2) for example
    '''

    def __init__(self):
        # instantiation
        self.one = 1    
    
#----------------------------------------------------------------------------------------------------#
    def readWFdata(self, fname):
        '''read in data from filename
        '''
        txtName  = '/Users/rupton/Documents/DDEProject/data/' + fname
        
        fid      = open(txtName,"r")              # get the wavefront data
        data     = fid.readlines()
        fid.close()
        
        # make data into an array
        wfmap = num.zeros((len(data[16:]),len(data[16:])))
        ii    = 0
        for lines in data[16:]:
            test = lines.split()
            for jj in num.arange(len(wfmap)-1):
                wfmap[ii,jj] = test[jj]
            ii = ii + 1
        
        # plot the data
        py.ion()
        py.figure()
        py.imshow(wfmap)
        py.colorbar()
        py.axis('off')
        
        self.wfmap = wfmap

        return self

#----------------------------------------------------------------------------------------------------#
    def WHfilter(self,data,fg,SNR):
        '''The Wiener-Helstrom filter to regularize 
           signals gf using fg and data.  Assume 
           a linear-shift invraint convolution 
           between fg, data, and gf
        '''    
    
        Cu  = num.fft.fft2(data)
        FGu = num.fft.fft2(fg) 
        GFu = Cu*num.conj(FGu)/((FGu*num.conj(FGu)).real+1/SNR)
    
        gf = num.fft.ifft2(GFu)
        gf = num.fft.ifftshift(gf)
    
        self.gf = gf
        return self
    
#----------------------------------------------------------------------------------------------------#
    def padFactor(self,wfmap,padFac):
        '''data contains the matrix to be FFT to generate the MTF.  There is some zero-padding
            introduced inorder to adequately scale the size of the FFT/MTF.
            introduce the data into args.  
            If you want to pad a mask: args[0] = WFmap, args[1] = padFactor
            If you want to pad WFdata: args[0] = dummyVal, args[1] = padFactor, args[2] = WFmap
            '''
# introduce zero-padding and make a binary mask
        data      = num.array(wfmap)                                  # convert the data to an array
        sVec      = num.shape(data)                                   # get the size of the data
        NsampRows = sVec[0]*(padFac - 1)                              # set up the numbers of rows and columns
        NsampCols = sVec[1]*(padFac - 1)
        zBuffer   = num.zeros((sVec[0],NsampCols/2))                  # set up the 0-padding for the column order
        dataCond  = num.concatenate((data,zBuffer),axis=1)            # concatenate the data
        dataCond  = num.concatenate((zBuffer,dataCond),axis=1)
    
        zBuffer   = num.zeros((NsampRows/2,sVec[0]*padFac))           # set up the 0-padding for the row order
        dataCond  = num.concatenate((zBuffer,dataCond),axis=0)        # concatenate the data
        dataCond  = num.concatenate((dataCond,zBuffer),axis=0)
    
# make the mask from the wavefront map using (wfmap/wfmap+1e-16)/max(wfmap)
        mask = dataCond+1e-016
        mask = dataCond/mask
        mask = mask/num.max(mask)
        self.mask = mask
        self.data = dataCond
        
        return self

#----------------------------------------------------------------------------------------------------#
    def makeMask(self,nsamp,padFac,xpos,ypos):
        '''make circular mask with xpos, ypos offset
        '''
    
        x       = num.matrix(num.arange(-1,1,2.0/nsamp))                # form a row vector 
        y       = num.transpose(x)
        unitVec = num.matrix(num.ones((nsamp,1)))                       # form a row vector of ones
        XX      = unitVec*x
        YY      = y*num.transpose(unitVec)
        XX      = XX - xpos
        YY      = YY - ypos
        RR      = num.sqrt(num.multiply(XX,XX)+num.multiply(YY,YY)) 
        RR      = (RR==0).choose(RR,1e-011)

        mask    = padFac*RR
        RR      = padFac*RR
    
        mask    = (mask<=1).choose(mask,1)                              # select vals <=1 and assign to 1 
        mask    = (mask>1).choose(mask,0)   
        
        self.XX = XX
        self.YY = YY
        self.mask = mask

        return self

#----------------------------------------------------------------------------------------------------#
    def propFilter(self,padFac,nsamp,zVec,dZ):
        '''
        propagation filter
        '''

        # form the pupil and wavefront map
        self.pol2cartMat(nsamp,padFac)
        self.waveCreate(zVec) 
        wfmap = self.waveFunc
        mask  = self.maskPol
        
        # calculate the field
        phaseStart = num.multiply(wfmap,2*num.pi*1j)
        fieStart   = num.multiply(num.exp(phaseStart),mask)

        # calculate spectrum
        FIEstart   = num.fft.fft2(fieStart)
        FIEstart   = num.fft.fftshift(FIEstart)

        # propagation kernel
        xi      = num.matrix(num.arange(-padFac,padFac,2.0*padFac/nsamp))      # form a row vector 
        eta     = num.transpose(xi)
        unitVec = num.matrix(num.ones((nsamp,1)))                       # form a row vector of ones
        XI      = unitVec*xi
        ETA     = eta*num.transpose(unitVec)
        RXE     = num.sqrt(num.power(XI,2)+num.power(ETA,2))
        RXE     = num.multiply(RXE,mask)
        KERNEL  = num.sqrt(1-num.power(RXE,2))*dZ
        
        KERNEL  = num.multiply(2*num.pi*1j,KERNEL)
        KERNEL  = num.exp(KERNEL)                                       # introduce the mask
        KERNEL  = num.multiply(mask,KERNEL)

        # propagated spectrum
        FIEprop = num.multiply(FIEstart,KERNEL)
        # propagated field
        fieComplex = num.fft.ifft2(FIEprop)

        self.fieComplexProp = fieComplex

        return fieComplex


#----------------------------------------------------------------------------------------------------#
    def psfCoherent(self, mask, *args):
        '''
        Define the coherent superposition of PSFs in field space.  The tuple *args
        contains the wavefunctions of interest
        '''
        # calculate the psfs
        sizeVec = num.shape(mask)
        fieTOT  = num.zeros((sizeVec[0],sizeVec[1]))

        #pdb.set_trace()
        # the args are the waveFunctions
        kk = 0
        for ii in args:
            opd    = ii
            phase  = num.multiply((1j),(opd))
            phase  = num.multiply((2*num.pi),(phase))
            if kk == 0:
                fie = 1.0*num.exp(phase)
            else:
                fie = num.exp(phase)

            fieTOT = fieTOT + num.multiply((mask),(fie))
            kk = kk + 1

        psf0         = num.fft.ifft2(mask)
        psf0         = num.fft.ifftshift(psf0)
        psf0         = (num.conj(psf0)*psf0).real
        
        psf          = num.fft.ifft2(fieTOT)
        psf          = num.fft.ifftshift(psf)
        psf          = (num.conj(psf)*psf).real
        psf          = psf/num.max(psf0)           

        self.psfCOH  = psf

        return self

#----------------------------------------------------------------------------------------------------#
    def psfOTF(self,mask,dataPupil):
        '''calculate the PSF and OTF of the data included
            psf, OTF = psfOTF(mask, dataComplexPupil)
            It really helps to put in the mask as part of the psf calculation because
            the phase of a complex number with a small amplitude is wierd
        '''
# form the complex pupil data
        pi = num.pi
        complexPupil = num.multiply((1j),(dataPupil))
        complexPupil = num.multiply((complexPupil),(2*pi))
        complexPupil = num.exp(complexPupil)
        complexPupil = num.multiply((mask),(complexPupil))
        
        psf0         = num.fft.ifft2(mask)
        psf0         = num.fft.ifftshift(psf0)
        psf0         = (num.conj(psf0)*psf0).real
        
        psf          = num.fft.ifft2(complexPupil)
        psf          = num.fft.ifftshift(psf)
        psf          = (num.conj(psf)*psf).real
        psf          = psf/num.max(psf0)                               # normalize the peak, in the conjugate
                                                                       # domain the area is normalized by the 
        OTF          = num.fft.fft2(psf)                               # central ordinate theorem
        OTF          = num.fft.fftshift(OTF)
        OTF          = num.sqrt((num.conj(OTF)*OTF).real)
        
        self.OTF = OTF
        self.psf = psf
        self.complexPupil = complexPupil
        
        return self

#----------------------------------------------------------------------------------------------------#
    def pol2cartMat(self,nsamp,padFac):  
        '''define the RR and THETA coordinates in spherical polar 
           coordinates.  The mask is also generated.  nsamp is 
           the number of sample points on a side, and padFac is the
           amount of padding introduced into the pupil.  This works
           differently to padFactor function
        '''
# generate linear vectors and meshgrids 
        x       = num.matrix(num.arange(-1,1,2.0/nsamp))                # form a row vector 
        y       = num.transpose(x)
        unitVec = num.matrix(num.ones((nsamp,1)))                       # form a row vector of ones
        XX      = unitVec*x
        YY      = y*num.transpose(unitVec)
    
# form a rotationally symmetric mask #
        RR      = num.sqrt(num.multiply(XX,XX) + num.multiply(YY,YY))
        RR      = (RR==0).choose(RR,1e-011)
        THET    = num.arccos(num.multiply(XX,1/RR))
        pi      = num.arccos(-1)
# condition THETA 
        uniMat  = num.matrix(num.ones((nsamp,nsamp)))
        THETtop = num.fliplr(THET[0:nsamp/2,:])-pi*uniMat[0:nsamp/2,:]
        THETbot = THET[nsamp/2:nsamp,:]
        THET    = num.concatenate((THETtop,THETbot), axis=0)
        mask    = padFac*RR
        RR      = padFac*RR

        mask    = (mask<=1).choose(mask,1)                              # select vals <=1 and assign to 1 
        mask    = (mask>1).choose(mask,0)                               # select vals > 1 and assign to 0
    
        self.maskPol = mask
        self.XX      = XX
        self.YY      = YY
        self.RR      = RR
        self.TH      = THET
        
        return self
#----------------------------------------------------------------------------------------------------#
    def cart2Phase(self,cMat):  
        '''input the matrix or vector cMat that contains complex numbers 
           and get the amplitude and phase back
        '''
# generate linear vectors and meshgrids 
        
        cMat = num.matrix(cMat)
        nrows, ncols = num.shape(cMat) 
        XX = cMat.real
        YY = cMat.imag
    
# form a rotationally symmetric mask #
        RR      = num.sqrt(num.multiply(XX,XX) + num.multiply(YY,YY))
        RR      = (RR==0).choose(RR,1e-021)
        THET    = num.arccos(num.multiply(XX,1/RR))
        pi      = num.arccos(-1)
# condition THETA 
        uniMat  = num.matrix(num.ones((nrows,ncols)))
        THETtop = num.fliplr(THET[0:nrows/2,:])-pi*uniMat[0:nrows/2,:]
        THETbot = THET[nrows/2:ncols,:]
        THET    = num.concatenate((THETtop,THETbot), axis=0)
    
        self.phase = THET
        self.amp   = RR
    
        return self

#----------------------------------------------------------------------------------------------------#
    def ZernikeWaveFunc(self, RR, TH, mask, nterm):
        '''Function that generates the standard RMS weighted Zernike
           polynomial wavefunctions
        '''
        
        cosTH     = num.cos(TH)
        sinTH     = num.cos(TH)
        zernFunc  = num.zeros((len(RR),len(RR)))
        factorial = self.factorial
        Afunc     = num.zeros((len(RR),len(RR)))
        Rfunc     = num.zeros((len(RR),len(RR)))                      
# set up azimuthal index
        nOrder = num.ceil((-3+num.sqrt(9+8*nterm))/2)
        mOrder = 2*nterm-nOrder*(nOrder+2)
        if nOrder <> 0:
            if mOrder < 0:
                Afunc = num.sin(num.abs(mOrder)*TH)                                      # odd azimuthal function 
            if mOrder > 0:
                Afunc = num.cos(num.abs(mOrder)*TH)                                      # even azimuthal function
            if mOrder == 0:                                                         
                Afunc = num.ones((len(RR),len(RR)))                                      # no azimuthal function   
            nSum = num.linspace(0, (nOrder-abs(mOrder))/2, (nOrder-abs(mOrder))/2+1)     # form the radial index
            #pdb.set_trace()
            for ss in nSum:
                factNum  = factorial(nOrder-ss)
                factDen1 = factorial((nOrder+num.abs(mOrder))/2-ss)
                factDen2 = factorial((nOrder-num.abs(mOrder))/2-ss)
                factDen3 = factorial(ss)
                if (nOrder - 2*ss) ==  0:
                    Rpower = num.ones((len(RR),len(RR)))
                else:
                    Rpower  = num.power(RR,(nOrder-2*ss))
                Rupdate = (-1)**ss*factNum/(factDen3*factDen1*factDen2)*Rpower
                if mOrder == 0:
                    normVal = num.sqrt(nOrder+1)
                elif mOrder <> 0:
                    normVal = num.sqrt(2*(nOrder+1))      
                Rfunc   = Rfunc + normVal*Rupdate
                zernFunc = num.multiply((Rfunc),(Afunc))   
                    
        elif nOrder == 0:
            zernFunc = num.ones((len(RR),len(RR)))
                                
        zernFunc = num.multiply((zernFunc),(mask))
        self.zernFunc = zernFunc
            
        return self
#----------------------------------------------------------------------------------------------------#
    def waveCreate(self,zVec): 
        '''calculate the wavefront from the RMS weighted Zernike vector zVec.  Need to 
           run pol2Cart to generate Zernike wavefront function 
           zVec has to be a list and not a numpy array 
        '''
    
# assign matrices from cart2pol calculation
        maskPol = self.maskPol
        RR  = self.RR
        TH  = self.TH
        uni = num.ones((len(maskPol),len(maskPol)))   
        waveErr = num.zeros((len(maskPol),len(maskPol)))
        waveErr = num.matrix(waveErr)
    
# generate the Zernike wavefront function  
        indx = 0
        for ii in zVec:
            self.ZernikeWaveFunc(RR, TH, maskPol, indx)
            waveErr = waveErr + ii*self.zernFunc
            indx = indx + 1
        
        waveFunc = num.multiply((waveErr),(maskPol))
        self.waveFunc = waveFunc
    
        return self

#----------------------------------------------------------------------------------------------------#
    def encEnergy(self, funcEval, radOffSet):
        '''this function generates an encirled energy
        '''
        funcEval = num.matrix(funcEval)
        funcEval = funcEval + abs(num.min(funcEval))
        funcEval = abs(funcEval)
        nsamp    = len(funcEval)
    
    # generate radial average as a difference between encircled energies
        EE = num.zeros((len(funcEval)/2,1))
        self.pol2cartMat(nsamp,1)
        RR   = self.RR*nsamp/2
    
        for ii in num.arange(nsamp/2):
            RR[RR <= ii+radOffSet] = 1
            RR[RR >  ii+radOffSet] = 0
            prod   = num.multiply(RR,funcEval)
            EE[ii] = num.sum(prod)/num.sum(funcEval)
            self.pol2cartMat(nsamp,1)
            RR = self.RR*nsamp/2
    
        self.encE = EE
        return self

#----------------------------------------------------------------------------------------------------#
    def structFcn(self, funcEval,diameter):
        '''evaluate the structure function of fcnEval
        '''
        sizeVec = num.shape(funcEval)
        minVal  = num.min([sizeVec[0], sizeVec[1]]);
        Dfunc1  = num.zeros((minVal));
        Dfunc2  = num.zeros((minVal));
        Dfunc   = num.zeros((minVal));
        sep     = num.linspace(0,minVal-1,minVal)
    
        for ii in sep:
            ff        = (funcEval[0:,1+ii:sizeVec[1]]-funcEval[0:,1:sizeVec[1]-ii])**2    # separation values along X
            gg        = (funcEval[ii:sizeVec[0],:]-funcEval[0:sizeVec[0]-ii,:])**2        # separation values along Y
            Dfunc1[ii] = num.sum(ff[:])#num.mean([ff[:], gg[:]]);
            Dfunc2[ii] = num.sum(gg[:])#num.mean([ff[:], gg[:]]);
            Dfunc[ii]  = num.mean([Dfunc1[ii], Dfunc2[ii]])
    
        ddx = diameter/len(Dfunc)
        sep = sep*ddx;
        self.Dfunc = Dfunc
        self.sep   = sep
        return self

#----------------------------------------------------------------------------------------------------#
    def psfEvalFunc(self, zVec):
        """the model function"""
        zernCalc = lambda RR, TH, mask, nterm: self.waveCreate(zVec)
        waveFunc = zernCalc(RR,TH,mask,nterm)
        self.psfOTF(mask,waveFunc.waveFunc)
        psf = self.psf
        OTF = self.OTF
        OTF = (OTF==0).choose(OTF,1e-011)
        
        self.prOTF = OTF
        self.prPSF = psf
        return self
    
#----------------------------------------------------------------------------------------------------#
    def waveDecomp(self,waveFuncEval,nterm,padFac,nsamp):
        '''A decomposition of the waveFunc into the RMS weighted Zernike polynomials
        '''
# set up the vector of ones to extract from the waveCreate function
        waveFuncEval = num.matrix(waveFuncEval)
        waveFuncEval = num.ravel(waveFuncEval)
        waveFuncEval = num.transpose(waveFuncEval)
        zVec = num.zeros(nterm)
        coeffZ = []
        for ii in num.arange(nterm):
            zVec[ii] = 1 
            self.waveCreate(zVec)
            waveFunc = self.waveFunc
            wavefunc = num.matrix(waveFunc)
            waveFunc = num.ravel(waveFunc)
            coeffVal = num.double(num.dot(waveFunc,waveFuncEval))
            coeffVal = nsamp**2*num.pi/4/padFac**2
            coeffZ.append(coeffVal)
            zVec = num.zeros(nterm)
            
        self.coeffZ = coeffZ
        
        return self    

#----------------------------------------------------------------------------------------------------#
    def factorial(self,n):
        """
        This function calculates the factorial of a number.
        """
        sum = 1.0
        for m in range(1, int(n)+1):
            sum = float(m)*sum
        return sum

#----------------------------------------------------------------------------------------------------#
