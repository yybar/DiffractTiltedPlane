#==============================================================================#
#  author       : Robert Upton
#
#  summary      : A fresnel diffraction calculation between source and rotated
#                 plane
#
#  file name    : DiffractTiltPlane.py
#
#  date         : Mar 25, 2013 5:13:49
#==============================================================================#

import numpy as num, pylab as py
import pdb
import ImageScienceToolBox as ISTB

class CoordSys(object):
    '''
        CoordSys defines three tuples that represent the xp, yp, and zp coordinates
        in an untiled frame with respect to a source.  The frame is located at point
        z0 relative to the source.  We then rotate the frame to get to a tilted
        coordinate system represented by xpp, ypp, zpp.  In this case the rotation is
        accomplished by a right handed rotation about y.  Hence xpp, and zpp are
        updated.
        
        The rotation has to be accomplished with numpy matrix types.
        '''
    def __init__(self,ISTB,parms):
        # constructor for CoordSys
        self.parms    = parms
        self.ISTB     = ISTB
    
#------------------------------------------------------------#
    def vecForm(self,xS0,yS0):
        # form the [xp, zp] vector and perform the necessary rotation
        # form a list, then transform to numpy matrices, and do the
        # matrix multiplication
        
        srcSD    = self.parms['srcSD']
        detSD    = self.parms['detSD']                       # detector semi-diameter
        nsampSrc = self.parms['nsampSrc']
        nsampDet = self.parms['nsampDet']                    # detector sample points
        wavel    = self.parms['wavel']
        z0       = self.parms['z0']
        vecS     = num.matrix(num.zeros((nsampSrc,2)))
        vecD     = num.matrix(num.zeros((nsampDet,3)))       # form the empty vector array
        
        kk = 0
        for ii in range(nsampSrc):
            vecS[kk,0] = 2.0*(ii-nsampSrc/2)*srcSD/nsampSrc-xS0
            vecS[kk,1] = 2.0*(ii-nsampSrc/2)*srcSD/nsampSrc-yS0
            kk = kk + 1
        
        dxS      = abs(vecS[0,0]-vecS[1,0])
        etaRange = 0.5/dxS
        deta     = etaRange/nsampDet
        dxD      = deta
        
        kk = 0
        for ii in range(nsampDet):
            vecD[kk,0] = (ii-nsampDet/2)*dxD                 # x values
            vecD[kk,1] = (ii-nsampDet/2)*dxD                 # y values
            kk = kk + 1
        
        vecSM     = num.matrix(vecS)
        vecDM     = num.matrix(vecD)
        self.vecS, self.vecD = vecSM, vecDM
        
        return vecSM, vecDM
    
#------------------------------------------------------------#
    def CoordDelta(self,vecP):
        # define coordinate system and rotation matrices
        # VecIn has to be a row vector of the type matrix
        
        theta = self.parms['theta']*num.pi/180.0
        RMat  = num.matrix([[num.cos(theta), num.sin(theta)],\
                            [-1*num.sin(theta), num.cos(theta)]])
        
        # Form the rotation matrix
        theta   = self.parms['theta']*num.pi/180
        RMatrix = num.matrix([[num.cos(theta) ,  0, -num.sin(theta)],\
                              [    0          ,  1,      0         ],
                              [-num.sin(theta),  0,  num.cos(theta)]])
        
        vecPP   = RMatrix*vecP.transpose()
        self.vecPP = vecPP.transpose()
        
        return vecPP.transpose()
    
#------------------------------------------------------------#
    def SourceModel(self,vecSM):
        # Generate the source model
        
        wavel     = self.parms['wavel']
        z0        = self.parms['z0']
        srcSD     = self.parms['srcSD']
        nsampSrc  = self.parms['nsampSrc']
        srcPadFac = self.parms['srcPadFac']
        
        # form the source matrices
        unitVec  = num.matrix(num.ones((1,parms['nsampSrc'])))
        SX       = unitVec.transpose()*vecSM[:,0].transpose()[0]
        SY       = vecSM[:,1]*unitVec
        SXA      = num.array(SX)
        SYA      = num.array(SY)
        
        # source phase model
        phaseA   = num.pi*(SXA**2+SYA**2)/(wavel*z0)
        
        # source amplitude model
        rectXA   = SXA
        rectYA   = SYA
        rectXA   = (abs(rectXA) <= srcSD/(2.*srcPadFac)).choose(rectXA,srcSD/(2.*srcPadFac))
        rectXA   = (abs(rectXA) >  srcSD/(2.*srcPadFac)).choose(rectXA,1e-012)
        rectYA   = (abs(rectYA) <= srcSD/(2.*srcPadFac)).choose(rectYA,srcSD/(2.*srcPadFac))
        rectYA   = (abs(rectYA) >  srcSD/(2.*srcPadFac)).choose(rectYA,1e-012)
        rectXA   = rectXA*(2.*srcPadFac/srcSD)
        rectYA   = rectYA*(2.*srcPadFac/srcSD)
        sourceA  = rectXA*rectYA
        
        #self.ISTB.ISTB.pol2cartMat(nsampSrc,srcPadFac)
        #sourceA = num.array(self.ISTB.ISTB.maskPol)
        
        # include phase
        sourceA = sourceA*num.exp(1j*phaseA)
        
        return SXA, SYA, sourceA
    
#------------------------------------------------------------#
    def DetectorModel(self,vecDM):
        # Generate the source model
        
        wavel     = self.parms['wavel']
        detSD     = self.parms['detSD']
        nsampDet  = self.parms['nsampDet']
        
        # form the source matrices
        unitVec  = num.matrix(num.ones((1,parms['nsampDet'])))
        DX       = unitVec.transpose()*vecDM[:,0].transpose()[0]
        DY       = vecDM[:,1]*unitVec
        DXA      = num.array(DX)
        DYA      = num.array(DY)
        
        self.DXA, self.DYA = DXA, DYA
        
        return DXA, DYA
    
#------------------------------------------------------------#
    def DetectorPhaseModel(self,DXA,DYA):
        # introduce phase variations at detector surface
        
        parms = self.parms
        z0    = parms['z0']
        theta = num.pi*parms['theta']/180.0
        wavel = parms['wavel']
        
        phaseTerm1A = z0/wavel
        phaseTerm2A = (z0*wavel/2.0)*((DXA/num.cos(theta))**2+DYA**2)
        phaseTerm3A = -2*z0*DXA*num.tan(theta)
        
        detectorPhaseA = phaseTerm1A + phaseTerm2A + phaseTerm3A
        
        return detectorPhaseA

#------------------------------------------------------------#


#============================================================#
class Diffraction(CoordSys):
    def __init__(self,object,parms):
        # constructor of Diffraction() class
        # with class inheritance from CoordSys()
        CoordSys.__init__(self,object,parms)
        self.CoordSys = CoordSys
    
#------------------------------------------------------------#
    def EvalSelf(self):
        # confirm the methods and attributes in class object
        print "Attributes and methods"
        print "----------------------"
        for ii in dir(self):
            print '%s'%(ii)
        
        print " "
        print "Elements in parms"
        print "-----------------"
        for ii, jj in enumerate(self.parms):
            print '%d%s%s'%(ii,' ',jj)
    
#------------------------------------------------------------#
    def DFTMatrix1D(self):
        # generate the dicrete Fourier transform matrix
        #
        # the DFT matrix elements have the form exp(-2*pi*i*nm/Nd)
        # the IDFT elements have the form exp(2*pi*i*nm/Ns).
        # Note that the Kernel has frequnecy (far-field coordinate)
        # units that are normalized by the number of points in the
        # frequnecy domain.  This is very important to remember
        # otherwise the DFT sampling does not work for nsampSrc <>
        # nsampDet
        
        nsampSrc = self.parms['nsampSrc']
        nsampDet = self.parms['nsampDet']
        srcSD    = self.parms['srcSD']
        detSD    = self.parms['detSD']
        
        # initialize matrix elements
        # real
        HupR = num.array(num.zeros((nsampDet,nsampSrc))) # x and xi
        # imaginary
        HupI = num.array(num.zeros((nsampDet,nsampSrc))) # x and xi
        
        kk  = -nsampDet/2.0             # rows
        ll  = -nsampSrc/2.0             # columns
        for ii in range(nsampDet):
            for jj in range(nsampSrc):
                HupR[ii,jj] = num.real(num.exp(-2.*num.pi*1j*ii*jj/nsampSrc))
                HupI[ii,jj] = num.imag(num.exp(-2.*num.pi*1j*ii*jj/nsampSrc))
                kk = kk + 1
                ll = ll + 1
        
        Hup  = HupR + 1j*HupI
        
        HupM = num.matrix(Hup)
        return HupM
    
#------------------------------------------------------------#
    def Integral1D(self,HupM,sourceA1D):
        # Evaluate the diffraction integral
        
        # convert the source to a matrix
        sourceMp = num.matrix(sourceA1D)
        
        # Evaluate the diffraction integral
        detectorM = HupM*sourceMp.transpose()
        #detectorM = HvqM*(HupM*sourceMpq).transpose()
        detectorM1D = num.fft.fftshift(detectorM)
        
        return detectorM1D
    
    
#------------------------------------------------------------#
    def DFTMatrix2D(self):
        # generate the dicrete Fourier transform matrix
        #
        # the DFT matrix elements have the form exp(-2*pi*i*nm/Ns)
        # the IDFT elements have the form exp(2*pi*i*nm/Nd)
        
        nsampSrc = self.parms['nsampSrc']
        nsampDet = self.parms['nsampDet']
        srcSD    = self.parms['srcSD']
        detSD    = self.parms['detSD']
        
        # initialize matrix elements
        # real
        HupR = num.array(num.zeros((nsampDet,nsampSrc))) # x and xi
        HvqR = num.array(num.zeros((nsampDet,nsampSrc))) # y and eta
        # imaginary
        HupI = num.array(num.zeros((nsampDet,nsampSrc))) # x and xi
        HvqI = num.array(num.zeros((nsampDet,nsampSrc))) # y and eta
        
        for ii in range(nsampDet):
            for jj in range(nsampSrc):
                HupR[ii,jj] = num.real(num.exp(-2.*num.pi*1j*ii*jj/nsampDet))
                HvqR[ii,jj] = num.real(num.exp(-2.*num.pi*1j*ii*jj/nsampDet))
                HupI[ii,jj] = num.imag(num.exp(-2.*num.pi*1j*ii*jj/nsampDet))
                HvqI[ii,jj] = num.imag(num.exp(-2.*num.pi*1j*ii*jj/nsampDet))
        
        Hup = HupR + 1j*HupI
        Hvq = HvqR + 1j*HvqI
        
        HupM, HvqM = num.matrix(Hup), num.matrix(Hvq)
        return HupM, HvqM
    
#------------------------------------------------------------#
    def DFTMatrix2Dscaled(self,vecSM,vecDM):
        # generate DFT components that are scaled by real detector and source coordinates
        #
        #
        
        nsampSrc = self.parms['nsampSrc']
        nsampDet = self.parms['nsampDet']
        srcSD    = self.parms['srcSD']
        detSD    = self.parms['detSD']
        
        # initialize matrix elements
        # real
        HupR = num.array(num.zeros((nsampDet,nsampSrc))) # x and xi
        HvqR = num.array(num.zeros((nsampDet,nsampSrc))) # y and eta
        # imaginary
        HupI = num.array(num.zeros((nsampDet,nsampSrc))) # x and xi
        HvqI = num.array(num.zeros((nsampDet,nsampSrc))) # y and eta
        
        kk = 0
        for ii in range(nsampDet):
            ll = 0
            for jj in vecSM[:,0]:
                HupR[kk,ll] = num.real(num.exp(-2.*num.pi*1j*ii*jj/srcSD))
                HupI[kk,ll] = num.imag(num.exp(-2.*num.pi*1j*ii*jj/srcSD))
                ll = ll + 1
            kk = kk +1
        
        kk = 0
        for ii in range(nsampDet):
            ll = 0
            for jj in vecSM[:,1]:
                HvqR[kk,ll] = num.real(num.exp(-2.*num.pi*1j*ii*jj/srcSD))
                HvqI[kk,ll] = num.imag(num.exp(-2.*num.pi*1j*ii*jj/srcSD))
                ll = ll + 1
            kk = kk +1
        
        
        
        Hup = HupR + 1j*HupI
        Hvq = HvqR + 1j*HvqI
        
        HupM, HvqM = num.matrix(Hup), num.matrix(Hvq)
        return HupM, HvqM
    
#------------------------------------------------------------#
    def Integral2D(self,HupM,HvqM,sourceA, detectorPhaseA, DXA, DYA):
        # Evaluate the diffraction integral
        
        wavel = self.parms['wavel']
        theta = num.pi/180.*self.parms['theta']
        z0    = self.parms['z0']
        nsampDet = self.parms['nsampDet']
        
        # convert the source to a matrix
        sourceMpq = num.matrix(sourceA)
        
        # Evaluate the diffraction integral
        detectorM = HupM*sourceMpq*HvqM.transpose()
        detectorM = num.fft.fftshift(detectorM)
        detectorA = num.array(detectorM)
        
        detectorMult  = (-1j/wavel)*num.exp(-2*num.pi*1j*detectorPhaseA)
        radiusA       = z0-wavel*z0*DXA*num.tan(theta)
        detectorMultA = detectorMult/radiusA
        detectorAtot  = detectorMultA*detectorA
        detectorTotM  = num.matrix(detectorAtot)
        
        return detectorTotM
    
#------------------------------------------------------------#
    def EvalIntegral(self):
        # evaluate and plot a test case
        parms          = self.parms
        nsampSrc       = parms['nsampSrc']
        nsampDet       = parms['nsampDet']
        wavel          = parms['wavel']
        z0             = parms['z0']
        theta          = num.pi/180.0*parms['theta']
        srcSD          = parms['srcSD']
        nSourcePoints  = parms['nSourcePoints']
        
        detectorIrradAccumA = num.zeros((nsampDet,nsampDet))
        sourceIrradAccumA   = num.zeros((nsampSrc,nsampSrc))
        
        for ii in range(nSourcePoints):
            for jj in range(nSourcePoints):
            
                yS0 = (ii-nSourcePoints/2)*srcSD/nSourcePoints
                xS0 = (jj-nSourcePoints/2)*srcSD/nSourcePoints
                vecSM, vecDM   = self.vecForm(xS0,yS0)
        
                SXA, SYA, sourceA = self.SourceModel(vecSM)
                DXA, DYA          = self.DetectorModel(vecDM)
                sourceIrradAccumA = sourceIrradAccumA+sourceA
                detectorPhaseA    = self.DetectorPhaseModel(DXA,DYA)
        
        #HupM, HvqM        = self.DFTMatrix2Dscaled(vecSM,vecDM)
                HupM, HvqM          = self.DFTMatrix2D()
                detectorM2D         = self.Integral2D(HupM,HvqM,sourceA, detectorPhaseA, DXA, DYA)
                detectorM2A         = num.array(detectorM2D)
                detectorIrradA      = num.real(num.conj(detectorM2A)*detectorM2A)
                detectorIrradAccumA = detectorIrradAccumA + detectorIrradA
        
        # scale the DXA and DYA matrices
        vecSM, vecDM      = self.vecForm(0,0)
        SXA, SYA, sourceA = self.SourceModel(vecSM)
    
        DXAscale, DYAscale = wavel*z0*DXA/num.cos(theta), wavel*z0*DYA
        
        # plot things
        py.ion()
        py.figure(figsize = (18,6))
        
        py.subplot(131)
        py.pcolor(SXA,SYA,abs(sourceIrradAccumA))
        ax = py.gca()
        ax.axis(aspect=[1,1,1])
        py.xlabel(r'$x_{S}$' ' ' r'$(mm)$')
        py.ylabel(r'$y_{S}$' ' ' r'$(mm)$')
        py.title(r'$S(x_{S},y_{S})$')
        
        py.subplot(132)
        py.pcolor(DXAscale,DYAscale,(detectorIrradAccumA/num.max(num.max(detectorIrradAccumA))))
        #py.clim(-4,0)
        py.xlim(num.min(DXAscale),-1*num.min(DXAscale))
        ax = py.gca()
        py.xlabel(r'$x_{D}$' ' ' r'$(mm)$')
        py.ylabel(r'$y_{D}$' ' ' r'$(mm)$')
        py.title(r'$D(x_{D},y_{D})$')
        ax.axis('equal')
        
        py.subplot(133)
        py.plot(DXAscale[nsampDet/2,:],(abs(detectorIrradAccumA[:,nsampDet/2.])/\
                                        num.max(abs(detectorIrradAccumA))))
        py.plot(DXAscale[nsampDet/2,:],(abs(detectorIrradAccumA[nsampDet/2.,:])/\
                                        num.max(abs(detectorIrradAccumA))))
        deltaIrradA = detectorIrradAccumA[:,nsampDet/2.] - detectorIrradAccumA[nsampDet/2.,:]
        py.plot(DXAscale[nsampDet/2,:],100*((deltaIrradA)/num.max(abs(detectorIrradAccumA))))
        py.xlim(num.min(DXAscale),-1*num.min(DXAscale))
        py.xlabel(r'$x_{D}$' ' ' r'$(mm)$')
        py.title(r'$D(x_{D},0)$' ', ' r'$D(0,x_{D})$' ', ' r'$D(x_{D},0)-D(0,x_{D})$')


#============================================================#
if __name__ == "__main__":
    
    def main():
        # instantiate the class
        parms = {'wavel'     : 0.5e-03   ,                   # wavelength
            'detSD'     : 12.255         ,                   # detector semi-diameter
            'nsampDet'  : 128            ,                    # detector sample points
            'srcSD'     : 0.5            ,                   # source semi-diameter
            'nsampSrc'  : 256            ,                    # source sample points
            'nSourcePoints' : 15         ,                   # number of incoherent source points
            'srcPadFac' : 64             ,                   # source padding factor
            'z0'        : 221.72         ,                   # distance to tilted plane
            'theta'     : 60.0           ,                   # tilt angles of plane
                
            }
        
        ImClass = ISTB.ImageScienceClass()
        CoordSysObj = CoordSys(ImClass,parms)
        DiffObj = Diffraction(CoordSysObj,parms)
        
        
        return DiffObj, ImClass, parms
#------------------------------------------------------------#
    
    DiffObj, ImClass, parms = main()                         # instantiate the class
    DiffObj.EvalIntegral()
    
    raw_input('Press <Enter> to finish:')
