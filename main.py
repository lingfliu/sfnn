from sig_simulator import simu_sinusoid_sig, simu_gaussian
from matplotlib import pyplot as pp
import mpmath

def main():
    sps = 200
    sig_len = 6000
    (x,y) = simu_sinusoid_sig(sps=sps, x_len=sig_len, phase=mpmath.pi/6)
    n = simu_gaussian(x, 0, 0.1)


    # define a 1*N input layer
    
    # define a N*N hidden layer
    #define a N*1 output layer

    pp.plot(x,y+n)
    pp.show()


if __name__ == '__main__':
    main()