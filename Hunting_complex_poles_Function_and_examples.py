"""###################################################
Python code for the paper:

"Extracting an accurate model for permittivity
from experimental data : Hunting complex poles
from the real line"

Elaborated by:

Mauricio Garcia-Vergara*
Guillaume Demesy
Frederic Zolla

Affiliation:

Aix-Marseille Universite, CNRS, Centrale Marseille,
Institut Fresnel UMR 7249, 13013 Marseille, France

* Corresponding author: mauricio.garcia-vergara@fresnel.fr

December 2016
###################################################"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.signal as signal

#Mathematical constants
pi    = np.pi
j     = complex(0,1)
micro = 1e-6
femto = 1e-15
nm    = 1e-9
peta  = 1e15
#Physical constants
c     = 299792458
mu_0  = pi*4e-7
eps_0 = 1./(mu_0*c**2)
###################################################
#-------------------------------------------------------------------------------------------------------#
def Fitting_PQ(w, w_BG, Chi_dat, N_num, N_den):
    #This function :
	#(i)   computes the fitting parameters P_k's and Q_l's
    #(ii)  computes the poles Omega_j and the associated amplitudes A_j
    #(iii) sorts them by decreasing order of amplitude modulus
    j = complex(0,1)
    def error_N2(A,B):
        return linalg.norm(A-B)/linalg.norm(B)*100.

    def Coefs_pq(w_BG, Chi_dat, N_num, N_den):
        #Normalizing the variable w for stability purposes
        w_scale = 0.5*(w_BG.max() + w_BG.min())
        w_BG    = w_BG/w_scale

        #Enforcing Hermitian Symmetry
        Chi_dat   = np.concatenate((np.conj(Chi_dat[::-1]),Chi_dat), axis = 0)
        w_BG      = np.concatenate((-w_BG[::-1]           , w_BG  ), axis = 0)

        #Definition of the matrix Xi
        Xi = np.asarray([(+w_BG*j)**n if n in range(N_num+1) 
                        else -Chi_dat*(+w_BG*j)**(n-N_num) 
                        for n in range(N_num+N_den +1)]).T

        #Computation of the vector r via scipy's least squares routine
        r  = linalg.lstsq(Xi, Chi_dat)[0]
        r = np.insert(r, N_num+1, 1.0)

        P_norm = r[:N_num+1]
        Q_norm = r[N_num+1:N_num+2+N_den]

        chi_aprox  =  np.asarray(sum([(+w_BG*j)**n*P_norm[n] for n in range(len(P_norm))])) 
        chi_aprox /= np.asarray(sum([(+w_BG*j)**n*Q_norm[n] for n in range(len(Q_norm))]))
        error      = error_N2(chi_aprox, Chi_dat)

        print('--'*10)
        print('Fitting Error PQ(%)')
        print(error)

        # Renormalizing the P_k's and Q_l's
        p_norm = np.asarray([P_norm[n]/(w_scale)**n for n in range(len(P_norm))])
        q_norm = np.asarray([Q_norm[n]/(w_scale)**n for n in range(len(Q_norm))])

        return  p_norm, q_norm
    #----------------------------------------------------------------------------------------------#
    #Setting the p's and q's as global variables
    p_norm, q_norm = Coefs_pq(w_BG, Chi_dat, N_num, N_den)

    def Chi_final(w):
        chi  = np.asarray(sum([(+j*w)**n*p_norm[n] for n in range(len(p_norm))]))
        chi /= np.asarray(sum([(+j*w)**n*q_norm[n] for n in range(len(q_norm))]))
        return chi
    #----------------------------------------------------------------------------------------------#

    Chi_f = Chi_final(w)

    #Getting the poles by a numpy routine
    poles = np.roots(q_norm[::-1])/j

    # The Amplitudes A_j's are computed by another least squares procedure
    Mat_part_frac = np.asarray([1/(w-poles[n]) for n in range(len(poles))]).T
    Amps = linalg.lstsq(Mat_part_frac, Chi_f)[0]

    #The results are sorted by the modulus of the A_j's
    Mat_pol_amp = np.array([abs(Amps),Amps, poles])
    Mat_pol_amp = Mat_pol_amp[:, np.argsort(Mat_pol_amp[0])[::-1]]
    return p_norm, q_norm, Mat_pol_amp
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
def Chi_aprox(w, p_norm, q_norm):
    #Chi as a rational function
    chi  = np.asarray(sum([(+j*w)**n*p_norm[n] for n in range(len(p_norm))]))
    chi /= np.asarray(sum([(+j*w)**n*q_norm[n] for n in range(len(q_norm))]))
    return chi
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
def Chi_PP(w,Mat_pol_amp,M):
    #Principal part of Chi, this can be truncated.
    if M == None:
        M = len(Mat_pol_amp[0])
    return np.asarray(sum([Mat_pol_amp[1][n]/(w-Mat_pol_amp[2][n]) for n in range(M) ]))
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
def error(A,B,order):
    return linalg.norm(A-B, ord = order)/linalg.norm(B, ord = order)*100.
####################################################################################################
#Setting the order of approximation and truncation 
J         = 6
Trunc_pol = 4
N_den = J*2
N_num = N_den
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#Loading data
# filetxt = "Ag_Babar_data.txt"
# filetxt = "Ag_Johnson_data.txt"
# filetxt = "Ag_Stahrenberg_data.txt"
# filetxt = "Al_McPeak_data.txt"
# filetxt = "Al_Ordal_data.txt"
# filetxt = "Au_Johnson_data.txt"
# filetxt = "Cu_Johnson_data.txt"
# filetxt = "GaAs_Aspnes_data.txt"
# filetxt = "GaAs_Jellison_data.txt"
# filetxt = "GaP_Aspnes_data.txt"
# filetxt = "GaP_Jellison_data.txt"
# filetxt = "GaSb_Aspnes_data.txt"
# filetxt = "GaSb_Ferrini_data.txt"
# filetxt = "InAs_Aspnes_data.txt"
# filetxt = "ITO_Konig_data.txt"
# filetxt = "Si_Aspnes_data.txt"
filetxt = "Si_Green_data.txt"
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
print('File loaded:', filetxt) 
Lam_dat, n_dat, k_dat = np.loadtxt(filetxt, usecols=(0,1,2), skiprows = 1, unpack = True )


w_BG    = 2*pi*c/(Lam_dat*micro)/peta
Chi_dat = (n_dat + j*k_dat)**2-np.ones_like(w_BG)
w_BG    = w_BG[::-1]
Chi_dat = Chi_dat[::-1]

if np.imag(Chi_dat).max() > 0:
    Chi_dat = np.conj(Chi_dat)

#--------------------------------------------------------------------------------------------------#
## Definition of the vector omega (w)

w_min_val, w_max_val = w_BG.min(), w_BG.max()*1.5

Power = 12
N     = 2**Power
dw    = 2**(1-Power)*w_max_val
w_0   = -np.floor(N/2)*dw
w     = np.linspace(0, N-1, N)*dw + w_0*np.ones(N)
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
print('--'*10)
print('N num = ', N_num)
print('N den = ', N_den)
print('--'*10)

p_norm, q_norm, Mat_pol_amp = Fitting_PQ(w, w_BG, Chi_dat, N_num, N_den)
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#Computing the Fitting errors
Chi_list = [Chi_aprox(w_BG, p_norm, q_norm), Chi_PP(w_BG,Mat_pol_amp,None),
            Chi_PP(w_BG,Mat_pol_amp,2*Trunc_pol)]
chi_name = ['Chi rational', 'Chi PP', 'Chi PP trunc']

for n in range(len(Chi_list)):
    chi = Chi_list[n]
    print(chi_name[n])
    print('error norm 2 (%):', error(chi, Chi_dat, None))
    print('error norm inf (%):',error(chi, Chi_dat, np.inf))
    print('..'*10)
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#Plotting 
w_BG_smooth = np.linspace(w_BG[0], w_BG[-1], 2000)
Chi         = [Chi_aprox(w_BG_smooth, p_norm, q_norm),
               Chi_PP(w_BG_smooth,Mat_pol_amp,None),
               Chi_PP(w_BG_smooth,Mat_pol_amp,2*Trunc_pol)]
Color       = ['y', 'g', 'b']
Title       = [r'$Rational$ $function$ $of$ $order$ $%g$'%(2*J), r'$Partial$ $Fraction$ $Decomposition$',\
               r'$Truncation$ $with$ $%g$ $true$ $poles$'%(Trunc_pol)]
Label       = [r'$\hat{\chi}_{Rational}$',r'$\hat{\chi}_{PF}$',r'$\hat{\chi}_{PF}^{trunc}$']

plt.figure(figsize= (24,16))
for i in range(len(Chi)*2):
    i += 1
    plt.subplot(2,len(Chi),i)
    plt.grid()
    plt.xlim([w_BG[0], w_BG[-1]])    
    if i < 4:
        plt.title(Title[i-1], fontsize = 18)
        if i == 1:
            plt.ylabel(r'$Real$ $part$', fontsize = 18)
        plt.plot(w_BG, np.real(Chi_dat), 'ro', ms = 7.5, label = r'$\hat{\chi}^{Data}$')
        plt.plot(w_BG_smooth, np.real(Chi[i-1]),
                color = Color[i-1], lw = 2, label = Label[i-1])
    else:
        if i == 1+len(Chi):
            plt.ylabel(r'$Imaginary$ $part$', fontsize = 18)
        plt.plot(w_BG, np.imag(Chi_dat), 'ro', ms = 7.5, label = r'$\hat{\chi}^{Data}$')
        plt.plot(w_BG_smooth, np.imag(Chi[i-len(Chi)-1]),
                 color = Color[i-1-len(Chi)], lw = 2, label = Label[i-1-len(Chi)])
        plt.xlabel(r'$\omega$ $[Prad/s]$', fontsize = 18)
    plt.legend(loc = 'best')    
plt.tight_layout()        
plt.show()
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
