#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.optimize import minimize
import create_Colormap as cmap
import os
"""import pylustrator
pylustrator.start()"""

file_path = os.getcwd()
"global variables"
c0 = 299792458
"simulation parameters"
min_lam = 300
min_lam_sel = int(min_lam/3)
max_lam = 2200
lam_step = 10
I0 = 1e14
c_corr = 1.3
A0_chi_2 = c_corr*(1.1*1e-15*(4*np.pi)**3)
a1 = 1e-25
num_steps = 1e3
Z = 15e-6
DZ = Z/num_steps 
Lambda_sel = np.linspace(min_lam_sel, max_lam,int((max_lam-min_lam/3)+1), dtype='int32')
Lambda_sim = np.linspace(min_lam, max_lam, int((max_lam-min_lam)/lam_step + 1), dtype='int32')
SHG_Int = np.zeros(len(Lambda_sim))
Z_t = np.zeros(len(SHG_Int))
alpha_lookup = np.zeros(len(Lambda_sel))
n_lookup = np.zeros(len(Lambda_sel))
Chi2_lookup = np.zeros(len(Lambda_sel))
#%% Function declarations
def Lorentz_Oscillator(Omega, Omega_0, A, B, C):
    return 1 + A * (Omega_0**2 - Omega**2)/((Omega_0**2 - Omega**2)**2 + B * Omega**2) + C

def Sellmeier_n(B1,B2,B3,C1,C2,C3,Lambda):
    Lambda = Lambda/1000
    n_square_min_one = B1*Lambda**2/(Lambda**2-C1) + B2*Lambda**2/(Lambda**2-C2) + B3*Lambda**2/(Lambda**2-C3) #calculate n^2-1 via sellmeier
    n = np.sqrt(n_square_min_one + 1) 
    return n

def Sellmeier_Chi1(B1,B2,B3,C1,C2,C3,Lambda):
    Lambda = Lambda/1000
    n_square_min_one = B1*Lambda**2/(Lambda**2-C1) + B2*Lambda**2/(Lambda**2-C2 ) + B3*Lambda**2/(Lambda**2-C3) #calculate n^2-1 via sellmeier
    return n_square_min_one 

def Miller_Rule_Sel_Chi2(A0, Lambda):
    n_square_2w = Sellmeier_Chi1(LN_B1, LN_B2, LN_B3, LN_C1, LN_C2, LN_C3, Lambda/2)
    n_square_w = Sellmeier_Chi1(LN_B1, LN_B2, LN_B3, LN_C1, LN_C2, LN_C3, Lambda)
    Chi2 = A0*n_square_2w*n_square_w
    return Chi2

def Miller_Rule_Sel_Chi3(A0, A1, Lambda):
    n_square_3w = Sellmeier_Chi1(LN_B1, LN_B2, LN_B3, LN_C1, LN_C2, LN_C3, Lambda/3)
    n_square_2w = Sellmeier_Chi1(LN_B1, LN_B2, LN_B3, LN_C1, LN_C2, LN_C3, Lambda/2)
    n_square_w = Sellmeier_Chi1(LN_B1, LN_B2, LN_B3, LN_C1, LN_C2, LN_C3, Lambda)
    Chi3 = A0*n_square_3w*n_square_w**3 + A1*n_square_3w*n_square_2w*n_square_w**3 
    return Chi3

def Miller_Rule_Lor_Chi2(A0, Lambda, *popt):
    n_square_2w = get_n(Lambda/2,*popt)**2 - 1
    n_square_w = get_n(Lambda,*popt)**2 - 1
    Chi2 = A0*n_square_2w*n_square_w
    return Chi2

def Miller_Rule_Lor_Chi3(A0, A1, Lambda,*popt):
    n_square_3w = get_n(Lambda/3,*popt)**2 - 1
    n_square_2w = get_n(Lambda/2,*popt)**2 - 1
    n_square_w = get_n(Lambda,*popt)**2 - 1
    #n_square_3w = Sellmeier_Chi1(LN_B1, LN_B2, LN_B3, LN_C1, LN_C2, LN_C3, Lambda/3)
    #n_square_2w = Sellmeier_Chi1(LN_B1, LN_B2, LN_B3, LN_C1, LN_C2, LN_C3, Lambda/2)
    #n_square_w = Sellmeier_Chi1(LN_B1, LN_B2, LN_B3, LN_C1, LN_C2, LN_C3, Lambda)
    Chi3 = A0*n_square_3w*n_square_w**3 + A1*n_square_3w*n_square_2w*n_square_w**3 
    return Chi3

def fit_alpha(x, a ,b, c):
    return a/(x-b)**2 + c

Lambda_alpha_e = np.array([378,381,384,388,392,397,403,412,423,431,439,451,463,485,510,533,558,581,606,630,653,680,704,727,753,771,793])
alpha_e_rez_cm = np.array([0.0788, 0.0686, 0.0626, 0.0570, 0.0514, 0.0462, 0.0401, 0.0348, 0.0292, 0.0260, 0.0240, 0.0218, 0.0200, 0.0170, 0.0137, 0.0112, 0.0085, 0.0066, 0.0058, 0.0053, 0.0052, 0.0051, 0.0049, 0.0044, 0.0037, 0.0037, 0.0034]) #unit 1/cm
alpha_e_rez_m = alpha_e_rez_cm*1e2

def set_alpha(Lambda, *popt):
    alpha_dummy = fit_alpha(Lambda, *popt)
    for idx, element in enumerate(alpha_dummy):
        if (element > 1e4 or Lambda[idx] <= popt[1]): alpha_dummy[idx] = 1e4
    return alpha_dummy

def get_alpha(Lambda, *popt):
    alpha = fit_alpha(Lambda, *popt)
    if (alpha > 1e4 or Lambda <= popt[1]): alpha = 1e4
    return alpha

def get_n(Lambda, *popt):
    Omega = 2*np.pi*c0/Lambda*1e9/1e15
    n_0 = Lorentz_Oscillator(Omega, *popt)
    return n_0

def get_Chi2(Lambda):

    return Chi2_lookup[idx]

def fit_Chi2(x, a ,b ,c):
    return a/(x-b)**2 + c

def f_3ord():
    return 1

def jac_3ord():
    return 1

def f(t, A, omega1,omega2,alpha1,alpha2,Chi2_Fund, Chi2_SHG,c0,n1,n2):
    return [1j*omega1/(2*n1*c0)*Chi2_Fund*np.conj(A[0])*A[1] - 1/2 * alpha1*A[0], 1j*omega2/(4*n2*c0)*Chi2_Fund*A[0]**2 - 1/2 * alpha2*A[1]]

def jac(t, A, omega1,omega2,alpha1,alpha2,Chi2_Fund,Chi2_SHG,c0,n1,n2):
    return [[1j*omega1/(2*n1*c0)*Chi2_Fund*A[1] - 1/2 * alpha1, 1j*omega1/(2*n1*c0)*Chi2_Fund*np.conj(A[0])], [1j*omega2/(2*n2*c0)*Chi2_Fund*A[0],-1/2 * alpha2]]

def solve_NLO_waveq_2(z, dz, Lambda, I0):
    A0, z0 = [np.sqrt(I0), 0], 0
    OmegaFund = 2*np.pi*c0*1e9/Lambda
    OmegaSHG = 4*np.pi*c0*1e9/Lambda
    alphaFund = get_alpha(Lambda,*popt_alpha)
    alphaSHG = get_alpha(Lambda/2,*popt_alpha)
    nFund = get_n(Lambda,*popt_lor)
    nSHG = get_n(Lambda/2,*popt_lor)
    chi2 = Miller_Rule_Lor_Chi2(A0_chi_2, Lambda, *popt_lor)
    chi2_Fund = Miller_Rule_Lor_Chi2(A0_chi_2, Lambda, *popt_lor)
    chi2_SHG = Miller_Rule_Lor_Chi2(A0_chi_2, Lambda/2, *popt_lor)
    Chi2 = chi2
    r = ode(f, jac).set_integrator('zvode', method='bdf')

    r.set_initial_value(A0, z0).set_f_params(OmegaFund,OmegaSHG,alphaFund,alphaSHG,chi2_Fund,chi2_SHG,c0,nFund,nSHG).set_jac_params(OmegaFund,OmegaSHG,alphaFund,alphaSHG,chi2_Fund,chi2_SHG,c0,nFund,nSHG)

    z1 = 1000e-6

    dz = 1e-7

    z = np.zeros(int(z1/dz))
    y = np.zeros((np.size(z),2),dtype=np.complex_)

    for j in range(int(z1/dz)):
        if r.successful():
            z[j] = r.t+dz
            y[j,:] = r.integrate(r.t+dz)
    return z, y  #returns SHG intensity




popt_alpha, pcov_alpha = curve_fit( fit_alpha, Lambda_alpha_e, alpha_e_rez_m, bounds = (0,[np.inf, 400, 1]))
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12,
    "font.family": "DejaVu Sans"
})
omega = 2*np.pi*c0/Lambda_sel*1e9/1e15
popt_lor = 7.8, 15, 1, .973
n_Lor = Lorentz_Oscillator(omega, *popt_lor)


'DOI: 10.1364/JOSAB.14.003319'
"ordinary"
LN_B1 = 2.6734
LN_B2 = 1.2290
LN_B3 = 12.614
LN_C1 = 0.01764
LN_C2 = 0.05914
LN_C3 = 474.60
n_Sel = Sellmeier_n(LN_B1, LN_B2, LN_B3, LN_C1, LN_C2, LN_C3, Lambda_sel)


popt_alpha, pcov_alpha = curve_fit( fit_alpha, Lambda_alpha_e, alpha_e_rez_m, bounds = (0,[np.inf, 400, 1]))
f1 = plt.figure()
plt.plot(Lambda_sel, Miller_Rule_Lor_Chi2(1.5*A0_chi_2, Lambda_sel,*popt_lor), label = r'$\chi^{(2)}$')
plt.plot(Lambda_sel, Miller_Rule_Lor_Chi3(1.5*A0_chi_2*1e-2,A0_chi_2*1e-2, Lambda_sel,*popt_lor),label = r'$\chi^{(3)}$')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Tensor Strength (a.u.)')
plt.xlim(200,2000)
plt.ylim(0,2e-10)
leg = plt.legend(loc = 'best')
leg.get_frame().set_linewidth(0.0)
f1.set_size_inches(5.48, 5.48*9/16)
plt.tight_layout()

savefilename_disp = file_path + '\\LN_SHG_millerdisp.pdf'
f1.savefig(savefilename_disp)
plt.show()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(Lambda_sel,n_Sel,':', label = r'$\mathrm{n}_{Sel}$')
ax1.plot(Lambda_sel,n_Lor, label = r'$\mathrm{n}_{Lor}$')
ax2.plot(Lambda_sel, set_alpha(Lambda_sel,*popt_alpha), label = r'$\alpha$')
ax1.set_xlabel('X data')
ax1.set_ylim(0,4)
ax1.set_ylabel('Y1 data', color='g')
ax2.set_ylabel('Y2 data', color='b')
ax2.set_yscale('log')
plt.xlim(min(Lambda_sel),max(Lambda_sel))
plt.figure(1).axes[0].lines[0].set_color("#737373ff")
plt.figure(1).axes[0].lines[0].set_linestyle("--")
plt.figure(1).axes[0].lines[0].set_linewidth(2.5)
plt.figure(1).axes[0].lines[1].set_color("#900000ff")
plt.figure(1).axes[0].lines[1].set_linewidth(2.5)
plt.figure(1).axes[0].get_xaxis().get_label().set(position=(0.5, 32.90909377462568), text='Wavelength (nm)', ha='center', va='top')
plt.figure(1).axes[0].get_yaxis().get_label().set(position=(60.47409377462569, 0.5), text='n (a.u.)', ha='center', va='bottom', color='#000000ff', rotation=90.0)
plt.figure(1).axes[1].lines[0].set_linewidth(2.5)
plt.figure(1).axes[1].lines[0].set_color("#0000ff")
plt.figure(1).axes[1].get_yaxis().get_label().set(position=(60.47409377462569, 0.5), text=r'$\alpha$ ($\mathrm{m}^{-1}$)', ha='center', va='top', color='#000000ff')
plt.figure(1).axes[0].legend(frameon=False, markerscale=2.0, fontsize=10, title_fontsize=10)
plt.figure(1).axes[0].get_legend()._set_loc((0.78, 0.80))
plt.figure(1).axes[1].legend(frameon=False, markerscale=2.0, fontsize=10, title_fontsize=10)
plt.figure(1).axes[1].get_legend()._set_loc((0.78, 0.70))
#plt.figure(1).axes[0].set_yticklabels(["0", "1.0", "2.0", "3.0", "4.0"], fontsize=12.0, fontweight="normal", color="black", horizontalalignment="center")
#plt.figure(1).axes[1].set_yticklabels(["0","0", "2e3", "4e3", "6e3", "8e3","10e3"], fontsize=12.0, fontweight="normal", color="black", horizontalalignment="center")
fig.set_size_inches(5.48, 5.48*9/16)
plt.tight_layout()

savefilename_coeff = file_path + '\\LN_SHG_coeff.pdf'
fig.savefig(savefilename_coeff)
plt.show()
#%%

for idx,LAMBDA in enumerate(Lambda_sim):
    #Chi_3 = Miller_Rule_Chi3(A0_chi_3,A1_chi_3,Lambda)
    z, y = solve_NLO_waveq_2(Z, DZ, LAMBDA, I0)
    #plt.plot(z,y[:,0]*np.conj(y[:,0]), label = 'Fund')
    #plt.plot(z,y[:,1]*np.conj(y[:,1]), label = 'SHG')
    #plt.show()

    SHG_Int[idx] = y[-1,1]*np.conj(y[-1,1])

#%%

Fwavel, SHG_norm = cmap.create_cmap()
popt,pcov = curve_fit(fit_Chi2, Lambda_sim[130:], SHG_Int[130:], bounds = ([1e18, 350,1e11],[np.inf, 500,1e14]))
f2 = plt.figure()
Chi2_Fit = fit_Chi2(Lambda_sim,*popt)/max(SHG_Int)
Chi2_Sim = SHG_Int/max(SHG_Int)
plt.plot(Lambda_sim/2,Chi2_Sim,'--b', label = 'SHG Sim')
plt.plot(Lambda_sim/2,Chi2_Fit,'r-',label = 'SHG Fit')

SHG_interp = np.interp(Lambda_sim/2, Fwavel/2, (SHG_norm/max(SHG_norm)))
cond_fit = Lambda_sim/2 > min(Fwavel/2)
fun = lambda x: np.mean((Chi2_Sim[cond_fit] - (SHG_interp[cond_fit]*x[0] + x[1]))**2)
x0 = [1,0]
bnds = ((0,2),(0,1))
res = minimize(fun, x0, method='SLSQP', bounds = bnds)
X = res.x
plt.plot(Fwavel/2, (SHG_norm/max(SHG_norm))*X[0]+ X[1],'x', label = 'Exp. data')

plt.ylim(0, 1.1)
plt.xlim(200, 1000)
plt.xlabel(r'SHG Wavelength (nm)')
plt.ylabel(r'SHG Intensity (a.u.)')

plt.grid()
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
plt.figure(1).axes[0].lines[0].set_color("#737373ff")
plt.figure(1).axes[0].lines[0].set_linestyle("--")
plt.figure(1).axes[0].lines[0].set_linewidth(2.5)
plt.figure(1).axes[0].lines[0].set_markeredgecolor("#737373ff")
plt.figure(1).axes[0].lines[0].set_markerfacecolor("#737373ff")
plt.figure(1).axes[0].lines[1].set_color("#900000ff")
plt.figure(1).axes[0].lines[1].set_linewidth(2.5)
plt.figure(1).axes[0].lines[1].set_markeredgecolor("#900000ff")
plt.figure(1).axes[0].lines[1].set_markerfacecolor("#900000ff")
plt.figure(1).axes[0].lines[2].set_color("#0000ff")
plt.figure(1).axes[0].lines[2].set_markersize(5)
plt.figure(1).axes[0].lines[2].set_markeredgecolor("#0000ff")
plt.figure(1).axes[0].lines[2].set_markerfacecolor("#0000ff")
plt.legend(loc = 'best')
plt.figure(1).axes[0].legend(frameon=False, fontsize=10, title_fontsize=10)

f2.set_size_inches(5.48, 5.48*9/16)
plt.tight_layout()

savefilename_sim = file_path + '\\LN_SHG_sim.pdf'
f2.savefig(savefilename_sim)
plt.show()
#%%
#Account for different ulse durations and beam diameters