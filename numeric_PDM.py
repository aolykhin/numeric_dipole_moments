from logging import raiseExceptions
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyscf import gto, scf, mcscf, lib, lo
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import os
from pyscf.tools import molden
import copy
from colored import fg, attr
from pyscf.data import nist

def cs(text): return fg('light_green')+str(text)+attr('reset')

def pdtabulate(df, line1, line2): return tabulate(df, headers=line1, tablefmt='psql', floatfmt=line2)

# ------------------ NUMERICAL DIPOLE MOMENTS ----------------------------
def numer_run(dist, x, mol, mo_zero, ci_zero, method, field, ifunc, out, dip_cms):
    '''
    Returns numeric permanent dipole moment found by differentiation of
    electronic energy with respect to electric field strength. 
    '''
    # Set reference point to be center of charge
    mol.output='num_'+ out
    mol.build()
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nuc_charge_center = np.einsum(
        'z,zx->x', charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    if x.unit.upper() == 'DEBYE':
        fac = nist.AU2DEBYE
    elif x.unit.upper() == 'AU':
        fac = 1
    else:
        RuntimeError 

    h_field_off = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
    dip_num = np.zeros((len(field), 1+4*x.iroots))# 1st column is the field column
    for i, f in enumerate(field): # Over field strengths
        dip_num[i][0]=f # the first column is the field column 
        if x.formula == "2-point":
            disp = [-f, f]
        elif x.formula == "4-point":
            disp = [-2*f, -f, f, 2*f]
        e = np.zeros((len(disp),x.iroots))
        if i==0: #set zero-field MOs as initial guess 
            mo_field = []
            for icoord in range(3): mo_field.append([mo_zero]*len(disp))

        for j in range(3): # Over x,y,z directions
            for k, v in enumerate(disp): # Over stencil points 
                if j==0:   # X-axis
                    E = [v, 0, 0]
                elif j==1: # Y-axis
                    E = [0, v, 0]
                elif j==2: # Z-axis
                    E = [0, 0, v]
                # HF step
                h_field_on = np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
                h = h_field_off + h_field_on
                mf_field = scf.RHF(mol)
                mf_field.get_hcore = lambda *args: h
                mf_field.max_cycle = 1  # next CASSCF starts from CAS orbitals not RHF
                mf_field.kernel()

                mc = mcpdft.CASSCF(mf_field, ifunc, x.norb, x.nel, grids_level=x.grid)
                mc.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
                mc.fcisolver.wfnsym = x.irep
                # mc.fix_spin_(ss=x.ispin)
                # print('j=%s and k=%s' %(j,k))
                if i==0: #First field starts with zero-field MOs
                    mc.max_cycle_macro = 600
                    mo=mo_zero 
                else: # Try MOs from the previous filed/point
                    mc.max_cycle_macro = 5
                    mo=mo_field[j][k]
                # if the threshold is too tight the active space is unstable
                if thresh: 
                    mc.conv_tol = conv_tol
                    mc.conv_tol_grad = conv_tol_grad
                weights=[1/x.iroots]*x.iroots #Equal weights only
                # MC-PDFT 
                if method == 'SS-PDFT':
                    e_mcpdft = mc.kernel(mo)[0]
                    if mc.converged==False: 
                        mc.max_cycle_macro = 600
                        e_mcpdft = mc.kernel(mo_zero)[0]
                    e[k] = e_mcpdft
                # SA-PDFT 
                elif method == 'SA-PDFT':
                    mc=mc.state_average_(weights)
                    mc.kernel(mo)
                    if mc.converged==False:
                        mc.max_cycle_macro = 600
                        mc.run(mo_zero)
                    e[k] = mc.e_states #List of energies
                # CMS-PDFT 
                elif method == 'CMS-PDFT':
                    mc=mc.multi_state(weights,'cms')
                    mc.kernel(mo)
                    if mc.converged==False:
                        mc.max_cycle_macro = 600
                        mc.run(mo_zero)
                    e[k] = mc.e_states.tolist() #List of  energies
                else:
                    raise NotImplementedError
                mo_field[j][k] = mc.mo_coeff #save MOs for the next stencil point k and axis j 

            for m in range(x.iroots): # Over states
                shift=m*4 # shift to the next state by 4m columns (x,y,z,mu)
                if x.formula == "2-point":
                    # print('Energy diff',-e[0][m]+e[1][m])
                    dip_num[i,1+j+shift] = (-1)*fac*(-e[0][m]+e[1][m])/(2*f)
                elif x.formula == "4-point":
                    # print('Energy diff',e[0][m]-8*e[1][m]+8*e[2][m]-e[3][m])
                    dip_num[i,1+j+shift] = (-1)*fac*(e[0][m]-8*e[1][m]+8*e[2][m]-e[3][m])/(12*f)
        
        # Get absolute dipole moment    
        for m in range(x.iroots):
            shift=m*4 # shift to the next state by 4m columns (x,y,z,mu)    
            dip_num[i,4+shift] = np.linalg.norm(dip_num[i,1+shift:4+shift])
    
    #Save covergence plots
    # num_conv_plot(x, field, dip_num, dist, method, dip_cms)
    return dip_num

def num_conv_plot(x, field, dip_num, dist, method, dip_cms):
    y1=[None]*x.iroots
    x1=field.flatten()
    # x1=field
    for m in range(x.iroots):
        plt.figure(m)
        shift=m*4
        y1[m]=dip_num[:,4+shift]
        plt.scatter(x1, y1[m])
        if method == 'CMS-PDFT':
            analyt_dipole=dip_cms[shift+3]
        else:
            raise NotImplementedError

        x_new = np.linspace(x1.min(), x1.max(),500)
        f = interp1d(x1, y1[m], kind='quadratic')
        y_smooth=f(x_new)
        plt.plot (x_new,y_smooth)

        plt.title('Distance = %s and Sate = %s' %((dist,m+1)))
        plt.xlabel('Field / au')
        plt.ylabel('Dipole moment / D')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        plt.tick_params(axis='x', which='minor', bottom=False)
        plt.axhline(y = analyt_dipole, color = 'k', linestyle = ':')
        plt.savefig('fig_'+x.iname+'_'+f"{dist:.2f}"+'_st_'+str(m+1)+'_'+method+'.png')
    return

def init_guess(y, analyt, numer):
    ''' 
    Runs preliminary state-specific or state-averaged CASSCF 
    at a given geometry and returns molecular orbitals and CI vectors
    '''
    out = y.iname+'_cas'
    xyz = open('00.xyz').read() if y.geom == 'frames' else y.geom
    mol = gto.M(atom=xyz, charge=y.icharge, spin=y.ispin,
                output=y.iname+'_init.log', verbose=4, basis=y.ibasis, symmetry=y.isym)
    weights=[1/x.iroots]*x.iroots
    mf = scf.RHF(mol).run()
    molden.from_mo(mol,'orb_'+y.iname+'_init_hf.molden', mf.mo_coeff)

    fname = 'orb_'+ out
    cas = mcscf.CASSCF(mf, y.norb, y.nel)
    cas.chkfile = fname
    cas.fcisolver.wfnsym = y.irep
    cas.fix_spin_(ss=y.ispin) 
    if thresh: 
        cas.conv_tol = conv_tol 
        cas.conv_tol_grad = conv_tol_grad 

    print(f'Guess MOs from HF at {y.init:3.5f} ang')
    mo = mcscf.sort_mo(cas, mf.mo_coeff, y.cas_list)

    if 'MC-PDFT' in (analyt + numer):
        # cas.fcisolver.wfnsym = x.irep
        # cas.fix_spin_(ss=x.ispin) 
        cas.kernel(mo)
        mo_ss=cas.mo_coeff
    sa_required_methods=['CMS-PDFT','SA-PDFT','SA-CASSCF']
    if any(x in analyt+numer for x in sa_required_methods):
        cas.state_average_(weights)
        cas.fcisolver.wfnsym = y.irep
        cas.fix_spin_(ss=y.ispin) 
        cas.kernel(mo)
    cas.analyze()
    mo = cas.mo_coeff
    ci = cas.ci
    molden.from_mo(mol, out+'_init_cas.molden', cas.mo_coeff)
    return mo, ci

#-------------Compute energies and dipoles for the given geometry-----------
def get_dipole(x, field, numer, analyt, mo, ci, dist, dmcFLAG=True):
    '''
    Evaluates energies and dipole moments along the bond contraction coordinate
    Ruturns analytical and numerical dipoles for a given geometry for each functional used 
    Also returns the MOs and CI vectors for a given geometry 
    '''
    # 'frames'
    out = x.iname+'_'+f"{dist:.2f}"
    # xyz = open(str(dist).zfill(2)+'.xyz').read() if x.iname == 'spiro' else x.iname+'.xyz'
    mol = gto.M(atom=x.geom,charge=x.icharge,spin=x.ispin,output=out+'.log',
                verbose=4, basis=x.ibasis, symmetry=x.isym)
    
    #Determine origin
    mass    = mol.atom_mass_list()
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    origin = "Charge_center" if x.icharge !=0 else "Coord_center" 
    # mol.set_common_orig_(mass_center)

    weights=[1/x.iroots]*x.iroots
    
    #HF step
    mf = scf.RHF(mol).set(max_cycle = 1).run()

    #SS/SA-CASSCF step
    fname = 'orb_'+ out #file with orbitals
    cas = mcscf.CASSCF(mf, x.norb, x.nel)
    # cas.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
    cas.chkfile = fname
    if thresh: 
        cas.conv_tol = conv_tol
        cas.conv_tol_grad = conv_tol_grad
    #! may be redundant if geometry is the same as ini 
    # print('Project orbs from the previous point')
    # mo = mcscf.project_init_guess(cas, mo)

    if 'MC-PDFT' in (analyt + numer):
        cas.fcisolver.wfnsym = x.irep
        cas.fix_spin_(ss=x.ispin) 
        e_casscf = cas.kernel(mo,ci)[0]
        mo_ss=cas.mo_coeff
        cas.analyze()
        molden.from_mo(mol, out+'_ss.molden', cas.mo_coeff)

    sa_required_methods=['CMS-PDFT','SA-PDFT','SA-CASSCF']
    if any(x in analyt+numer for x in sa_required_methods):
        cas.state_average_(weights)
        cas.fcisolver.wfnsym = x.irep
        cas.fix_spin_(ss=x.ispin)
        e_casscf = cas.kernel(mo,ci)[0]
        mo_sa = cas.mo_coeff
        ci = cas.ci
        cas.analyze()
        molden.from_mo(mol, out+'_sa.molden', cas.mo_coeff)
    mo=cas.mo_coeff 
    ci=cas.ci 
    
    #MC-PDFT step
    numeric  = [None]*len(x.ontop)
    analytic = [None]*len(x.ontop)
    energy   = [None]*len(x.ontop) # energy array indexed by functional

    for k, ifunc in enumerate(x.ontop): # Over on-top functionals

        mol.output=x.iname+'_'+ifunc+'_'+f"{dist:.2f}"+'.log'
        mol.build()
        #Initialize to zero
        dip_cms  = np.zeros(4*x.iroots).tolist()
        abs_pdft = 0
        abs_cas  = 0
        e_pdft   = 0
#-------------------- Energy ---------------------------
        if not analyt:
            print("Analytic Energy and Dipole are ignored")
        else:
            for method in analyt:
                #---------------- Make a PDFT object ---------------------------
                mc = mcpdft.CASSCF(mf, ifunc, x.norb, x.nel, grids_level=x.grid)
                mc.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
                mc.fcisolver.wfnsym = x.irep
                # mc.fix_spin_(ss=x.ispin)
                mc.chkfile = fname
                mc.max_cycle_macro = 600

                if method == 'MC-PDFT': 
                    mc.fcisolver.wfnsym = x.irep
                    mc.fix_spin_(ss=x.ispin)
                    e_pdft = mc.kernel(mo)[0]
                    mo_ss = mc.mo_coeff #SS-CASSCF MOs
                    molden.from_mo(mol, out+'_ss.molden', mc.mo_coeff)
                    #make sure mrh prints out both CASSCF and MC-PDFT dipoles
                    if dmcFLAG == True:
                        print("Working on Analytic MC-PDFT Dipole")
                        dipoles = mc.dip_moment(unit='Debye') 
                        dip_pdft, dip_cas = dipoles[0], dipoles[1]
                        abs_pdft = np.linalg.norm(dip_pdft)
                        abs_cas  = np.linalg.norm(dip_cas)
                    else:
                        print("Analytic MC-PDFT Dipole is ignored")
                
                elif method == 'CMS-PDFT':
                    mc = mc.multi_state(weights,'cms')
                    # mc.fix_spin_(ss=x.ispin)
                    if thresh: 
                        mc.conv_tol = conv_tol
                        mc.conv_tol_grad = conv_tol_grad
                    mc.max_cyc=max_cyc
                    mc.kernel(mo,ci)
                    e_states=mc.e_states.tolist() 
                    mo_sa = mc.mo_coeff #SA-CASSCF MOs
                    molden.from_mo(mol, out+'_sa.molden', mc.mo_coeff)
                    if dmcFLAG:
                        print("Working on Analytic CMS-PDFT Dipole")
                        for m in range(x.iroots):
                            shift=m*4
                            dipoles = mc.dip_moment(state=m, unit=x.unit, origin=origin)
                            abs_cms = np.linalg.norm(dipoles)
                            dip_cms[shift:shift+3] = dipoles
                            dip_cms[shift+3] = abs_cms
                    else:
                        print("Analytic CMS-PDFT Dipole is ignored")
                        dip_cms = np.zeros(4*x.iroots).tolist()
                
                elif method == 'SA-PDFT':
                    mc = mc.state_average_(weights)
                    mc.fcisolver.wfnsym = x.irep
                    mc.fix_spin_(ss=x.ispin)
                    if thresh: 
                        mc.conv_tol = conv_tol
                        mc.conv_tol_grad = conv_tol_grad
                    mc.kernel(mo,ci)
                    e_states=mc.e_states 
                    mo_sa = mc.mo_coeff #SA-CASSCF MOs
                    molden.from_mo(mol, out+'_sa.molden', mc.mo_coeff)
                    if dmcFLAG == True:
                        print("Working on Analytic SA-PDFT Dipole")
                        for m in range(x.iroots):
                            shift=m*4
                            dipoles = mc.dip_moment(state=m, unit='Debye')
                            dipoles = np.array(dipoles)
                            abs_cms = np.linalg.norm(dipoles)
                            dip_cms[shift:shift+3] = dipoles
                            dip_cms[shift+3] = abs_cms
                    else:
                        print("Analytic SA-PDFT Dipole is ignored")
                        dip_cms = np.zeros(4*x.iroots).tolist()

        # ---------------- Numerical Dipoles ---------------------------
        #---------------------------------------------------------------
        if numer:
            for method in numer:
                if method == 'MC-PDFT': 
                    # raise NotImplementedError
                    mo=mo_ss
                elif method == 'CMS-PDFT' or method == 'SA-PDFT':
                    mo=mo_sa

                dip_num = numer_run(dist, x, mol, mo, ci, method, field, ifunc, out, dip_cms)
        else:
            print("Numerical dipole is ignored")
            dip_num = np.zeros((len(field), 4))
        
   
        analytic[k] = [dist, abs_cas, abs_pdft] + dip_cms
        numeric [k] = dip_num
        energy  [k] = [dist, e_casscf, e_pdft] + e_states
    return numeric, analytic, energy, mo, ci

def save_data(x, analyt, numer, dataname=None, data=None, dist=None):
    ''' 
    ENERGIES and ANALYTIC PDMS is a list: [bond [functional [values] ] ]
    NUMERIC PDMS               is a list: [functional [filed [values] ] ]
    where values could be arrays of energies, permanent, or transition dipole moments
    '''
    if data == None or dataname == None: 
        raise ValueError('data/datname was not provided')

    if dataname.upper() == 'ENERGIES' or dataname.upper() =='ANALYTIC PDMS':
        data = list(zip(*data)) #flip dimensions over bonds with functionals

    #set up headers and significant figures
    head=['Distance', 'CASSCF', 'MC-PDFT']
    if dataname.upper() == 'ENERGIES':
        sig = (".2f",)+(".8f",)+(".8f",) + (".8f",)*(x.iroots)
        for method in analyt:
            for j in range(x.iroots): 
                head.extend([f'{method} ({cs(j+1)})'])
    elif dataname.upper() == 'ANALYTIC PDMS':
        sig = (".2f",)+(".5f",)+(".5f",) + (".5f",)*(4*x.iroots)
        for j in range(x.iroots):
            head.extend(['X', 'Y', 'Z', f'ABS ({cs(j+1)})'])
    elif dataname.upper() == 'NUMERIC PDMS':
        for method in numer:
            head = ['Field']
            for i in range(x.iroots): 
                head.extend(['X', 'Y', 'Z', f'ABS ({i+1})']) 
            sig = (".4f",) + (".5f",)*4*x.iroots
    
    for k, ifunc in enumerate(x.ontop):
        table = pdtabulate(list(data[k]), head, sig)
        if dataname.upper() == 'NUMERIC PDMS':
            message = f' \n{dataname} found with {cs(method)}:{cs(ifunc)} at point {dist:.3f}\n'
        else:
            message = f"{dataname} found with {cs(ifunc)}"
        print(message)
        print(table)

        out = 'results_'+x.iname+'_'+ifunc+'.txt'
        action='a' 
        # action='a' if numer else 'w'
        with open(out, action) as f:
            f.write(message)
            f.write(f'{table}\n')

from dataclasses import dataclass, field
from typing import List
@dataclass
class Molecule:
    iname   : str
    nel     : int
    norb    : int
    cas_list: list
    ontop   : List[str] = field(default_factory=lambda: ['tPBE'])
    geom    : str = 'frame'
    init    : float = 3.0
    iroots  : int = 1
    istate  : int = 0
    icharge : int = 0
    isym    : str = "C1"
    irep    : str = "A" 
    ispin   : int = 0
    ibasis  : str = "julccpvdz"
    grid    : int = 9    
    formula : str = "2-point"
    unit    : str = 'Debye'

    def __post_init__(self):
        for func in self.ontop:
            if len(func) > 10:
                raise NotImplementedError('Hybrid functionals were not tested')
            elif func[0] =='f':
                raise NotImplementedError('Fully-translated functionals were not tested')
#--------------------- Set Up Parameters for Dipole Moment Curves--------------------
# Set the bond range
bonds = np.array([1.0])
bonds = np.array([1.0, 1.1])
# bonds = np.array([])
# bonds = np.array([2.1])
inc=0.1
# inc=0.02
# bonds = np.arange(2.0,2.2+inc,inc) # for energy curves
# bonds = np.array([1.6,2.0,2.1,2.2,2.3,2.4,2.5]) # for energy curves
# bonds = np.arange(1.0,3.0+inc,inc) # for energy curves
# bonds = np.arange(3.0,0.9-inc,-inc) # for energy curves

# External XYZ frames
# bonds = np.arange(0,31,1) 
# bonds = np.arange(0,2,1) 

# Set field range
field = np.linspace(1e-3, 1e-2, num=19)
field = np.linspace(1e-3, 1e-2, num=10)
field = np.linspace(1e-5, 1e-2, num=30)
field = np.linspace(2e-3, 1e-2, endpoint=True, num=81)
field = np.linspace(5e-4, 5e-3, endpoint=True, num=46)
field = np.linspace(1e-3, 1e-2, num=2)
field = np.linspace(1e-3, 1e-2, num=1)
# inc= 1e-3
# field = np.arange(inc, 1e-2, inc)
# thresh = 5e-9
conv_tol = 1e-11
conv_tol_grad= 1e-6
thresh = [conv_tol]+[conv_tol_grad]
max_cyc = 100

# Set how dipole moments should be computed
numer  = []
analyt = []
# numer = ['SS-PDFT']
numer = ['CMS-PDFT']
# numer = ['SA-PDFT']
# analyt = ['MC-PDFT','CMS-PDFT']
analyt = ['CMS-PDFT']
# analyt = ['SA-PDFT']
dmcFLAG=False 
dmcFLAG=True

# See class Molecule for the list of variables.
crh_7e7o      = Molecule('crh',   7,7, [10,11,12,13,14,15, 19], init=1.60, ispin=5, ibasis='def2tzvpd')
ch5_2e2o      = Molecule('ch5',   2,2, [29, 35],                init=1.50, iroots=1, ibasis='augccpvdz')
co_10e8o      = Molecule('co',   10,8, [3,4,5,6,7, 8,9,10],     init=1.20, iroots=3, ibasis='augccpvdz',)
h2co_6e6o     = Molecule('h2co',  6,6, [6,7,8,9,10,12],         init=1.20, iroots=2)
butadiene_4e4o= Molecule('butadiene', 4,4, [14,15,16,17],    ibasis='631g*', iroots=2)
spiro_11e10o  = Molecule('spiro',11,10, [35,43,50,51,52,53, 55,76,87,100], iroots=2, icharge=1, ispin=1)
# scans
h2co_6e6o     = Molecule('h2co_scan', 6,6, [6,7,8,9,10,12],  init=1.20, iroots=2)
phenol_10e9o  = Molecule('phenol_scan_10e9o',     10,9, [19,20,23,24,25,26,31,33,34], init=1.3, iroots=3)
phenol_12e11o = Molecule('phenol_scan_12e11o',   12,11, [19,20,21,23,24,25,26,31,33,34,58], iroots=3)
# unit tests
h2o_4e4o      = Molecule('h2o', 4,4, [4,5,8,9], iroots=3, grid=1, isym='c2v', irep='A1', ibasis='aug-cc-pVDZ')
furancat_5e5o = Molecule('furancat', 5,5, [12,17,18,19,20], iroots=3, grid=1, icharge = 1, ispin =1, ibasis='sto-3g')
furan_6e5o    = Molecule('furancat', 6,5, [12,17,18,19,20], iroots=3, grid=1, isym='C2v', irep ='A1', ibasis='sto-3g')

#Select species for which the dipole moment curves will be constructed
species=[phenol_12e11o]
species=[furancat_5e5o]
species=[h2co_6e6o]
species=[h2o_4e4o]

# ---------------------- MAIN DRIVER OVER DISTANCES -------------------
for x in species:
    if x.iname == 'spiro':
        xyz = open(str(dist).zfill(2)+'.xyz').read()
    else:
        xyz = open('geom_'+x.iname+'.xyz').read()
    dip_scan = []
    num_scan = []
    en_scan  = []
    # Get MOs before running a scan
    x.geom = xyz.format(x.init)
    mo, ci = init_guess(x, analyt, numer) if x !=spiro_11e10o else 0 #spiro MOs should be provided manually

    # MOs and CI vectors are taken from the previous point
    bonds = np.sort(bonds)[::-1] #always start from longer bonds
    for i, dist in enumerate(bonds):
        x.geom = xyz.format(dist)
        numeric, analytic, energy, mo, ci = get_dipole(x, field, numer, analyt, mo, ci, dist, dmcFLAG=dmcFLAG)
        dip_scan.append(analytic) 
        en_scan.append(energy) 
        num_scan.append(numeric)
        if numer: save_data(x, analyt, numer, dataname='Numeric PDMs', data=num_scan[i], dist=dist)

    save_data(x, analyt, numer, dataname='Energies', data=en_scan)
    if analyt: save_data(x, analyt, numer, dataname='Analytic PDMs', data=dip_scan)
        