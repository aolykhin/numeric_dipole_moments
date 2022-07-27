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

def cs(text): return fg('light_green')+text+attr('reset')

class Molecule:
    def __init__(self, iname, geom, icharge, isym, irep, ispin, ibasis, iroot,
                 nel, norb, init, cas_list):
        self.iname    = iname
        self.geom     = geom
        self.icharge  = icharge
        self.isym     = isym
        self.irep     = irep
        self.ispin    = ispin
        self.ibasis   = ibasis
        self.iroot    = iroot
        self.nel      = nel
        self.norb     = norb
        self.init     = init #distance at which initial guess is calculated
        self.cas_list = cas_list

# ------------------ NUMERICAL DIPOLE MOMENTS ----------------------------
def numer_run(dist, x, mol, mo_zero, method, field, formula, ifunc, out, dip_cms):
    global thresh
    au2D = 2.5417464
    # Set reference point to be center of charge
    mol.output='num_'+ out
    mol.build()
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nuc_charge_center = np.einsum(
        'z,zx->x', charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    h_field_off = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')

    dip_num = np.zeros((len(field), 1+4*x.iroot))# 1st column is the field column
    for i, f in enumerate(field): # Over field strengths
        dip_num[i][0]=f # the first column is the field column 
        if formula == "2-point":
            disp = [-f, f]
        elif formula == "4-point":
            disp = [-2*f, -f, f, 2*f]
        e = np.zeros((len(disp),x.iroot))
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

                mc = mcpdft.CASSCF(mf_field, ifunc, x.norb, x.nel, grids_level=9)
                mc.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
                mc.fcisolver.wfnsym = x.irep
                # mc.natorb = True
                print('j=%s and k=%s' %(j,k))
                if i==0: #First field starts with zero-field MOs
                    mc.max_cycle_macro = 200
                    mo=mo_zero 
                else: # Try MOs from the previous filed/point
                    mc.max_cycle_macro = 5
                    mo=mo_field[j][k]
                # if the threshold is too tight the active space is unstable
                mc.conv_tol = mc.conv_tol_sarot = thresh 
                # MC-PDFT 
                if method == 'SS-PDFT':
                    e_mcpdft = mc.kernel(mo)[0]
                    if mc.converged==False: 
                        mc.max_cycle_macro = 200
                        e_mcpdft = mc.kernel(mo_zero)[0]
                    e[k] = e_mcpdft
                # CMS-PDFT 
                elif method == 'CMS-PDFT':
                    weights=[1/x.iroot]*x.iroot #Equal weights only
                    mc=mc.state_interaction(weights,'cms').run(mo)
                    if mc.converged==False:
                        mc.max_cycle_macro = 200
                        mc.run(mo_zero)
                    e_cms = mc.e_states.tolist() #List of CMS energies
                    e[k] = e_cms
                else:
                    raise NotImplementedError
                mo_field[j][k] = mc.mo_coeff #save MOs for the next stencil point k and axis j 

            for m in range(x.iroot): # Over CMS states
                shift=m*4 # shift to the next state by 4m columns (x,y,z,mu)
                if formula == "2-point":
                    dip_num[i,1+j+shift] = (-1)*au2D*(-e[0][m]+e[1][m])/(2*f)
                elif formula == "4-point":
                    dip_num[i,1+j+shift] = (-1)*au2D*(e[0][m]-8*e[1][m]+8*e[2][m]-e[3][m])/(12*f)
        
        # Get absolute dipole moment    
        for m in range(x.iroot):
            shift=m*4 # shift to the next state by 4m columns (x,y,z,mu)    
            dip_num[i,4+shift] = np.linalg.norm(dip_num[i,1+shift:4+shift])
    
    #Save covergence plots
    num_conv_plot(x, field, dip_num, dist, method, dip_cms)
    return dip_num

def num_conv_plot(x, field, dip_num, dist, method, dip_cms):
    y1=[None]*x.iroot
    x1=field.flatten()
    # x1=field
    for m in range(x.iroot):
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

def init_guess(y):
    global thresh
    out = y.iname+'_cas'
    mol = gto.M(atom=y.geom, charge=y.icharge, spin=y.ispin,
                output=y.iname+'_init.log', verbose=4, basis=y.ibasis, symmetry=y.isym)
    mf = scf.RHF(mol).run()
    molden.from_mo(mol,'orb_'+y.iname+'_init_hf.molden', mf.mo_coeff)
    # nocc = mol.nelectron // 2
    # loc = lo.PM(mol, mf.mo_coeff[:,:nocc]).kernel()
    # molden.from_mo(mol,'orb_'+y.iname+'_init_hf_loc.molden', loc)

    # fname = 'orb_'+y.iname+'_init_casscf' #file with SAVED orbitals
    fname = 'orb_'+ out
    cas = mcscf.CASSCF(mf, y.norb, y.nel)
    cas.natorb = True
    cas.chkfile = fname
    cas.fcisolver.wfnsym = y.irep
    cas.conv_tol = thresh 
    if os.path.isfile(fname) == True:
        print('Read orbs from the SAVED calculations')
        mo = lib.chkfile.load(fname, 'mcscf/mo_coeff')
        mo = mcscf.project_init_guess(cas, mo)
    else:
        print('Guess orbs from HF at bond length of %3.5f' % (y.init))
        mo = mcscf.sort_mo(cas, mf.mo_coeff, y.cas_list)
    e_casscf = cas.kernel(mo)[0]
    cas.analyze()
    mo = cas.mo_coeff
    molden.from_mo(mol, out+'_init_cas.molden', cas.mo_coeff)
    return mo

#-------------Compute energies and dipoles for the given geometry-----------
def get_dipole(x, field, formula, numer, analyt, mo, dist, ontop):
    global thresh
    out = x.iname+'_'+f"{dist:.2f}"
    mol = gto.M(atom=x.geom,charge=x.icharge,spin=x.ispin,output=out+'.log',
                verbose=4, basis=x.ibasis, symmetry=x.isym)
    #HF step
    mf = scf.RHF(mol).set(max_cycle = 1).run()

    #CASSCF step
    fname = 'orb_ss_cas_'+ out #file with orbitals
    cas = mcscf.CASSCF(mf, x.norb, x.nel)
    # cas.natorb = True
    cas.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
    cas.fcisolver.wfnsym = x.irep
    cas.chkfile = fname
    cas.conv_tol = thresh 
    # if os.path.isfile(fname) == True:
    #     print('Read orbs from the previous calculations')
    #     mo = lib.chkfile.load(fname, 'mcscf/mo_coeff')
    # else:
    #     print('Project orbs from the previous point')
    #     mo = mcscf.project_init_guess(cas, mo)
    e_casscf = cas.kernel(mo)[0]
    cas.analyze()
    mo = cas.mo_coeff
    molden.from_mo(mol, out+'.molden', cas.mo_coeff)

    #MC-PDFT step
    numeric  = [None]*len(ontop)
    analytic = [None]*len(ontop)
    en_dist  = [None]*len(ontop) # energy array indexed by functional

    for k, ifunc in enumerate(ontop): # Over on-top functionals

        ot='htPBE0' if len(ifunc)>10 else ifunc
        mol.output=x.iname+'_'+ot+'_'+f"{dist:.2f}"+'.log'
        mol.build()

        #-------------------- MC-PDFT Energy ---------------------------
        mc = mcpdft.CASSCF(mf, ifunc, x.norb, x.nel, grids_level=9)
        mc.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
        mc.fcisolver.max_cycle = 200
        mc.max_cycle_macro = 200
        mc.fcisolver.wfnsym = x.irep
        e_pdft = mc.kernel(mo)[0]
        mo_ss = mc.mo_coeff #SS-CASSCF MOs
        molden.from_mo(mol, out+'_ss_cas.molden', mc.mo_coeff)

        #Initialize dipole moments to zero
        dip_cms = np.zeros(4*x.iroot).tolist()
        abs_pdft = 0
        abs_cas  = 0

        # ---------------- Analytic MC-PDFT Dipole ----------------------
        if analyt == None:
            print("Analytic MC-PDFT dipole is ignored")
            abs_pdft = 0
            abs_cas  = 0
        else:
            for method in analyt:
                if method == 'MC-PDFT' and len(ifunc) < 10 and ifunc!='ftPBE':
                    #make sure mrh prints out both CASSCF and MC-PDFT dipoles
                    dipoles = mc.dip_moment(unit='Debye') 
                    dip_pdft, dip_cas = dipoles[0], dipoles[1]
                    abs_pdft = np.linalg.norm(dip_pdft)
                    abs_cas  = np.linalg.norm(dip_cas)

        #-------------------- CMS-PDFT Energy ---------------------------
        # if cms_method == true:
        weights=[1/x.iroot]*x.iroot #Equal weights only
        mc = mc.state_interaction(weights,'cms').run(mo)
        e_cms=mc.e_states.tolist() #List of CMS energies
        mo_sa = mc.mo_coeff #SA-CASSCF MOs
        molden.from_mo(mol, out+'_sa_cas.molden', mc.mo_coeff)
        # print('\nEnergies: ',e_cms)

        # ---------------- Analytic CMS-PDFT Dipole----------------------
        if analyt == None:
            print("Analytic CMS-PDFT dipole is ignored")
            dip_cms = np.zeros(4*x.iroot).tolist()
        else:
            for method in analyt:
                if method == 'CMS-PDFT' and len(ifunc) < 10 and ifunc!='ftPBE':
                    for m in range(x.iroot):
                        shift=m*4
                        dipoles = mc.dip_moment(state=m, unit='Debye')
                        abs_cms = np.linalg.norm(dipoles)
                        dip_cms[shift:shift+3] = dipoles
                        dip_cms[shift+3] = abs_cms
        
        # ---------------- Numerical Dipoles ---------------------------
        #---------------------------------------------------------------
        if numer == None:
            print("Numerical dipole is ignored")
            dip_num = np.zeros((len(field), 4))
        else:
            for method in numer:
                if method == 'MC-PDFT': 
                    mo=mo_ss
                elif method == 'CMS-PDFT':
                    mo=mo_sa
                dip_num = numer_run(dist, x, mol, mo, method, field, formula, ifunc, out, dip_cms)
            
        analytic[k] = [dist, abs_cas, abs_pdft] + dip_cms
        numeric [k] = dip_num
        en_dist [k] = [dist, e_casscf, e_pdft] + e_cms
    return numeric, analytic, en_dist, mo

def pdtabulate(df, line1, line2): return tabulate(df, headers=line1, tablefmt='psql', floatfmt=line2)

# Get dipoles & energies for a fixed distance
def run(x, field, formula, numer, analyt, mo, dist, ontop, scan, dip_scan, en_scan):
    numeric, analytic, en_dist, mo = get_dipole(x, field, formula, numer, analyt, mo, dist, ontop)

    # Accumulate analytic dipole moments and energies
    for k, ifunc in enumerate(ontop):
        out = 'dmc_'+x.iname+'_'+ifunc+'.txt'
        dip_scan[k].append(analytic[k]) 
        en_scan[k].append(en_dist[k]) 

        # Print & save numeric dipole moments
        if numer != None:
            for method in numer:
                ot='htPBE0' if len(ifunc)>10 else ifunc
                print("Numeric dipole at the bond length %s found with %s (%s)" \
                    %(cs(str(dist)),cs(method),cs(ot)))
                header=['Field',]
                for i in range(x.iroot): 
                    header=header+['X', 'Y', 'Z',] 
                    header.append('ABS ({})'.format(str(i+1)))
                sigfig = (".4f",)+(".4f",)*4*x.iroot
                numer_point = pdtabulate(numeric[k], header, sigfig)
                print(numer_point)
                action='w' if scan==False else 'a'
                with open(out, action) as f:
                    f.write("Numeric dipole at %.3f ang found with %s (%s)\n" \
                        %(dist,method,ot))
                    # f.write("Analytic dipoles are %.4f\n" \
                    #     %(dist,method,ot))
                    f.write(numer_point)
                    f.write('\n')



#--------------------- Set Up Parameters for Dipole Moment Curves--------------------
# Set the bond range
bonds = np.array([1.2])
# bonds = np.array([2.1])
# bonds = np.array([1.2,1.3])
# inc=0.05
# inc=0.1
# bonds = np.arange(1.2,3.0+inc,inc) # for energy curves

# Set field range
# field = np.linspace(1e-3, 1e-2, num=1)
field = np.linspace(1e-3, 1e-2, num=2)
field = np.linspace(1e-3, 1e-2, num=3)
field = np.linspace(1e-3, 1e-2, num=19)
# field = np.array([0.0070])
# inc= 1e-3
# field = np.arange(inc, 1e-2, inc)
thresh = 5e-9

# Set name for tPBE0 (HMC-PDFT functional)
hybrid = 't'+ mcpdft.hyb('PBE', 0.25, 'average')
# Set on-top functional
ontop= ['tPBE']
# ontop= ['tPBE','tOPBE']
# ontop= ['tPBE', 'tBLYP', 'tOPBE']
# ontop= [hybrid,'ftPBE']

# Set differentiation technique
formula= "2-point"
# formula = "4-point"

# #!!!!!!!!!!!!!
# field = np.linspace(1e-3, 1e-2, num=19)
# inc=0.1
# bonds = np.arange(1.2,3.0+inc,inc) # for energy curves


# Set how dipole moments should be computed
numer  = None
analyt = None
# numer = ['SS-PDFT']
# numer = ['CMS-PDFT']
analyt = ['MC-PDFT','CMS-PDFT']
# analyt = ['CMS-PDFT']


# List of molecules and molecular parameters
geom_ch5 = '''
C         0.000000000     0.000000000     0.000000000
F         0.051646177     1.278939741    -0.461155429
Cl       -1.493022429    -0.775046725    -0.571629295
Br        1.591285348    -0.956704103    -0.592135206
H         0.000000000     0.000000000     {}
'''
geom_crh = '''
Cr        0.000000000     0.000000000     0.000000000
H         0.000000000     0.000000000     {}
'''
geom_co= '''
C         0.000000000     0.000000000     0.000000000
O         0.000000000     0.000000000     {}
'''
geom_h2co= '''
H       -0.000000000      0.950627350     -0.591483790
H       -0.000000000     -0.950627350     -0.591483790
C        0.000000000      0.000000000      0.000000000
O        0.000000000      0.000000000     {}
'''
# GS equilibrium CMS-PDFT (tPBE) (6e,6o) / julccpvdz
# O        0.000000000      0.000000000      1.214815430 

# See class Molecule for the list of variables.
crh_7e7o = Molecule('crh_7e7o', geom_crh, 0, 'Coov', 'A1', 5, 'def2tzvpd', 1, 7,7,  1.60, [10,11,12,13,14,15, 19])
ch5_2e2o = Molecule('ch5_2e2o', geom_ch5, 0, 'C1',   'A',  0, 'augccpvdz', 1, 2,2,  1.50, [29, 35])
co_10e8o = Molecule('co_10e8o', geom_co,  0, 'C1',   'A',  0, 'augccpvdz', 3, 10,8, 1.20, [3,4,5,6,7, 8,9,10])
h2co_6e6o= Molecule('h2co_6e6o',geom_h2co,0, 'C1',   'A',  0, 'julccpvdz', 2, 6,6,  1.20, [6,7,8,9,10,12])

#Select species for which the dipole moment curves will be constructed
# species=[crh_7e7o]
# species=[ch5_2e2o]
# species=[co_10e8o]
species=[h2co_6e6o]

# ---------------------- MAIN DRIVER OVER DISTANCES -------------------
dip_scan = [ [] for _ in range(len(ontop)) ]
en_scan  = [ [] for _ in range(len(ontop)) ] # Empty lists per functional
scan=False if len(bonds)==1 else True #Determine if multiple bonds are passed
#In case if numer and analyt are not supplied
exec('try:numer\nexcept:numer=None')
exec('try:analyt\nexcept:analyt=None')

for x in species:
    # Get MOs before running a scan
    y=copy.deepcopy(x)
    y.geom=y.geom.format(y.init)
    mo = init_guess(y)

    for i, dist in enumerate(bonds, start=0):
        # Update the bond length in the instance
        if i==0: template=x.geom
        x.geom=template.format(dist)

        run(x, field, formula, numer, analyt, mo, dist, ontop, scan, dip_scan, en_scan)
        
        dip_head = ['Distance','CASSCF','MC-PDFT']
        for j in range(x.iroot): 
            dip_head=dip_head+['X', 'Y', 'Z',] 
            dip_head.append('ABS ({})'.format(cs(str(j+1))))
        dip_sig = (".2f",)+(".4f",)*(2+4*x.iroot)
        
        en_head=['Distance', 'CASSCF', 'MC-PDFT']
        for jj in range(x.iroot): 
            line='CMS-PDFT ({})'.format(cs(str(jj+1)))
            en_head.append(line)
        en_sig = (".2f",)+(".8f",)*(2+x.iroot)

        for k, ifunc in enumerate(ontop):
            out = x.iname+'_'+ifunc+'.txt'
            if analyt!=None:
                print("Analytic dipole moments found with %s" %cs(ifunc))
                dip_table = pdtabulate(dip_scan[k], dip_head, dip_sig)
                print(dip_table)

            print("Energies found with %s" %cs(ifunc))
            en_table = pdtabulate(en_scan[k], en_head, en_sig)
            print(en_table)
            action='w' if numer==None else 'a'
            with open(out, action) as f:
                f.write("The on-top functional is %s \n" %(ifunc))
                f.write(en_table)
                f.write('\n')
                if analyt!=None:
                    f.write(dip_table)
                    f.write('\n')
