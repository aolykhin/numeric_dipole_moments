from tabulate import tabulate
from pyscf import gto, scf, mcscf, lib
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import os
from pyscf.tools import molden
import copy

os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

        # ------------------ NUMERICAL DIPOLE MOMENTS ----------------------------
def numer_run(x, mol, mo, numer, field, formula, ifunc, out):

    mol.output='num_'+ out
    mol.build()

    dip_num = np.zeros((len(field), 1+4*x.iroot))

    for i, f in enumerate(field[:, 0]): #Set field strength
        au2D = 2.5417464
        dip_num[i][0]=f # the first column is the field column 
        if formula == "2-point":
            disp = [-f, f]
        elif formula == "4-point":
            disp = [-2*f, -f, f, 2*f]
        e = np.zeros((len(disp),x.iroot))

        # Set reference point to be center of charge
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        nuc_charge_center = np.einsum(
            'z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)
        h_field_off = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
        for j in range(3): #Orient field
            for k, v in enumerate(disp): #Select derivative term
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

                # MC-PDFT 
                if numer == 'SS-PDFT':
                    e_mcpdft = mc.kernel(mo)[0]
                    e[k] = e_mcpdft
                # CMS-PDFT 
                elif numer=='CMS-PDFT':
                    weights=[1/x.iroot]*x.iroot #Equal weights only
                    mc=mc.state_interaction(weights,'cms').run(mo)
                    e_cms = mc.e_states.tolist() #List of CMS energies
                    e[k] = e_cms

            for m in range(x.iroot):
                shift=1+m*4 # shift to the next state by 4m columns (x,y,z,mu) and one field column
                if formula == "2-point":
                    dip_num[i,j+shift] = (-1)*au2D*(-e[0][m]+e[1][m])/(2*f)  # 2-point
                    # dip_num[i,j+shift] = (-1)*au2D*(-e[m][0]+e[m][1])/(2*f)  # 2-point
                elif formula == "4-point":
                    dip_num[i,j+shift] = (-1)*au2D*(e[0][m]-8*e[1][m]+8*e[2][m]-e[3][m])/(12*f)  # 4-point
                    # dip_num[i,j+shift] = (-1)*au2D*(e[m][0]-8*e[m][1]+8*e[m][2]-e[m][3])/(12*f)  # 4-point
        
        # Get absolute dipole moment    
        for m in range(x.iroot):
            shift=1+m*4 # shift to the next state by 4m columns (x,y,z,mu) and one field column    
            dip_num[i,3+shift] = np.linalg.norm(dip_num[i,0+shift:3+shift])
    return dip_num

def init_guess(y):
    out = y.iname+'_cas'
    mol = gto.M(atom=y.geom, charge=y.icharge, spin=y.ispin,
                output=y.iname+'_init.log', verbose=4, basis=y.ibasis, symmetry=y.isym)
    mf = scf.RHF(mol)
    mf.run()
    molden.from_mo(mol,'orb_'+y.iname+'_init_hf.molden', mf.mo_coeff)

    # fname = 'orb_'+y.iname+'_init_casscf' #file with SAVED orbitals
    fname = 'orb_'+ out
    cas = mcscf.CASSCF(mf, y.norb, y.nel)
    cas.natorb = True
    cas.chkfile = fname
    cas.fcisolver.wfnsym = y.irep
    if os.path.isfile(fname) == True:
        print('Read orbs from the SAVED calculations')
        mo = lib.chkfile.load(fname, 'mcscf/mo_coeff')
        mo = mcscf.project_init_guess(cas, mo)
    else:
        print('Guess orbs from HF at bond length of %3.5f' % (y.init))
        mo = mcscf.sort_mo(cas, mf.mo_coeff, y.cas_list)
    e_casscf = cas.mc2step(mo)[0]
    cas.analyze()
    mo = cas.mo_coeff
    # molden.from_mo(mol, 'orb_'+y.iname+'_init_casscf', cas.mo_coeff)
    molden.from_mo(mol, out+'.molden', cas.mo_coeff)
    return mo

#-------------Compute energies and dipoles for the given geometry-----------
def get_dipole(x, field, formula, numer, analyt, dist, mo, ontop):
    out = x.iname+'_cas_'+f"{dist:.2f}"
    mol = gto.M(atom=x.geom, charge=x.icharge, spin=x.ispin,
                output=out+'.log', verbose=4, basis=x.ibasis, symmetry=x.isym)
    #HF step
    mf = scf.RHF(mol)
    mf.max_cycle = 1
    mf.run()

    #CASSCF step
    fname = 'orb_'+ out #file with orbitals
    cas = mcscf.CASSCF(mf, x.norb, x.nel)
    cas.natorb = True
    cas.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
    cas.fcisolver.wfnsym = x.irep
    cas.chkfile = fname
    if os.path.isfile(fname) == True:
        print('Read orbs from the previous calculations')
        mo = lib.chkfile.load(fname, 'mcscf/mo_coeff')
    else:
        print('Project orbs from the previous point')
        mo = mcscf.project_init_guess(cas, mo)
    e_casscf = cas.kernel(mo)[0]
    cas.analyze()
    mo = cas.mo_coeff
    molden.from_mo(mol, out+'.molden', cas.mo_coeff)

    #MC-PDFT step
    numeric  = [None]*len(ontop)
    analytic = [None]*len(ontop)
    en_dist  = [None]*len(ontop) # energy array indexed by functional

    for k, ifunc in enumerate(ontop):

        ot='htPBE0' if len(ifunc)>10 else ifunc
        mol.output=x.iname+'_'+ot+'_'+f"{dist:.2f}"+'.log'
        mol.build()
        mf = scf.RHF(mol)
        mf.max_cycle = 1
        mf.run()
        mc = mcpdft.CASSCF(mf, ifunc, x.norb, x.nel, grids_level=9)
        mc.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
        mc.fcisolver.max_cycle = 200
        mc.max_cycle_macro = 200
        mc.fcisolver.wfnsym = x.irep
        e_pdft = mc.kernel(mo)[0]

        #CMS-PDF step
        # if cms_method == true:
        weights=[1/x.iroot]*x.iroot #Equal weights only
        mc = mc.state_interaction(weights,'cms').run(mo)
        e_cms=mc.e_states.tolist() #List of CMS energies
        # print('\nEnergies: ',e_cms)

        # Numerical
        if numer == 'CMS-PDFT' or numer == 'SS-PDFT':
            dip_num = numer_run(x, mol, mo, numer, field, formula, ifunc, out)
        elif bool(numer) == False:
            print("Numerical dipole is ignored")
            dip_num = np.zeros((len(field), 4))
        else:
            raise NotImplementedError
        
        # Analytical
        if analyt == True and len(ifunc) < 10 and ifunc!='ftPBE':
            dipoles = mc.dip_moment(unit='Debye')
            dip_pdft, dip_cas = dipoles[0], dipoles[1]
            abs_pdft = np.linalg.norm(dip_pdft)
            abs_cas  = np.linalg.norm(dip_cas)
        # Skip analytic MC-PDFT for a hybrid translated functional
        else:
            print("Analytical dipole is ignored")
            dip_pdft, dip_cas = np.array([0, 0, 0]), np.array([0, 0, 0])
            abs_pdft = 0
            abs_cas  = 0

        # tmp = np.stack((field[:, 0], dip_num[:, 0], dip_num[:, 1], dip_num[:, 2], dip_num[:, 3]), axis=1)
        # numeric[k]  = tmp.tolist()
        numeric[k]  = dip_num
        analytic[k] = np.concatenate(([dist, e_casscf, e_pdft], dip_cas, [abs_cas], dip_pdft, [abs_pdft]))
        en_dist[k] = [dist, e_casscf, e_pdft] + e_cms
    return numeric, analytic, en_dist, mo

def pdtabulate(df, line1, line2): return tabulate(df, headers=line1, tablefmt='psql', floatfmt=line2)

def run(x, field, formula, numer, analyt, mo, dist, ontop, scan, full, en_scan):
    # Get numeric and analytic dipoel moments over all ontop functionals
    numeric, analytic, en_dist, mo = get_dipole(x, field, formula, numer, analyt, dist, mo, ontop)

    # Save Analytic dipole moments
    for k, ifunc in enumerate(ontop):
        out = 'dmc_'+x.iname+'_'+ifunc+'.txt'
        list_analytic = [analytic[k]]
        if bool(full[k])==False:# first element is zero
            full[k] = list_analytic
        else:
            full[k] = full[k] + list_analytic
        en_scan[k].append(en_dist[k]) 

        # Save numeric dipole moments
        if numer == 'CMS-PDFT' or numer == 'SS-PDFT':
            ot='htPBE0' if len(ifunc)>10 else ifunc
            print("Numeric dipole at the bond length %s found with %s (%s)" %(dist,numer,ot))
            header=['Field',]
            for i in range(3): 
                header=header+['X', 'Y', 'Z',] 
                header.append('ABS ({})'.format(i+1))
            sigfig = (".5f",)+(".6f",)*4*x.iroot
            numer_point = pdtabulate(numeric[k], header, sigfig)
            print(numer_point)
            action='w' if scan==False else 'a'
            with open(out, action) as f:
                f.write('Bond length is %s \n' % dist)
                f.write(numer_point)
                f.write('\n')

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

# Set name for tPBE0 (HMC-PDFT functional)
hybrid = 't'+ mcpdft.hyb('PBE', 0.25, 'average')

#--------------------- Set Up Parameters for Dipole Moment Curves--------------------
# Set the bond range
bonds = np.array([1.2])
# inc=0.05
# bonds = np.arange(2.6,4.0+inc,inc) # for energy curves
# inc=0.1
# bonds = np.arange(1.2,4.0+inc,inc) # for energy curves

# Set field range
field = np.array(np.linspace(1e-2, 1e-3, num=2), ndmin=2).T
field = np.array(np.linspace(1e-2, 1e-3, num=1), ndmin=2).T
# field = np.array(np.linspace(1e-2, 1e-3, num=19), ndmin=2).T
# inc= 5e-3
# field = np.array(np.arange(1e-2, 1e-3+inc, inc), ndmin=2).T

# Set on-top functional
ontop= ['tPBE']
# ontop= ['tPBE','tOPBE']
# ontop= ['tPBE', 'tBLYP', 'tOPBE']
# ontop= [hybrid,'ftPBE']

# Set differentiation technique
formula= "2-point"
# formula = "4-point"

# Set how dipole moments should be computed
numer  = None
analyt = None
# numer = 'SS-PDFT'
numer = 'CMS-PDFT'
# analyt = 'CMS-PDFT'


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
# See class Molecule for the list of variables.
# It's important to provide a molecule-specific cas_list to get initial MOs for CASSCF
crh_7e7o = Molecule('crh_7e7o', geom_crh, 0, 'Coov', 'A1', 5, 'def2tzvpd', 1, 7,7,  1.60, [10,11,12,13,14,15, 19])
ch5_2e2o = Molecule('ch5_2e2o', geom_ch5, 0, 'C1',   'A',  0, 'augccpvdz', 1, 2,2,  1.50, [29, 35])
co_10e8o = Molecule('co_10e8o', geom_co,  0, 'C1',   'A',  0, 'augccpvdz', 3, 10,8, 1.20, [3,4,5,6,7, 8,9,10])

#Select species for which the dipole moment curves will be constructed
# species=[crh_7e7o]
# species=[ch5_2e2o]
species=[co_10e8o]
# species=[crh_7e7o, ch5_2e2o]

# ----------------------Here is the main driver of this script-------------------
full= [0] * len(ontop)
en_scan = [ [] for _ in range(len(ontop)) ] # Empty lists per functional

scan=False if len(bonds)==1 else True #Determine if multiple bonds are passed
for x in species:
    # Get MOs before running a scan
    y=copy.deepcopy(x)
    y.geom=y.geom.format(y.init)
    mo = init_guess(y)

    for i, dist in enumerate(bonds, start=0):
        # Update the bond length in the instance
        if i==0: template=x.geom
        x.geom=template.format(dist)

        run(x, field, formula, numer, analyt, mo, dist, ontop, scan, full, en_scan)
        
        line1 = ['Distance', 'Energy CASSCF', 'Energy MC-PDFT'] + \
                ['X', 'Y', 'Z', 'Dipole CASSCF'] + ['X', 'Y', 'Z', 'Dipole MC-PDFT']
        line2 = (".2f",)+(".8f",)*2+ (".6f",)*8
        
        header_CMS_PDFT=[]
        for i in range(x.iroot): 
            str='CMS-PDFT ({})'.format(i+1)
            header_CMS_PDFT.append(str)
        line3 = ['Distance', 'CASSCF', 'MC-PDFT'] + header_CMS_PDFT
        line4 = (".2f",)+(".8f",)*(2+x.iroot)

        for k, ifunc in enumerate(ontop):
            print("Analytic dipole moments found with %s" %ifunc)
            out = x.iname+'_'+ifunc+'.txt'

            # print(full[k])
            # print(cms_full[k])

            full_table = pdtabulate(full[k], line1, line2)
            cms_table = pdtabulate(en_scan[k], line3, line4)
            print(full_table)
            print(cms_table)
            action='w' if numer==False else 'a'
            with open(out, action) as f:
                f.write("The on-top functional is %s \n" %ifunc)
                f.write(full_table)
                f.write(cms_table)
