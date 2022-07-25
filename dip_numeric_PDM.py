from tabulate import tabulate
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


os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

def cs(text): return fg('light_green')+text+attr('reset')

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
            for k, v in enumerate(disp): #Select point on a stencil
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
                if numer[0] == 'SS-PDFT':
                    e_mcpdft = mc.kernel(mo)[0]
                    e[k] = e_mcpdft
                # CMS-PDFT 
                elif numer[0]=='CMS-PDFT':
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
    # nocc = mol.nelectron // 2
    # loc = lo.PM(mol, mf.mo_coeff[:,:nocc]).kernel()
    # molden.from_mo(mol,'orb_'+y.iname+'_init_hf_loc.molden', loc)

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
    molden.from_mo(mol, out+'_init_cas.molden', cas.mo_coeff)
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

        #Initialize dipole moments to zero
        dip_cms = np.zeros(4*x.iroot).tolist()
        abs_pdft = 0
        abs_cas  = 0

        # Analytic MC-PDFT
        for method in analyt:
            if method == 'MC-PDFT' and len(ifunc) < 10 and ifunc!='ftPBE':
                dipoles = mc.dip_moment(unit='Debye')
                dip_pdft, dip_cas = dipoles[0], dipoles[1]
                abs_pdft = np.linalg.norm(dip_pdft)
                abs_cas  = np.linalg.norm(dip_cas)
        # else:
        #     print("Analytical dipole is ignored")
        #     dip_pdft = np.array([0, 0, 0])
        #     dip_cas  = np.array([0, 0, 0])
        #     abs_pdft = 0
        #     abs_cas  = 0

        #CMS-PDF step
        # if cms_method == true:
        weights=[1/x.iroot]*x.iroot #Equal weights only
        mc = mc.state_interaction(weights,'cms').run(mo)
        e_cms=mc.e_states.tolist() #List of CMS energies
        # print('\nEnergies: ',e_cms)

        # Analytic CMS-PDFT dipole
        for method in analyt:
            if method == 'CMS-PDFT' and len(ifunc) < 10 and ifunc!='ftPBE':

                # dipoles = mc.dip_moment(unit='Debye')
                # dip_cms = dipoles[0]
                # abs_cms = np.linalg.norm(dip_cms)
                # dip_cms = np.array([0, 0, 0])
                abs_cms = 0
        # else:
        #     ("Analytical dipole is ignored")
        #     dip_cms = np.array([0, 0, 0])
        #     abs_cms = 0
        
        # Numerical dipoles
        if numer[0] == 'CMS-PDFT' or numer[0] == 'SS-PDFT':
            dip_num = numer_run(x, mol, mo, numer, field, formula, ifunc, out)
        elif numer[0] == None:
        # elif bool(numer[0]) == False:
            print("Numerical dipole is ignored")
            dip_num = np.zeros((len(field), 4))
        else:
            raise NotImplementedError

        analytic[k] = [dist, abs_cas, abs_pdft] + dip_cms
        numeric [k] = dip_num
        en_dist [k] = [dist, e_casscf, e_pdft] + e_cms
    return numeric, analytic, en_dist, mo

def pdtabulate(df, line1, line2): return tabulate(df, headers=line1, tablefmt='psql', floatfmt=line2)

# Get dipoles & energies for a fixed distance
def run(x, field, formula, numer, analyt, mo, dist, ontop, scan, dip_scan, en_scan):
    numeric, analytic, en_dist, mo = get_dipole(x, field, formula, numer, analyt, dist, mo, ontop)

    # Accumulate analytic dipole moments and energies
    for k, ifunc in enumerate(ontop):
        out = 'dmc_'+x.iname+'_'+ifunc+'.txt'
        dip_scan[k].append(analytic[k]) 
        en_scan[k].append(en_dist[k]) 

        # Print & save numeric dipole moments
        if numer[0] == 'CMS-PDFT' or numer[0] == 'SS-PDFT':
            ot='htPBE0' if len(ifunc)>10 else ifunc
            #! Needs to be changed if multiple numrs are used
            print("Numeric dipole at the bond length %s found with %s (%s)" \
                %(cs(str(dist)),cs(numer[0]),cs(ot)))
            header=['Field',]
            for i in range(x.iroot): 
                header=header+['X', 'Y', 'Z',] 
                header.append('ABS ({})'.format(cs(str(i+1))))
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
inc=0.1
# bonds = np.arange(1.2,3.0+inc,inc) # for energy curves

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
numer  = [None]
analyt = [None]
# numer = ['SS-PDFT']
numer = ['CMS-PDFT']
# analyt = ['MC-PDFT','CMS-PDFT']

# from pyscf import gto, symm
# import basis_set_exchange
# julccpvdz = {
#         'H' : gto.load(basis_set_exchange.api.get_basis('jul-cc-pV(D+d)Z', elements='H', fmt='nwchem'), 'H'),
#         'C' : gto.load(basis_set_exchange.api.get_basis('jul-cc-pV(D+d)Z', elements='C', fmt='nwchem'), 'C'),
#         'N' : gto.load(basis_set_exchange.api.get_basis('jul-cc-pV(D+d)Z', elements='N', fmt='nwchem'), 'O'),
#         'O' : gto.load(basis_set_exchange.api.get_basis('jul-cc-pV(D+d)Z', elements='O', fmt='nwchem'), 'N'),
#         'F' : gto.load(basis_set_exchange.api.get_basis('jul-cc-pV(D+d)Z', elements='F', fmt='nwchem'), 'F'),
#         }
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
H       -0.000000000      0.950146000     -0.591726000
H       -0.000000000     -0.950146000     -0.591726000
C        0.000000000      0.000000000      0.000000000
O        0.000000000      0.000000000      {}
'''
# H        0.000000000      0.942900000    -0.587600000
# H        0.000000000     -0.942900000    -0.587600000
# C        0.000000000      0.000000000     0.000000000
# O        0.000000000      0.000000000     {}
# O        0.000000000      0.000000000      1.215152000
# See class Molecule for the list of variables.
# It's important to provide a molecule-specific cas_list to get initial MOs for CASSCF
crh_7e7o = Molecule('crh_7e7o', geom_crh, 0, 'Coov', 'A1', 5, 'def2tzvpd', 1, 7,7,  1.60, [10,11,12,13,14,15, 19])
ch5_2e2o = Molecule('ch5_2e2o', geom_ch5, 0, 'C1',   'A',  0, 'augccpvdz', 1, 2,2,  1.50, [29, 35])
co_10e8o = Molecule('co_10e8o', geom_co,  0, 'C1',   'A',  0, 'augccpvdz', 3, 10,8, 1.20, [3,4,5,6,7, 8,9,10])
h2co_8e8o= Molecule('h2co_8e8o',geom_h2co,0, 'C1',   'A',  0, 'julccpvdz', 2, 8,8,  1.20, [3,6,7,8,9,10,12,15])
# h2co_8e8o= Molecule('h2co_8e8o',geom_h2co,0, 'C1',   'A',  0, 'julccpvdz', 3, 8,8,  1.20, [3,6,7,8,9,10,12,15])
# h2co_8e7o= Molecule('h2co_8e7o',geom_h2co,0, 'C1',   'A',  0, julccpvdz, 2, 8,7,  1.20, [4,5,6,7,8,9,10])

#Select species for which the dipole moment curves will be constructed
# species=[crh_7e7o]
# species=[ch5_2e2o]
# species=[co_10e8o]
species=[h2co_8e8o]
# species=[crh_7e7o, ch5_2e2o]

# ---------------------- MAIN DRIVER OVER DISTANCES -------------------
dip_scan = [ [] for _ in range(len(ontop)) ]
en_scan  = [ [] for _ in range(len(ontop)) ] # Empty lists per functional

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

        run(x, field, formula, numer, analyt, mo, dist, ontop, scan, dip_scan, en_scan)
        
        dip_head = ['Distance','Dipole CASSCF','Dipole MC-PDFT']
        for i in range(x.iroot): 
            dip_head=dip_head+['X', 'Y', 'Z',] 
            dip_head.append('ABS ({})'.format(cs(str(i+1))))
        dip_sig = (".2f",)+(".6f",)*(2+4*x.iroot)
        
        en_head=['Distance', 'CASSCF', 'MC-PDFT']
        for i in range(x.iroot): 
            line='CMS-PDFT ({})'.format(cs(str(i+1)))
            en_head.append(line)
        en_sig = (".2f",)+(".8f",)*(2+x.iroot)

        for k, ifunc in enumerate(ontop):
            out = x.iname+'_'+ifunc+'.txt'
            if analyt[0]!=None:
                print("Analytic dipole moments found with %s" %cs(ifunc))
                # print(full[k])
                # print(cms_full[k])
                dip_table = pdtabulate(dip_scan[k], dip_head, dip_sig)
                print(dip_table)

            print("Energies found with %s" %cs(ifunc))
            en_table = pdtabulate(en_scan[k], en_head, en_sig)
            print(en_table)
            action='w' if numer==False else 'a'
            with open(out, action) as f:
                f.write("The on-top functional is %s \n" %cs(ifunc))
                f.write(en_table)
                if analyt[0]!=None:
                    f.write(dip_table)
