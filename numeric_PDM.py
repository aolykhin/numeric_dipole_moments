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

# os.environ['OMP_NUM_THREADS'] = "4"
# os.environ['MKL_NUM_THREADS'] = "4"
# os.environ['OPENBLAS_NUM_THREADS'] = "4"

# List of molecules and molecular parameters
geom_OH_phenol= '''
C       -2.586199811     -1.068328539     -2.343227944
C       -1.383719126     -0.571709401     -1.839139664
C       -3.598933171     -1.486733966     -1.476306928
H       -0.580901222     -0.240090937     -2.507686505
H       -4.543398025     -1.876891664     -1.874465443
C       -1.192810178     -0.493086491     -0.452249284
C       -3.398663509     -1.403899151     -0.097309120
H       -4.185490834     -1.729852747      0.594389093
C       -2.200418093     -0.909124041      0.423137737
H       -2.045275617     -0.844707274      1.509797436
H       -2.734632202     -1.130309287     -3.429202743
O        0.000000000      0.000000000      0.000000000
H        0.000000000      0.000000000      {}
'''
geom_phenol= '''
C        1.214793846     -1.192084882      0.000000000
C       -0.181182920     -1.223420370      0.000000000
C        1.885645321      0.032533975      0.000000000
H       -0.703107911     -2.176177034      0.000000000
H        2.971719251      0.056913772      0.000000000
C       -0.906309250     -0.030135294      0.000000000
C        1.160518991      1.225819051      0.000000000
H        1.682443982      2.178575715      0.000000000
C       -0.235457775      1.194483563      0.000000000
H       -0.799607233      2.122861284      0.000000000
H        1.778943305     -2.120462603      0.000000000
O       -2.345946581     -0.062451754      0.000000000
H       -2.680887890      0.843761215      0.000000000
'''
geom_phenol_12e11o= '''
C       -2.622730874     -1.013542229     -2.331647466
C       -1.406745441     -0.543467761     -1.836239389
C       -3.642230632     -1.407856573     -1.459091122
H       -0.600882859     -0.232144838     -2.511037202
H       -4.597665462     -1.776495891     -1.851638458
C       -1.204492019     -0.465471372     -0.448760082
C       -3.431148118     -1.326314087     -0.081269194
H       -4.220682665     -1.631881105      0.616909336
C       -2.218870662     -0.857801984      0.431663464
H       -2.057527810     -0.795466793      1.517843236
H       -2.777706083     -1.073150292     -3.417047881
O        0.000000000      0.000000000      0.000000000
H        0.000000000      0.000000000      {}
'''
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
geom_furan= '''
C        0.000000000     -0.965551055     -2.020010585
C        0.000000000     -1.993824223     -1.018526668
C        0.000000000     -1.352073201      0.181141565
O        0.000000000      0.000000000      0.000000000
C        0.000000000      0.216762264     -1.346821565
H        0.000000000     -1.094564216     -3.092622941
H        0.000000000     -3.062658055     -1.175803180
H        0.000000000     -1.688293885      1.206105691
H        0.000000000      1.250242874     -1.655874372
'''
geom_furan_shifted= '''
C        0.331781886     -2.024738405     -2.267259682
C        0.331781886     -3.053011573     -1.265775765
C        0.331781886     -2.411260551     -0.066107532
O        0.331781886     -1.059187350     -0.247249097
C        0.331781886     -0.842425086     -1.594070662
H        0.331781886     -2.153751566     -3.339872038
H        0.331781886     -4.121845405     -1.423052277
H        0.331781886     -2.747481235      0.958856594
H        0.331781886      0.191055524     -1.903123469
'''
geom_furan_rotated= '''
C        0.331781886     -2.024738405     -2.267259682
C        0.499026345     -3.383644209     -1.836307192
C        0.191974102     -3.399649555     -0.510972897
O       -0.154080920     -2.150654420     -0.085330697
C       -0.063190622     -1.326368167     -1.168476732
H        0.485859613     -1.627783839     -3.260148794
H        0.805962091     -4.228701792     -2.435314531
H        0.169185620     -4.174289090      0.239367081
H       -0.308755462     -0.290890412     -0.992182835
'''
geom_h2o='''
O  0.00000000   0.08111156   0.00000000
H  0.78620605   0.66349738   0.00000000
H -0.78620605   0.66349738   0.00000000
'''


def cs(text): return fg('light_green')+str(text)+attr('reset')

def pdtabulate(df, line1, line2): return tabulate(df, headers=line1, tablefmt='psql', floatfmt=line2)

# ------------------ NUMERICAL DIPOLE MOMENTS ----------------------------
def numer_run(dist, x, mol, mo_zero, ci_zero, method, field, formula, ifunc, out, dip_cms):
    # Set reference point to be center of charge
    mol.output='num_'+ out
    mol.build()
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nuc_charge_center = np.einsum(
        'z,zx->x', charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    h_field_off = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')

    dip_num = np.zeros((len(field), 1+4*x.iroots))# 1st column is the field column
    for i, f in enumerate(field): # Over field strengths
        dip_num[i][0]=f # the first column is the field column 
        if formula == "2-point":
            disp = [-f, f]
        elif formula == "4-point":
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
                # mc.natorb = True
                print('j=%s and k=%s' %(j,k))
                if i==0: #First field starts with zero-field MOs
                    mc.max_cycle_macro = 600
                    mo=mo_zero 
                else: # Try MOs from the previous filed/point
                    mc.max_cycle_macro = 5
                    mo=mo_field[j][k]
                # if the threshold is too tight the active space is unstable
                if thresh: mc.conv_tol = thresh
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
                if formula == "2-point":
                    dip_num[i,1+j+shift] = (-1)*nist.AU2DEBYE*(-e[0][m]+e[1][m])/(2*f)
                elif formula == "4-point":
                    dip_num[i,1+j+shift] = (-1)*nist.AU2DEBYE*(e[0][m]-8*e[1][m]+8*e[2][m]-e[3][m])/(12*f)
        
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
    out = y.iname+'_cas'
    xyz = open('00.xyz').read() if y.geom == 'frames' else y.geom
    mol = gto.M(atom=xyz, charge=y.icharge, spin=y.ispin,
                output=y.iname+'_init.log', verbose=4, basis=y.ibasis, symmetry=y.isym)
    weights=[1/x.iroots]*x.iroots
    mf = scf.RHF(mol).run()
    molden.from_mo(mol,'orb_'+y.iname+'_init_hf.molden', mf.mo_coeff)

    fname = 'orb_'+ out
    cas = mcscf.CASSCF(mf, y.norb, y.nel)
    # cas.natorb = True
    cas.chkfile = fname
    cas.fcisolver.wfnsym = y.irep
    cas.fix_spin_(ss=y.ispin) 
    if thresh: cas.conv_tol = thresh 

    print(f'Guess MOs from HF at {y.init:3.5f} ang')
    mo = mcscf.sort_mo(cas, mf.mo_coeff, y.cas_list)
    # cas.kernel(mo)
    # mo = cas.mo_coeff

    if 'MC-PDFT' in (analyt + numer):
        # cas.fcisolver.wfnsym = x.irep
        # cas.fix_spin_(ss=x.ispin) 
        cas.kernel(mo)
        mo_ss=cas.mo_coeff
        # cas.analyze()
        # molden.from_mo(mol, out+'_ss.molden', cas.mo_coeff)

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
def get_dipole(x, field, formula, numer, analyt, mo, ci, dist, ontop, dmcFLAG=True):
    out = x.iname+'_'+f"{dist:.2f}"
    xyz = open(str(dist).zfill(2)+'.xyz').read() if x.geom == 'frames' else x.geom
    mol = gto.M(atom=xyz,charge=x.icharge,spin=x.ispin,output=out+'.log',
                verbose=4, basis=x.ibasis, symmetry=x.isym)
    
    #Determine origin
    mass = mol.atom_mass_list()
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    # mol.set_common_orig_(mass_center)

    weights=[1/x.iroots]*x.iroots
    
    #HF step
    mf = scf.RHF(mol).set(max_cycle = 1).run()

    #SS/SA-CASSCF step
    fname = 'orb_'+ out #file with orbitals
    cas = mcscf.CASSCF(mf, x.norb, x.nel)
    # cas.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
    cas.chkfile = fname
    if thresh: cas.conv_tol = thresh
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
    numeric  = [None]*len(ontop)
    analytic = [None]*len(ontop)
    en_dist  = [None]*len(ontop) # energy array indexed by functional

    if x.icharge !=0: origin = "Charge_center"

    for k, ifunc in enumerate(ontop): # Over on-top functionals

        ot='htPBE0' if len(ifunc)>10 else ifunc
        mol.output=x.iname+'_'+ot+'_'+f"{dist:.2f}"+'.log'
        mol.build()
        if len(ifunc) > 10 or ifunc=='ftPBE':
            raise NotImplementedError

        #Initialize to zero
        dip_cms = np.zeros(4*x.iroots).tolist()
        abs_pdft = 0
        abs_cas  = 0
        e_pdft   = 0
#-------------------- Energy ---------------------------
        if analyt == None:
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
                    if thresh: mc.conv_tol = thresh
                    mc.kernel(mo,ci)
                    e_states=mc.e_states.tolist() 
                    mo_sa = mc.mo_coeff #SA-CASSCF MOs
                    molden.from_mo(mol, out+'_sa.molden', mc.mo_coeff)
                    if dmcFLAG == True:
                        print("Working on Analytic CMS-PDFT Dipole")
                        for m in range(x.iroots):
                            shift=m*4
                            dipoles = mc.dip_moment(state=m, unit='Debye', origin=origin)
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
                    mo=mo_ss
                    # raise NotImplementedError
                elif method == 'CMS-PDFT' or method == 'SA-PDFT':
                    mo=mo_sa

                dip_num = numer_run(dist, x, mol, mo, ci, method, field, formula, ifunc, out, dip_cms)
        else:
            print("Numerical dipole is ignored")
            dip_num = np.zeros((len(field), 4))
        
   
        analytic[k] = [dist, abs_cas, abs_pdft] + dip_cms
        numeric [k] = dip_num
        en_dist [k] = [dist, e_casscf, e_pdft] + e_states
    return numeric, analytic, en_dist, mo, ci

# Get dipoles & energies for a fixed distance
def run(x, field, formula, numer, analyt, mo, ci, dist, ontop, scan, dip_scan, en_scan, dmcFLAG=True):
    numeric, analytic, en_dist, mo, ci = get_dipole(x, field, formula, numer, analyt, mo, ci, dist, ontop, dmcFLAG=dmcFLAG)

    # Accumulate analytic dipole moments and energies
    for k, ifunc in enumerate(ontop):
        out = 'dmc_'+x.iname+'_'+ifunc+'.txt'
        dip_scan[k].append(analytic[k]) 
        en_scan[k].append(en_dist[k]) 

        # Print & save numeric dipole moments
        if numer:
            for method in numer:
                ot='htPBE0' if len(ifunc)>10 else ifunc
                print(f"Numeric dipole at {cs(dist)} ang found with {cs(method)} ({cs(ot)})")
                header=['Field',]
                for i in range(x.iroots): 
                    header+=['X', 'Y', 'Z', f'ABS ({i+1})'] 
                sigfig = (".4f",)+(".5f",)*4*x.iroots
                numer_point = pdtabulate(numeric[k], header, sigfig)
                print(numer_point)
                action='w' if scan==False else 'a'
                with open(out, action) as f:
                    f.write(f"Numeric dipole at {dist:.3f} ang found with {method} ({ot})\n")
                    f.write(numer_point)
                    f.write('\n')
    return mo, ci


class Molecule:
    def __init__(self, iname, geom, nel, norb, cas_list, init=0, iroots=1, istate=0, 
                icharge=0, isym='C1', irep='A', ispin=0, ibasis='julccpvdz', grid=9):
        self.iname    = iname
        self.geom     = geom
        self.nel      = nel
        self.norb     = norb
        self.cas_list = cas_list
        self.init     = init
        self.iroots   = iroots
        self.istate   = istate
        self.icharge  = icharge
        self.isym     = isym
        self.irep     = irep
        self.ispin    = ispin
        self.ibasis   = ibasis
        self.grid     = grid

#--------------------- Set Up Parameters for Dipole Moment Curves--------------------
# Set the bond range
bonds = np.array([1.0])
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
# field = np.linspace(1e-3, 1e-2, num=1)
field = np.linspace(1e-3, 1e-2, num=3)
field = np.linspace(1e-3, 1e-2, num=19)
field = np.linspace(1e-3, 1e-2, num=2)
field = np.linspace(1e-3, 1e-2, num=10)
field = np.linspace(1e-5, 1e-2, num=30)
field = np.linspace(2e-3, 1e-2, endpoint=True, num=81)
# field = np.array([\
# 0.00300,0.00325,0.00350,0.00375,0.00400,0.00425,0.00450,0.00475,
# 0.00500,0.00525,0.00550,0.00575,0.00600,0.00625,0.00650,0.00675,
# 0.00700,0.00725,0.00750,0.00800,0.00850,0.00900,0.00950,0.01000])
# inc= 1e-3
# field = np.arange(inc, 1e-2, inc)
thresh = None
thresh = 1e-11
conv_tol_grad= 1e-6
# thresh = 5e-9

# Set name for tPBE0 (HMC-PDFT functional)
hybrid = 't'+ mcpdft.hyb('PBE', 0.25, 'average')
# Set on-top functional
ontop= ['tPBE']
ontop= ['tBLYP']
# ontop= ['tPBE','tOPBE', hybrid,'ftPBE']

# Set differentiation technique
# formula= "2-point"
formula = "4-point"

# #!!!!!!!!!!!!!
# field = np.linspace(1e-3, 1e-2, num=19)
# inc=0.1
# bonds = np.arange(1.2,3.0+inc,inc) # for energy curves


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
crh_7e7o = Molecule('crh_7e7o', geom_crh,   7,7, [10,11,12,13,14,15, 19], init=1.60, ispin=5, ibasis='def2tzvpd')
ch5_2e2o = Molecule('ch5_2e2o', geom_ch5,   2,2, [29, 35],                init=1.50, iroots=1, ibasis='augccpvdz')
co_10e8o = Molecule('co_10e8o', geom_co,   10,8, [3,4,5,6,7, 8,9,10],     init=1.20, iroots=3, ibasis='augccpvdz',)
h2co_6e6o        = Molecule('h2co_6e6o',    geom_h2co,        6,6, [6,7,8,9,10,12],         init=1.20, iroots=2)
phenol_8e7o_sto  = Molecule('phenol_8e7o_sto',geom_phenol,    8,7, [19,23,24,25,26,27,28], iroots=2, ibasis='sto-3g')
phenol_8e7o      = Molecule('phenol_8e7o',  geom_phenol,      8,7, [19,23,24,25,31,32,34], init=0.0, iroots=2)
OH_phenol_10e9o  = Molecule('OH_phenol_10e9o', geom_OH_phenol,10,9,[19,20,23,24,25,26,31,33,34], init=1.3, iroots=2)
phenol_12e11o  = Molecule('phenol_12e11o', geom_phenol_12e11o,12,11,[19,20,21,23,24,25,26,31,33,34,58], init=1.3, iroots=3)
OH_phenol3_10e9o =  copy.deepcopy(OH_phenol_10e9o)
OH_phenol3_10e9o.iroots=3
# OH_phenol4_10e9o =  copy.deepcopy(OH_phenol_10e9o)
# OH_phenol4_10e9o.iroots=4
spiro_11e10o  = Molecule('spiro_11e10o','frames',11,10,[35,43,50,51,52,53, 55,76,87,100], iroots=2, icharge=1, ispin=1)

# unit tests
h2o_4e4o      = Molecule('h2o_4e4o', geom_h2o, 4,4, [4,5,8,9], iroots=3, grid=1, isym='c2v', irep='A1', ibasis='aug-cc-pVDZ')
furancat_5e5o = Molecule('furancat_5e5o', geom_furan, 5,5, [12,17,18,19,20], iroots=3, grid=1, icharge = 1, ispin =1, ibasis='sto-3g')


furan_6e5o_2_shifted = Molecule('furan_6e5o_shift', geom_furan_shifted, 6,5, [12,17,18,19,20], ibasis='631g*', iroots=2)
furan_6e5o_2_rotated = Molecule('furan_6e5o_rot',   geom_furan_rotated, 6,5, [12,17,18,19,20], ibasis='631g*', iroots=2)
furan_6e5o_2 = Molecule('furan_6e5o',    geom_furan,      6,5, [12,17,18,19,20],       ibasis='631g*', iroots=2)
#Select species for which the dipole moment curves will be constructed
# species=[crh_7e7o]
# species=[ch5_2e2o]
# species=[co_10e8o]
species=[h2co_6e6o]
# species=[phenol_8e7o]
species=[phenol_8e7o_sto]
species=[OH_phenol_10e9o]
species=[spiro_11e10o]
species=[phenol_8e7o]
# species=[OH_phenol4_10e9o]
species=[OH_phenol3_10e9o]
# species=[OH_phenol3_12e11o]
species=[phenol_12e11o]
species=[furan_6e5o_2_shifted]
species=[furan_6e5o_2]
species=[furan_6e5o_2_rotated]
species=[h2o_4e4o]
species=[furancat_5e5o]


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
    mo, ci = init_guess(y, analyt, numer) if x !=spiro_11e10o else 0 #spiro MOs should be provided manually

    for i, dist in enumerate(bonds, start=0):
        # Update the bond length in the instance
        if i==0: template=x.geom
        x.geom=template.format(dist)

        # MOs and CI vectors are taken from the previous point
        mo, ci = run(x, field, formula, numer, analyt, mo, ci, dist, ontop, scan, dip_scan, en_scan, dmcFLAG=dmcFLAG)
        
        dip_head = ['Distance','CASSCF','MC-PDFT']
        for j in range(x.iroots): 
            dip_head+=['X', 'Y', 'Z', f'ABS ({cs(j+1)})']
        dip_sig = (".2f",)+(".5f",)*(2+4*x.iroots)
        
        en_head=['Distance', 'CASSCF', 'MC-PDFT']
        for method in analyt:
            for jj in range(x.iroots): 
                line=f'{method} ({cs(jj+1)})'
                en_head.append(line)
            en_sig = (".2f",)+(".8f",)*(2+x.iroots)

        for k, ifunc in enumerate(ontop):
            out = x.iname+'_'+ifunc+'.txt'
            if analyt:
                print(f"Analytic dipole moments found with {cs(ifunc)}")
                dip_table = pdtabulate(dip_scan[k], dip_head, dip_sig)
                print(dip_table)

            print(f"Energies found with {cs(ifunc)}")
            en_table = pdtabulate(en_scan[k], en_head, en_sig)
            print(en_table)
            action='a' if numer else 'w'
            with open(out, action) as f:
                f.write(f"The on-top functional is {ifunc} \n")
                f.write(en_table)
                f.write('\n')
                if analyt:
                    f.write(dip_table)
                    f.write('\n')
