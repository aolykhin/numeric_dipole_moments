from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
from pyscf.tools import molden
from pyscf.geomopt import geometric_solver
from pyscf.gto import inertia_moment
from pyscf.data import nist
import copy
from pyscf.gto import inertia_moment
from pyscf.data import nist
from numpy.linalg import norm as norm
import math

def tdm_casci(mo,mc_eq,mol,state):
    from functools import reduce
    mass = mol.atom_mass_list()
    coords = mol.atom_coords()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    orb = mo[:,mc_eq.ncore:mc_eq.ncore+mc_eq.ncas]
    with mol.with_common_orig(mass_center):
        dip_ints = mol.intor('cint1e_r_sph', comp=3)
    ci_buf=mc_eq.get_ci_adiabats(uci='MCSCF')
    m, n = state[0], state[1]
    t_dm1 = mc_eq.fcisolver.trans_rdm1(ci_buf[n], ci_buf[m], mc_eq.ncas, mc_eq.nelecas)
    # t_dm1 = mc_eq.fcisolver.trans_rdm1(ci_buf[state[1]], ci_buf[state[0]], mc_eq.ncas, mc_eq.nelecas)
    t_dm1_ao = reduce(np.dot, (orb, t_dm1, orb.T))
    tdm = np.einsum('xij,ji->x', dip_ints, t_dm1_ao)
    return tdm

# def render_tdm_pdm(mass_center, abc_vec, pdm, tdm):
def render_tdm_pdm(x, mass_center, abc_vec, pdm, tdm, method):
    #set parameters
    scale_axes = 2
    scale_tdm  = 2
    scale_pdm  = 1
    label_size = 0.15
    label_axes = ['a','b']
    black  = '0.01  0.1   0.0   0.0   0.0  0'
    blue   = '0.02  0.1   0.0   0.5   1.0  0'
    red    = '0.02  0.1   1.0   0.0   0.0  0'
    green  = '0.02  0.1   0.0   1.0   0.0  0'
    purple = '0.02  0.1   0.5   0.0   1.0  0'
    grey   = '0.02  0.1   0.5   0.5   0.5  0'

    #shift abc frame to the COM and print out coordinated of abc vectors
    with open('chemcraft_dipoles_'+method, 'a') as f:
        print(x.iname, file=f)
        print('#Inertial axes', file=f)
        for i in range(2): #only a and b
            axis_main =  abc_vec[:,i]*scale_axes + mass_center
            axis_auxi = -abc_vec[:,i]*scale_axes + mass_center
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} N '.format(*mass_center,*axis_main)+black, file=f)
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} N '.format(*mass_center,*axis_auxi)+black, file=f)
            print('L {:8.4f} {:8.4f} {:8.4f} {:.2f} {}'.format(*axis_main*1.1, label_size, label_axes[i]), file=f)
        #ground state geometry may be not a good frame to render excited state dipole
        print('#Permanent dipole moment', file=f)
        color = [purple, grey, green]
        pdm_shifted = np.empty_like(pdm)
        for i in range(len(pdm)):
            if len(pdm)>len(color): RuntimeError('Add more colors to pdm')
            pdm_shifted[i] = pdm[i]*scale_pdm + mass_center
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} S '.format(*mass_center,*pdm_shifted[i]) + color[i], file=f)

        print('#Transiton dipole moment', file=f)
        color = [blue, red]
        tdm_main = np.empty_like(tdm)
        tdm_auxi = np.empty_like(tdm)
        for i in range(len(tdm)):
            if len(tdm)>len(color): RuntimeError('Add more colors to tdm')
            tdm_main[i] =  tdm[i]/norm(tdm[i])*scale_tdm + mass_center
            tdm_auxi[i] = -tdm[i]/norm(tdm[i])*scale_tdm + mass_center
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} S '.format(*mass_center,*tdm_main[i]) + color[i], file=f)
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} S '.format(*mass_center,*tdm_auxi[i]) + color[i], file=f)
        print(' ', file=f)
    return

# counterclockwise rotation from vec1 to vec2
def findClockwiseAngle(ini, fin):
    ind=2 # c-component in abc frame
    if len(ini)==3 and abs(ini[ind])<1e-4: ini=np.delete(ini,ind) 
    if len(fin)==3 and abs(fin[ind])<1e-4: fin=np.delete(fin,ind)
    if norm(ini) <1e-3: #Trivial case of zero TDM 
        ang = 0
    else:
        ang = -math.degrees(math.asin(np.cross(ini,fin)/(norm(ini)*norm(fin))))
    return ang 

def transform_dip(x, mol, pdm, tdm, method):
    mass = mol.atom_mass_list()
    coords = mol.atom_coords(unit='ANG')
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()

    matrix = inertia_moment(mol, mass=mass, coords=coords)
    moments, abc_vec=np.linalg.eigh(matrix) #ascending order

    # Get vectors for Chemcraft
    render_tdm_pdm(x, mass_center, abc_vec, pdm, tdm, method)
    # Projection of PDM onto ABC frame 
    origin  = np.eye(3)
    rot_mat = np.linalg.solve(origin,abc_vec)
    pdm_abc, pdm_ang = get_ang_with_a_axis(pdm, rot_mat)
    tdm_abc, tdm_ang = get_ang_with_a_axis(tdm, rot_mat)
    with open('tdms_angles_'+method, 'a') as f:
        for i in range(len(pdm_ang)):
            print(f'{x.iname} theta with pdm is {pdm_ang[i]:5.1f} degrees', file=f)
        for i in range(len(tdm_ang)):
            print(f'{x.iname} theta with tdm is {tdm_ang[i]:5.1f} degrees', file=f)
    return pdm_abc, tdm_abc

def get_ang_with_a_axis(dm, rot_mat):
    dm_abc = np.empty_like(dm)
    angles = [None]*len(dm)

    for i in range(len(dm)):
        dm_abc[i]=np.dot(dm[i],rot_mat)
        #Angle of pdm wrt a-axis
        ini = dm_abc[i]
        fin = np.array([1,0,0])
        dot = np.dot(ini,fin)/(norm(ini)*norm(fin))
        if abs(dot) < 1e-3: #accurate to 0.06 degrees
            ang = 90
        elif abs(dot-1) < 1e-3:
            ang = 0
        elif abs(dot+1) < 1e-3:
            ang = 180
        elif dot < 0: 
            fin = -fin
            ang = findClockwiseAngle(ini, fin)
        else:
            ang = findClockwiseAngle(ini, fin)
        angles[i] = ang
    return dm_abc, angles

def cms_dip(x):
    out = x.iname+'_'+str(x.nel)+'e'+str(x.norb)+'o'+'_'+str(x.istate)
    mol = gto.M(atom=open('geom_'+x.iname+'.xyz').read(), charge=x.icharge, spin=x.ispin,
                    symmetry=x.isym, output=out+'.log', verbose=4, basis=x.ibasis)
    weights=[1/x.iroots]*x.iroots 
    
    # -------------------- HF ---------------------------
    mf = scf.RHF(mol).run()
    molden.from_mo(mol, out+'_hf.molden', mf.mo_coeff)

    # -------------------- MC-PDFT ---------------------------
    mc = mcpdft.CASSCF(mf, x.ifunc, x.norb, x.nel)
    mc.fcisolver = csf_solver(mol, smult=x.ispin+1)
    mo = mcscf.sort_mo(mc, mf.mo_coeff, x.cas_list)
    mc.kernel(mo)
    mo = mc.mo_coeff 

    # ----------------- Geometry Optimization ----------------------
    if x.opt == True:
        mol_eq = mc.nuc_grad_method().as_scanner().optimizer().kernel()
    else: #Preoptimized geometry
        mc_eq  = copy.deepcopy(mc)
        mol_eq = copy.deepcopy(mol)
        # mc_eq  = mc
        # mol_eq = mol

    # -------------------- CMS-PDFT ---------------------------
    mf_eq = scf.RHF(mol_eq).run()
    mc_eq = mcpdft.CASSCF(mf_eq, x.ifunc, x.norb, x.nel)
    mc_eq.fcisolver = csf_solver(mol_eq, smult=x.ispin+1)
    mc_eq.fcisolver.wfnsym = x.irep
    mc_eq = mc_eq.multi_state(weights,'cms')
    mc_eq.max_cyc = 500
    mo = mcscf.project_init_guess(mc_eq, mo)
    mc_eq.kernel(mo)

    # ------------------- CMS-PDFT PDM AND TDM ------------------
    mo_sa = mc_eq.mo_coeff 
    e_opt = mc_eq.e_states.tolist() #List of CMS energies
    eq_dip= mc_eq.dip_moment(state=x.istate, unit='Debye')
    eq_pdm = [eq_dip]
    tdm = []
    if x.istate==0:
        for i in range(x.iroots):
            if i!=0:
                tmp = mc_eq.trans_moment(state=[0,i], unit='Debye')
                tdm.append(tmp)
        abs = np.linalg.norm(eq_dip)
        molden.from_mo(mol_eq, out+'_opt.molden', mc_eq.mo_coeff)

        with open('dipoles_cms', 'a') as f:
            print(x.iname, f'PDM <0|mu|0> ABS (D) = {abs:7.2f} XYZ (D) = {eq_dip[0]:7.3f} {eq_dip[1]:7.3f} {eq_dip[2]:7.3f}', file=f)
            for n in range(x.iroots):
                if n!=0:#
                    k = n-1 #
                    print(x.iname, f'TDM <0|mu|{n}> ABS (D) = {abs:7.2f} XYZ (D) = {tdm[k][0]:7.3f} {tdm[k][1]:7.3f} {tdm[k][2]:7.3f}' , file=f)
    else:
        tmp = mc_eq.trans_moment(state=[0,x.istate], unit='Debye')
        tdm.append(tmp)
        abs = np.linalg.norm(eq_dip)
        molden.from_mo(mol_eq, out+'_opt.molden', mc_eq.mo_coeff)
        with open('dipoles_cms', 'a') as f:
            print(x.iname, f'PDM <{x.istate}|mu|{x.istate}> ABS (D) = {abs:7.2f} XYZ (D) = {eq_dip[0]:7.3f} {eq_dip[1]:7.3f} {eq_dip[2]:7.3f}', file=f)
            k = 0 #
            print(x.iname, f'TDM <0|mu|{x.istate}> ABS (D) = {abs:7.2f} XYZ (D) = {tdm[k][0]:7.3f} {tdm[k][1]:7.3f} {tdm[k][2]:7.3f}' , file=f)

    # pdm_abc, tdm_abc = transform_dip(x, mol, eq_pdm, tdm, 'cms')


    # mass = mol_eq.atom_mass_list()
    # coords = mol_eq.atom_coords(unit='ANG')
    # com = np.einsum('i,ij->j', mass, coords)/mass.sum()

    # ------------------- CAS-CI PDM AND TDM ------------------
    eq_dip = tdm_casci(mo_sa,mc_eq,mol_eq,state=[0,0])
    eq_dip*= nist.AU2DEBYE
    eq_pdm = [eq_dip]
    abs = np.linalg.norm(eq_dip)
    with open('dipoles_sa_casscf', 'a') as f:
        print(x.iname, f'PDM <0|mu|0> ABS (D) = {abs:7.2f} XYZ (D) = {eq_dip[0]:7.3f} {eq_dip[1]:7.3f} {eq_dip[2]:7.3f}', file=f)
    #Compute & save TDMs from the optimized excited to ground state
    if x.istate==0:
        tdm=[None]*(x.iroots-1) # from <0| to others
        for n in range(x.iroots):
            if n!=0:#
                k = n-1 # TDM's id 
                tdm[k]=tdm_casci(mo_sa,mc_eq,mol_eq,state=[0,n])

                abs = np.linalg.norm(tdm[k])
                oscil= (2/3)*(e_opt[k]-e_opt[0])*(abs**2)
                tdm[k]*= nist.AU2DEBYE
                abs*= nist.AU2DEBYE

                with open('dipoles_sa_casscf', 'a') as f:
                    print(x.iname, f'TDM <0|mu|{n}> ABS (D) = {abs:7.2f} XYZ (D) = {tdm[k][0]:7.3f} {tdm[k][1]:7.3f} {tdm[k][2]:7.3f} Oscil = {oscil:7.2f}' , file=f)

        pdm_abc, tdm_abc = transform_dip(x, mol, eq_pdm, tdm, 'casci')
    else:    
        # tdm=mc.trans_moment(state=[x.istate,0])
        tdm=tdm_casci(mo_sa,mc_eq,mol_eq,state=[x.istate,0])
        abs = np.linalg.norm(tdm)
        oscil= (2/3)*(e_opt[x.istate]-e_opt[0])*(abs**2)
        tdm*= nist.AU2DEBYE
        abs*= nist.AU2DEBYE
        with open('tdms_from_excited_states', 'a') as f:
            print(x.iname, f'TDM <{x.istate}|mu|0> ABS = {abs:7.2f} Oscil = {oscil:7.2f} \
                XYZ: {tdm[0]:7.3f} {tdm[1]:7.3f} {tdm[2]:7.3f}', file=f)
            with open('tdms_from_excited_states_COM', 'a') as f:
                tdm+=com
                print(x.iname, f'<{x.istate}|mu|0> vec V {com[0]:7.3f} {com[1]:7.3f} {com[2]:7.3f} {tdm[0]:7.3f} {tdm[1]:7.3f} {tdm[2]:7.3f} S', file=f)
            print(x.iname, 'TDM %s0 ABS = %.2f Oscillator = %.2f XYZ: %.4f %.4f %.4f' % \
            (x.istate,abs,oscil, tdm[0],tdm[1],tdm[2]), file=f)



    # #Save TDMs
    # with open('tdms_from_excited', 'a') as f:
    #     print(x.iname, 'TDM is %s0 ', 'ABS = %.2f', 'Oscillator = %.2f' % (x.istate,abs,oscil), file=f )
    #     print('TDM along XYZ-coord axes: %.4f %.4f %.4f \n' % (tdm_opt[0],tdm_opt[1],tdm_opt[2]), file=f)
  
    #Save the optimized geometry to the xyz file
    with open('geom_opt_'+out+'.xyz', 'w') as f:
        for i in range(mol_eq.natm):
            print('%s  %s' % (mol_eq.atom_symbol(i), str(mol_eq.atom_coord(i,unit='Angstrom'))[1:-1]), file=f)

    #Save final energies
    with open('energies', 'a') as f:
        print('%s  %s' % (out, e_opt), file=f)
            

    # #Compute and save dipole moments at the optimized geometry 
    # eq_dip_abc=transform_dip(eq_dip,mol)
    # val=np.linalg.norm(eq_dip)
    # print(x.iname, 'state is ', x.istate, 'ABS = %.2f' % (val) )
    # print('Dipole along XYZ-coord axes: %8.4f %8.4f %8.4f \n' % (eq_dip[0],eq_dip[1],eq_dip[2]))
    # print('Dipole along principle axes: %8.4f %8.4f %8.4f \n' % (eq_dip_abc[0],eq_dip_abc[1],eq_dip_abc[2]))

    # #Save final dipoles
    # with open('dipoles', 'a') as f:
    #     print('%s  %8.2f' % (out, val), file=f)
    # with open('dipoles_abs', 'a') as f:
    #     print('%s  %8.2f  %8.2f  %8.2f' % (out, eq_dip_abc[0],eq_dip_abc[1],eq_dip_abc[2]), file=f)

    # #Compute state-specific NOONs
    # mc = mcscf.CASCI(mf, x.norb, x.nel).state_specific_(x.istate)
    # mc.natorb=True
    # mc.fix_spin_(ss=x.ispin)
    # emc = mc.casci(mo_sa)[0]
    # mc.analyze(large_ci_tol=0.05)
    # print(emc)
    # print(mc.mo_occ)

    return
class Molecule:
    def __init__(self, iname, nel, norb, cas_list, iroots=2, istate=1, 
                opt=False, icharge=0, isym='C1', irep='A', ispin=0, ibasis='julccpvdz', ifunc='tPBE'):
        self.iname    = iname
        self.nel      = nel
        self.norb     = norb
        self.cas_list = cas_list
        self.iroots   = iroots
        self.istate   = istate
        self.opt      = opt
        self.icharge  = icharge
        self.isym     = isym
        self.irep     = irep
        self.ispin    = ispin
        self.ibasis   = ibasis
        self.ifunc    = ifunc

x=[None]*20
x[0]  = Molecule('x7_azaindole'        , 10,9,  [22,27,29,30,31,35,38,43,44])
x[1]  = Molecule('benzonitrile'        , 10,10, [21,24,25,26,27,29,32,43,45,46])
x[2]  = Molecule('dimethoxybenzene'    , 10,8,  [24,25,35,36,37,47,50,56])
x[3]  = Molecule('fluorobenzene'       , 6,6,   [23,24,25,31,32,34])
x[4]  = Molecule('x5_cyanoindole'      , 14,13, [26,31,33,34,35,36,37, 41,44,45,46,49,56])
x[5]  = Molecule('x4_fluoroindole'     , 10,9,  [27,32,33,34,35,41,43,46,47])
x[6]  = Molecule('x5_fluoroindole'     , 10,9,  [27,32,33,34,35,40,44,45,51])
x[7]  = Molecule('x6_fluoroindole'     , 10,9,  [28,32,33,34,35,40,44,45,48])
x[8]  = Molecule('x1_fluoronaphthalene', 10,10, [32,35,36,37,38,42,45,47,50,53])
x[9]  = Molecule('x2_fluoronaphthalene', 10,10, [31,35,36,37,38,42,45,48,50,52])
x[10] = Molecule('formaldehyde'        ,  6, 6, [6,7,8,9,10,11])
x[11] = Molecule('anti_5_hydroxyindole', 12,10, [35,34,33,32,27,25,41,44,46,47])
x[12] = Molecule('indole'              , 10,9,  [23,28,29,30,31,37,40,42,45])
x[13] = Molecule('anti_4_methoxyindole', 12,10, [25,28,36,37,38,39,45,46,50,51])
x[14] = Molecule('anti_5_methoxyindole', 12,10, [28,33,36,37,38,39,45,46,50,52])
x[15] = Molecule('syn_6_methoxyindole' , 12,10, [29,33,36,37,38,39,45,47,50,52])
x[16] = Molecule('cis_2_naphthol'      , 12,11, [27,32,35,36,37,38,42,46,48,50,53])
x[17] = Molecule('trans_2_naphthol'    , 12,11, [28,32,35,36,37,38,42,45,48,50,53])
x[18] = Molecule('phenol'              ,  8,7,  [19,23,24,25,31,33,34])
x[19] = Molecule('propynal'            ,  8,7,  [11,12,13,14,16,21,22])


# x[12].istate = 0
# cms_dip(x[12])

x[10].istate = 0
# x[10].opt = False
cms_dip(x[10])

# x[3].istate = 0
# x[3].opt = False
# cms_dip(x[3])