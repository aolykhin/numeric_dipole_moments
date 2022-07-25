from logging import raiseExceptions
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyscf import gto, scf, mcscf, lib
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
from pyscf.fci.addons import overlap as braket
import itertools
from numpy import unravel_index
from icecream import ic

def cs(text): return fg('light_green')+text+attr('reset')

def get_sign_list(x):
    sign_list=[]
    combos=[]
    tmp=[1]*x.iroots
    for i in range(x.iroots): #all except all states flipped
    # for i in range(x.iroots+1):
        if i!=0: tmp[i-1]=-1
        combos=itertools.permutations(tmp,x.iroots)
        for i in combos:
            if i not in sign_list:
                sign_list.append(i)
    return sign_list

def fix_sa_reference_order(x, mol, ci_buf, ci_zero, si_sa_buf, si_sa_zero):
    regular=list(range(x.iroots))
    over=np.zeros((x.iroots,x.iroots),dtype=int)
    ci_order=[]
    ci_signs=[]
    for i in range(x.iroots):
        for j in range(x.iroots):
            val=braket(ci_zero[i],ci_buf[j],norb=x.norb,nelec=x.nel)   
            print('VAL before rounding\n',i,j,val)
            over[i,j]=int(round(val))
            if over[i,j] !=  0: ci_order.append(j)
            if over[i,j] == -1: ci_signs.append(-1)
            if over[i,j] ==  1: ci_signs.append(1)
    if (len(ci_order) > x.iroots) or (len(ci_signs) > x.iroots):
        raise ValueError('CI vectors are not orthogonal enough')
    
    # # For some reason the above code doesn't capture change of a sign
    # # in reference states (columns) -> An extra step here resolves the problem
    # # Reflect reference states to better fit si_in_zero 
    # sign_list=get_sign_list(x)
    # overlap=np.zeros(len(sign_list))
    # for i, signs in enumerate(sign_list):
    #     for j, coeff in enumerate(signs): 
    #         si_sa_buf[:,j]=coeff*si_sa_buf[:,j]
    #         diff=abs(si_sa_buf-si_sa_zero)
    #         overlap[i]=diff.sum()
    # ind=np.argmin(overlap)
    # ic(ind)
    # ci_signs=sign_list[ind]
    # ic(sign_list)
    # ic(overlap)
    # ic(ci_signs)

    #Print overlap of adjusted ci_buf with ci_zero
    if mol.verbose >= lib.logger.DEBUG:
        print('ci_order\n',ci_order)
        print('ci_signs\n',ci_signs)
        ci_buf=np.asanyarray(ci_buf)
        ci_buf[regular]=ci_buf[ci_order]
        for i, coeff in enumerate(ci_signs): ci_buf[i]=coeff*ci_buf[i]
        ci_buf=list(ci_buf)
        for i in range(x.iroots):
            for j in range(x.iroots):
                val=braket(ci_zero[i],ci_buf[j],norb=x.norb,nelec=x.nel)   
                over[i,j]=int(round(val))
        print('Overlap of CI vectors after adjustment\n',over)

    #Adjust order & signs of SA states in si_sa_buf 
    # i <-- j, the ci_signs array is given for the rotated states 
    print('BEFORE CI ADJUSTMENT\n',si_sa_buf)
    si_sa_buf[:,regular]=si_sa_buf[:,ci_order]
    for i, coeff in enumerate(ci_signs): si_sa_buf[:,i]=coeff*si_sa_buf[:,i]
    print('AFTER CI ADJUSTMENT\n',si_sa_buf)

    return si_sa_buf

def fix_order_of_states(x, mol, ci_buf, ci_zero, si_sa_zero, si_in_zero, ham_zero, si_sa_buf, si_in_buf, ham_buf):
    #Rotate sa reference states in expansion over sa reference states
    si_sa_buf=fix_sa_reference_order(x, mol, ci_buf, ci_zero, si_sa_buf, si_sa_zero)

    # Get all variations of signs & oredr of intermediate states (rows) 
    order=list(itertools.permutations(range(x.iroots)))
    sign_list=get_sign_list(x)

    n=len(order)
    # new_order=[]
    regular=list(range(x.iroots))
    print(order)

    # Rotate ROWS of a field-dependent matrix
    sign=np.zeros((x.iroots,x.iroots))
    wrk=np.zeros((x.iroots,x.iroots))
    dim_sign=len(sign_list)
    overlap=np.zeros((n,dim_sign))
    overlap_si=np.zeros((n,dim_sign))

    print('Original Ham\n=',ham_buf)
    
    old_diff_si=100 
    for m, new_signs in enumerate(sign_list): 
        for k, new_order in enumerate(order):# over all permutations
            #Selected order of intermediate states with all variations of signs
            for i in regular:
                for j in regular:
                    sign[i,j]=new_signs[i]*new_signs[j]
                    wrk[i,j] = sign[i,j]*ham_buf[new_order[i],new_order[j]]

            diff=abs(wrk-ham_zero)
            overlap[k,m]=diff.sum()

            if overlap[k,m]<0.15:
                print(' ')
                print('new_order ',new_order)
                print('new_sign ',new_signs)
                print('Overlap=',overlap[k,m])
                print('Adj Ham\n=',wrk)

                wrk_si=si_sa_buf.copy()
                print('Working on si_sa_buf\n',si_sa_buf)
                for ii in range(x.iroots):
                    wrk_si[ii,:]=new_signs[ii]*wrk_si[ii,:]
                wrk_si[regular,:]=wrk_si[new_order,:]

                print('Adj si_sa_buf\n',wrk_si)

                # wrk_si=adjust_columns(x, wrk_si, si_sa_zero)
                diff_si=abs(wrk_si-si_sa_zero)
                new_diff_si=diff_si.sum()
                print('si diff=',new_diff_si)
                if new_diff_si<old_diff_si:
                    old_diff_si=new_diff_si
                    fin_order=list(new_order)
                    fin_signs=list(new_signs)
    print('Overlap=',overlap[k,m])

    # ind_ham, ind_sign = np.unravel_index(np.argmin(overlap, axis=None), overlap.shape)
    # fin_order=list(order[ind_ham])
    # fin_signs=list(sign_list[ind_sign])

    print('final new_order=',fin_order)        
    print('final new_signs=',fin_signs)       
    
    #Rotate intermediate states in expansion over sa reference states
    for ii in range(x.iroots):
        si_sa_buf[ii,:]=fin_signs[ii]*si_sa_buf[ii,:] 
    si_sa_buf[regular,:]=si_sa_buf[fin_order,:]
    print('FINAL si_sa_buf\n',si_sa_buf)
    
    #Rotate intermediate states in expansion over intermediate states
    for ii in range(x.iroots):
        si_in_buf[ii,:]=fin_signs[ii]*si_in_buf[ii,:] 
    si_in_buf[regular,:]=si_in_buf[fin_order,:]
    
    #Rotate Hamiltonian
    tmp=ham_buf.copy()
    for i in regular:
        for j in regular:
            sign[i,j]=fin_signs[i]*fin_signs[j]
            ham_buf[i,j] = sign[i,j]*tmp[fin_order[i],fin_order[j]]

    print(overlap)

    return si_sa_buf, si_in_buf, ham_buf
    

# ------------------ NUMERICAL DIPOLE MOMENTS ----------------------------
def numer_run(dist, x, mol, mo_zero, ci_zero, method, field, formula, ifunc, out, dip_cms, \
    si_sa_zero, si_in_zero, ham_zero, ntdm):
    global thresh, max_cyc, nudge_tol
    # Set reference point to be center of charge
    mol.output='num_'+ out
    mol.build()
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nuc_charge_center = np.einsum(
        'z,zx->x', charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    h_field_off = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')

    dip_num = np.zeros((len(field), 1+4*ntdm))# 1st column is the field column
    for i, f in enumerate(field): # Over field strengths
        dip_num[i][0]=f # the first column is the field column 
        if formula == "2-point":
            disp = [-f, f]
        elif formula == "4-point":
            disp = [-2*f, -f, f, 2*f]
        si_in = np.zeros((len(disp),3,x.iroots,x.iroots)) #
        si_sa = np.zeros((len(disp),3,x.iroots,x.iroots)) #
        si_der = np.zeros((len(field),3,x.iroots,x.iroots)) 
        ham = np.zeros((len(disp),3,x.iroots,x.iroots)) #H_PQ matrix per  disp
        der = np.zeros((len(field),3,x.iroots,x.iroots)) #Der_PQ matrix per disp and per lambda
        ci_buf  = [None]*x.iroots
        # e = [ [] for _ in range(2) ]
        if i==0: #set zero-field MOs as initial guess 
            mo_field = []
            ci_field = []
            for icoord in range(3): mo_field.append([mo_zero]*len(disp))
            for icoord in range(3): ci_field.append([ci_zero]*len(disp))

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
                if j==0: coord='x'
                if j==1: coord='y'
                if j==2: coord='z'
                if k==0: point='+F'
                if k==1: point='-F'
                print('j=%s and k=%s' %(coord,point))
                if i==0: #First field starts with zero-field MOs
                    mc.max_cycle_macro = 200
                    mo=mo_zero 
                    ci=ci_zero 
                else: # Try MOs from the previous filed/point
                    mc.max_cycle_macro = 5
                    mo=mo_field[j][k]
                    ci=ci_field[j][k]
                # if the threshold is too tight the active space is unstable
                mc.conv_tol = mc.conv_tol_sarot = thresh 
                weights=[1/x.iroots]*x.iroots #Equal weights only
                # MC-PDFT 
                if method == 'SS-PDFT':
                    raise NotImplementedError
                # SA-PDFT 
                elif method == 'SA-PDFT':
                    raise NotImplementedError
                # CMS-PDFT 
                elif method == 'CMS-PDFT':
                    mc=mc.multi_state(weights,'cms').run(mo,ci)
                    mc.max_cyc=max_cyc
                    mc.conv_tol=thresh
                    mc.sing_tol=thresh
                    mc.nudge_tol=nudge_tol
                    if mc.converged==False:
                        mc.max_cycle_macro = 200
                        mc.run(mo_zero, ci_zero)
                    ci_buf = mc.ci
                    print('NEXT HAMILTONIAN')
                    e_cms   = mc.e_states.tolist() #List of CMS energies
                    ham_buf = mc.get_heff_pdft()
                    si_in_buf  = mc.si_pdft
                    si_sa_buf  = mc.si_mcscf

                    # In the presence of field, the sign and order of sa and intermediate states can change
                    # The following code aims to preserve the order and sings as in the zero-field case 
                    si_sa_buf, si_in_buf, ham_buf = fix_order_of_states(x, mol, ci_buf, ci_zero, \
                        si_sa_zero, si_in_zero, ham_zero, si_sa_buf, si_in_buf, ham_buf)
                    
                    ham[k,j,:,:] = ham_buf
                    si_in[k,j,:,:] = si_in_buf
                    si_sa[k,j,:,:] = si_sa_buf
                    print('si_in\n',si_in)
                    print('si_sa\n',si_sa)
                    print('H_PQ\n',ham)
                    # e[k] = mc.e_states.tolist() #List of  energies
                else:
                    raise NotImplementedError
                mo_field[j][k] = mc.mo_coeff #save MOs for the next stencil point k and axis j 

            #Loop over j = {x,y,z} directions
            #Get dH_PQ/dF derivative numerically        
            for p in range(x.iroots):
                for q in range(x.iroots):
                    if formula == "2-point":
                        der[i,j,p,q]    = (-1)*nist.AU2DEBYE*(-ham[0,j,p,q]+ham[1,j,p,q])/(2*f)
                        si_der[i,j,p,q] = (-1)*nist.AU2DEBYE*(-si_in[0,j,p,q]+si_in[1,j,p,q])/(2*f)
        #Loop over i = fields
        id_tdm=-1 #enumerate TDM
        for m in range(x.iroots): # TDMs between <m| and |n> states
            for n in range(m):
                id_tdm+=1
                shift=id_tdm*4 # shift to the next state by 4m columns (x,y,z,mu)
                for j in range(3): #over j = {x,y,z} directions
                    for p in range(x.iroots):
                        for q in range(x.iroots):
                            dip_num[i,1+j+shift]+=der[i,j,p,q]*si_in_zero[p][m]*si_in_zero[q][n]

                            # dip_num[i,1+j+shift]+=der[i,j,p,q]*si_in_zero[p][m]*si_in_zero[q][n] + \
                            #     (si_in_zero[p][m]*si_der[i,j,q,n]+si_in_zero[q][n]*si_der[i,j,p,m])*ham_zero[p][q]

        # Get permamment/transition dipole moment

        print('der\n',der) 
        print('si_der\n',si_der) 
           
        print('dip_num',dip_num)
        for mn in range(ntdm):
            shift=mn*4 # shift to the next state by 4m columns (x,y,z,mu)    
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
    global thresh, max_cyc, nudge_tol
    out = y.iname+'_cas'
    xyz = open('00.xyz').read() if y.geom == 'frames' else y.geom
    mol = gto.M(atom=xyz, charge=y.icharge, spin=y.ispin,
                output=y.iname+'_init.log', verbose=4, basis=y.ibasis, symmetry=y.isym)
    weights=[1/x.iroots]*x.iroots
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
    mo = cas.mo_coeff
    saFLAG=True if ('CMS-PDFT' or 'SA-PDFT' or 'SA-CASSCF') in (analyt + numer) else False
    if saFLAG==True:
        cas.state_average_(weights)
        e_casscf = cas.kernel(mo)[0]
    cas.analyze()
    mo = cas.mo_coeff
    ci = cas.ci
    molden.from_mo(mol, out+'_init_cas.molden', cas.mo_coeff)
    return mo, ci

#-------------Compute energies and dipoles for the given geometry-----------
def get_dipole(x, field, formula, numer, analyt, mo, ci, dist, ontop, ntdm, dmcFLAG=True):
    global thresh, max_cyc, nudge_tol
    out = x.iname+'_'+f"{dist:.2f}"
    xyz = open(str(dist).zfill(2)+'.xyz').read() if x.geom == 'frames' else x.geom
    mol = gto.M(atom=xyz,charge=x.icharge,spin=x.ispin,output=out+'.log',
                basis=x.ibasis, symmetry=x.isym, verbose=lib.logger.DEBUG)
                # basis=x.ibasis, symmetry=x.isym, verbose=4)
    #HF step
    weights=[1/x.iroots]*x.iroots
    mf = scf.RHF(mol).set(max_cycle = 1).run()

    #SS/SA-CASSCF step
    fname = 'orb_'+ out #file with orbitals
    cas = mcscf.CASSCF(mf, x.norb, x.nel)
    # cas.natorb = True
    cas.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
    cas.fcisolver.wfnsym = x.irep
    cas.chkfile = fname
    cas.conv_tol = thresh 
    if os.path.isfile(fname) == True:
        print('Read orbs from the previous calculations')
        mo = lib.chkfile.load(fname, 'mcscf/mo_coeff')
    else:
        print('Project orbs from the previous point')
        mo = mcscf.project_init_guess(cas, mo)
    
    if 'MC-PDFT' in (analyt + numer): 
        e_casscf = cas.kernel(mo,ci)[0]
        mo_ss=cas.mo_coeff
        cas.analyze()
        molden.from_mo(mol, out+'_ss.molden', cas.mo_coeff)
    if ('CMS-PDFT' or 'SA-PDFT' or 'SA-CASSCF') in (analyt + numer):
        cas.state_average_(weights)
        e_casscf = cas.kernel(mo,ci)[0]
        mo_sa = cas.mo_coeff
        ci = cas.ci
        cas.analyze()
        molden.from_mo(mol, out+'_sa.molden', cas.mo_coeff)
    mo=mo_sa #ONLY TRUE FOR TDM!!!!!!!!!!!!!!!
    #     mo=mo_ss
    # elif method == 'CMS-PDFT' or method == 'SA-PDFT':
    #     mo=mo_sa
    # for m in range(x.iroots):
    #     print('TEST NOONS CASCI ')
    #     mc_sa = mcscf.CASCI(mf, x.norb, x.nel).state_specific_(m)
    #     mc_sa.natorb=True
    #     mc_sa.fix_spin_(ss=x.ispin)
    #     emc = mc_sa.casci(mo)[0]
    #     mc_sa.analyze(large_ci_tol=0.05)

    #MC-PDFT step
    numeric  = [None]*len(ontop)
    analytic = [None]*len(ontop)
    en_dist  = [None]*len(ontop) # energy array indexed by functional

    for k, ifunc in enumerate(ontop): # Over on-top functionals

        ot='htPBE0' if len(ifunc)>10 else ifunc
        mol.output=x.iname+'_'+ot+'_'+f"{dist:.2f}"+'.log'
        mol.build()
        if len(ifunc) > 10 or ifunc=='ftPBE':
            raise NotImplementedError

        #Initialize to zero
        dip_cms = np.zeros(4*ntdm).tolist()
        abs_pdft = 0
        abs_cas  = 0
        e_pdft  = 0
#-------------------- Energy ---------------------------
        #---------------- Make a PDFT object ---------------------------
        mc = mcpdft.CASSCF(mf, ifunc, x.norb, x.nel, grids_level=9)
        mc.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
        mc.fcisolver.wfnsym = x.irep
        # fname = 'orb_main_'+ out 
        mc.chkfile = fname
        mc.max_cycle_macro = 300

        if 'MC-PDFT' in (analyt + numer): 
            raise NotImplementedError
                
        if 'CMS-PDFT' in (analyt + numer): 
            # mc = mc.state_interaction(weights,'cms').run(mo,ci)
            mc = mc.multi_state(weights,'cms')
            mc.max_cyc=max_cyc
            mc.conv_tol=thresh
            mc.sing_tol=thresh
            mc.nudge_tol=nudge_tol
            mc.run(mo,ci)
            e_states=mc.e_states.tolist() 
            # mo_sa = mc.mo_coeff #SA-CASSCF MOs
            # molden.from_mo(mol, out+'_sa.molden', mc.mo_coeff)
            if dmcFLAG == True:
                print("Working on Analytic CMS-PDFT Dipole")
                id_tdm=-1 #enumerate TDM
                for m in range(x.iroots):
                    for n in range(m):
                        id_tdm+=1
                        shift=id_tdm*4
                        tdm=mc.trans_moment(state=[m,n],unit='Debye')
                        # print(tdm)
                        abs_cms = np.linalg.norm(tdm)
                        dip_cms[shift:shift+3] = tdm
                        dip_cms[shift+3] = abs_cms
            else:
                print("Analytic CMS-PDFT TDM is ignored")
                dip_cms = np.zeros(4*ntdm).tolist()
        
        if 'SA-PDFT' in (analyt + numer): 
            mc = mc.state_average_(weights).run(mo,ci)
            e_states=mc.e_states 
            # mo_sa = mc.mo_coeff #SA-CASSCF MOs
            # molden.from_mo(mol, out+'_sa.molden', mc.mo_coeff)
            if dmcFLAG == True:
                raise NotImplementedError
                # print("Working on Analytic SA-PDFT TDM")
                # for m in range(x.iroots):
                #     shift=m*4
                #     dipoles = mc.dip_moment(state=m, unit='Debye')
                #     dipoles=np.array(dipoles)
                #     abs_cms = np.linalg.norm(dipoles)
                #     dip_cms[shift:shift+3] = dipoles
                #     dip_cms[shift+3] = abs_cms
            else:
                print("Analytic SA-PDFT TDM is ignored")

        if 'SA-CASSCF' in (analyt + numer): 
            # mc = mc.state_average_(weights).run(mo,ci)
            e_states = cas.e_states.tolist() #Note cas object not mc 
            # mo_sa = cas.mo_coeff 
            if dmcFLAG == True:
                print("Working on Analytic SA-CASSCF TDM")
                from functools import reduce
                ci_buf  = [None]*x.iroots
                orbcas = mo[:,cas.ncore:cas.ncore+cas.ncas]
                with mol.with_common_orig((0,0,0)):
                    dip_ints = mol.intor('cint1e_r_sph', comp=3)
                for i in range(x.iroots):
                    mc = mcscf.CASCI(mf, x.norb, x.nel).state_specific_(i)
                    mc.fix_spin_(ss=x.ispin)
                    mc.casci(mo)[0]
                    ci_buf[i] = mc.ci

                #charges = mol.atom_charges()
                #coords = mol.atom_coords()
                #nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
                #mol.set_common_orig_(nuc_charge_center)
                #dip_ints = mol.intor('cint1e_r_sph', comp=3)
                    #ao_dip = mol.intor_symmetric('int1e_r', comp=3)
                id_tdm=-1 #enumerate TDM
                for m in range(x.iroots):
                    for n in range(m):
                        id_tdm+=1
                        shift=id_tdm*4
                        # tdm=mc.trans_moment(state=[m,n],unit='Debye')
                        t_dm1 = mc.fcisolver.trans_rdm1(ci_buf[m], ci_buf[n], cas.ncas, cas.nelecas)
                        t_dm1_ao = reduce(np.dot, (orbcas, t_dm1, orbcas.T))
                        tdm = nist.AU2DEBYE*np.einsum('xij,ji->x', dip_ints, t_dm1_ao)
                        # print(tdm)
                        abs_cms = np.linalg.norm(tdm)
                        oscil = 2/3*abs(e_states[m]-e_states[n])*abs_cms**2
                        print()
                        dip_cms[shift:shift+3] = tdm
                        dip_cms[shift+3] = abs_cms
            else:
                print("Analytic SA-CASSCF TDM is ignored")
                dip_cms = np.zeros(4*x.iroots).tolist()

       
        # ---------------- Numerical Dipoles ---------------------------
        #---------------------------------------------------------------
        if numer == []:
            print("Numerical dipole is ignored")
            dip_num = np.zeros((len(field), 4))
        else:
            for method in numer:
                if method == 'MC-PDFT': 
                    raise NotImplementedError
                    # mo=mo_ss
                elif method == 'CMS-PDFT' or method == 'SA-PDFT':
                    # mo=mo_sa
                    # si_zero=mc.si_mat
                    # ham_zero=mc.ham
                    si_in_zero=mc.si_pdft
                    si_sa_zero=mc.si_mcscf
                    ham_zero=mc.get_heff_pdft()
                    ci = mc.ci
                    print('si_sa_zero\n',si_sa_zero)    
                    print('ham_zero\n',ham_zero) 
                dip_num = numer_run(dist, x, mol, mo, ci, method, field, formula, ifunc, out, dip_cms, \
                    si_sa_zero, si_in_zero, ham_zero, ntdm)
            
        analytic[k] = [dist, abs_cas, abs_pdft] + dip_cms
        numeric [k] = dip_num
        en_dist [k] = [dist, e_casscf, e_pdft] + e_states
    return numeric, analytic, en_dist, mo, ci

def pdtabulate(df, line1, line2): return tabulate(df, headers=line1, tablefmt='psql', floatfmt=line2)

# Get dipoles & energies for a fixed distance
def run(x, field, formula, numer, analyt, mo, ci, dist, ontop, scan, dip_scan, en_scan, ntdm, dmcFLAG=True):
    numeric, analytic, en_dist, mo, ci = \
        get_dipole(x, field, formula, numer, analyt, mo, ci, dist, ontop, ntdm, dmcFLAG=dmcFLAG)

    # Accumulate analytic dipole moments and energies
    for k, ifunc in enumerate(ontop):
        out = 'dmc_'+x.iname+'_'+ifunc+'.txt'
        dip_scan[k].append(analytic[k]) 
        en_scan[k].append(en_dist[k]) 

        # Print & save numeric dipole moments
        if numer != []:
            for method in numer:
                ot='htPBE0' if len(ifunc)>10 else ifunc
                print("Numeric dipole at the bond length %s found with %s (%s)" \
                    %(cs(str(dist)),cs(method),cs(ot)))
                header=['Field',]
                for i in range(x.iroots): 
                    header=header+['X', 'Y', 'Z',] 
                    header.append('ABS ({})'.format(str(i+1)))
                sigfig = (".4f",)+(".4f",)*4*ntdm
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
    return mo, ci



#--------------------- Set Up Parameters for Dipole Moment Curves--------------------
# Set the bond range
bonds = np.array([1.2])
bonds = np.array([1.0])
bonds = np.array([0.969197274])
# bonds = np.array([2.1])
# inc=0.1
inc=0.02
# bonds = np.arange(2.0,2.2+inc,inc) # for energy curves
# bonds = np.arange(1.0,3.0+inc,inc) # for energy curves
# bonds = np.array([1.6,2.0,2.1,2.2,2.3,2.4,2.5]) # for energy curves

# External XYZ frames
# bonds = np.arange(0,31,1) 
# bonds = np.arange(0,2,1) 

# Set field range
# field = np.linspace(1e-3, 1e-2, num=1)
field = np.linspace(1e-3, 1e-2, num=3)
field = np.linspace(1e-3, 1e-2, num=19)
field = np.linspace(1e-3, 1e-2, num=2)
field = np.linspace(1e-3, 1e-2, num=10)
field = np.linspace(5e-3, 4e-3, num=2)
field = np.array([0.001])
# inc= 1e-3
# field = np.arange(inc, 1e-2, inc)
thresh = 5e-9
max_cyc = 100
nudge_tol = 1e-4


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
numer  = []
analyt = []
# numer = ['SS-PDFT']
numer = ['CMS-PDFT']
# numer = ['SA-PDFT']
# analyt = ['MC-PDFT','CMS-PDFT']
analyt = ['CMS-PDFT']
# analyt = ['SA-CASSCF']
# analyt = ['SA-PDFT']
dmcFLAG=False 
dmcFLAG=True


# List of molecules and molecular parameters
geom_phenol_opt= '''
C       -1.182853111      1.200708431      0.00000000
C        0.212397551      1.216431879      0.00000000
C       -1.877648981     -0.011392736      0.00000000
H        0.770332141      2.159905341      0.00000000
H       -2.974311304     -0.021467798      0.00000000
C        0.919152698      0.005414786      0.00000000
C       -1.164162291     -1.211169261      0.00000000
H       -1.697713380     -2.169828607      0.00000000
C        0.232754869     -1.212093364      0.00000000
H        0.793712181     -2.157907002      0.00000000
H       -1.735970731      2.149003232      0.00000000
O        2.284795534      0.064463230      0.00000000
H        2.640406066     -0.838707361      0.00000000
'''
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
geom_h2co= '''
H       -0.000000000      0.950627350     -0.591483790
H       -0.000000000     -0.950627350     -0.591483790
C        0.000000000      0.000000000      0.000000000
O        0.000000000      0.000000000     {}
'''
geom_fluorobenzene= '''
C        0.183499999      1.226261448      0.000000000
C       -1.221421109      1.211246096      0.000000000
C        0.855142196      0.000760301      0.000000000
C       -1.923298868     -0.005687081      0.000000000
C        0.189109961     -1.227818476      0.000000000
H       -3.021636707     -0.008151529      0.000000000
H        0.764791554     -2.162287676      0.000000000
C       -1.215815905     -1.219374361      0.000000000
H       -1.759511085     -2.174539458      0.000000000
H        0.754810367      2.163408242      0.000000000
F        2.224458801      0.003499952      0.000000000
H       -1.769650313      2.163822617      0.000000000
'''

# GS geom_h2co equilibrium CMS-PDFT (tPBE) (6e,6o) / julccpvdz
# O        0.000000000      0.000000000      1.214815430 

class Molecule:
    def __init__(self, iname, geom, nel, norb, cas_list, init=0, iroots=1, istate=0, 
                icharge=0, isym='C1', irep='A', ispin=0, ibasis='julccpvdz'):
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

# See class Molecule for the list of variables.
h2co_6e6o        = Molecule('h2co_6e6o',    geom_h2co,        6,6, [6,7,8,9,10,12],         init=1.20, iroots=2)
phenol_8e7o_sto  = Molecule('phenol_8e7o_sto',geom_phenol,    8,7, [19,23,24,25,26,27,28], iroots=2, ibasis='sto-3g')
phenol_8e7o      = Molecule('phenol_8e7o',  geom_phenol,      8,7, [19,23,24,25,31,32,34], init=0.0, iroots=2)
phenol_8e7o_opt  = Molecule('phenol_opt',  geom_phenol_opt,   8,7, [18,23,24,25,31,33,34], init=0.0, iroots=2)
OH_phenol_10e9o  = Molecule('OH_phenol_10e9o', geom_OH_phenol,10,9,[19,20,23,24,25,26,31,33,34], init=1.3, iroots=2)
OH_phenol3_10e9o =  copy.deepcopy(OH_phenol_10e9o)
OH_phenol3_10e9o.iroots=3
phenol_8e7o_sto_3      =  copy.deepcopy(phenol_8e7o_sto)
phenol_8e7o_sto_3.iroots=3
spiro_11e10o  = Molecule('spiro_11e10o','frames',11,10,[35,43,50,51,52,53, 55,76,87,100], iroots=2, icharge=1, ispin=1)
fluorobenzene_6e6o  = Molecule('fluorobenzene_6e6o',geom_fluorobenzene, 6,6, [23,24,25,31,32,34],iroots=3)
#Select species for which the dipole moment curves will be constructed
species=[h2co_6e6o]
# species=[phenol_8e7o]
species=[phenol_8e7o_sto]
species=[spiro_11e10o]
species=[OH_phenol3_10e9o]
species=[OH_phenol_10e9o]
species=[phenol_8e7o]
species=[fluorobenzene_6e6o]
species=[phenol_8e7o_opt]
species=[phenol_8e7o_sto_3]

# ---------------------- MAIN DRIVER OVER DISTANCES -------------------
dip_scan = [ [] for _ in range(len(ontop)) ]
en_scan  = [ [] for _ in range(len(ontop)) ] # Empty lists per functional
scan=False if len(bonds)==1 else True #Determine if multiple bonds are passed
#In case if numer and analyt are not supplied
# exec('try:numer\nexcept:numer=None')
# exec('try:analyt\nexcept:analyt=None')

for x in species:
    # Get MOs before running a scan
    y=copy.deepcopy(x)
    y.geom=y.geom.format(y.init)
    mo, ci = init_guess(y, analyt, numer) if x !=spiro_11e10o else 0 #spiro MOs should be provided manually
    ntdm=np.math.factorial(x.iroots)//(np.math.factorial(x.iroots-2)*2)

    for i, dist in enumerate(bonds, start=0):
        # Update the bond length in the instance
        if i==0: template=x.geom
        x.geom=template.format(dist)

        # MOs and CI vectors are taken from the previous point
        mo, ci = run(x, field, formula, numer, analyt, mo, ci, dist, ontop, scan, dip_scan, en_scan, ntdm, dmcFLAG=dmcFLAG)
        
        dip_head = ['Distance','CASSCF','MC-PDFT']
        for j in range(x.iroots): 
            dip_head=dip_head+['X', 'Y', 'Z',] 
            dip_head.append('ABS ({})'.format(cs(str(j+1))))
        dip_sig = (".2f",)+(".4f",)*(2+4*ntdm)
        
        en_head=['Distance', 'CASSCF', 'MC-PDFT']
        for method in analyt:
            for jj in range(x.iroots): 
                line='%s ({})'.format(cs(str(jj+1))) % method
                en_head.append(line)
            en_sig = (".2f",)+(".8f",)*(2+x.iroots)

        for k, ifunc in enumerate(ontop):
            out = x.iname+'_'+ifunc+'.txt'
            if analyt!=[]:
                print("Analytic dipole moments found with %s" %cs(ifunc))
                dip_table = pdtabulate(dip_scan[k], dip_head, dip_sig)
                print(dip_table)

            print("Energies found with %s" %cs(ifunc))
            en_table = pdtabulate(en_scan[k], en_head, en_sig)
            print(en_table)
            action='w' if numer==[] else 'a'
            with open(out, action) as f:
                f.write("The on-top functional is %s \n" %(ifunc))
                f.write(en_table)
                f.write('\n')
                if analyt!=[]:
                    f.write(dip_table)
                    f.write('\n')
