import numpy as np
import scipy.constants as const

class equilibriumConstant:
    def __init__(self, kref, dH, product_charges, reactant_charges, new_name):
        self.kref  =  kref
        self.dH    =  dH
        self.prod  =  product_charges
        self.react =  reactant_charges
        self.name  =  new_name

# ---- constants ----
is0_Ka_atp      = equilibriumConstant(2.512e-08, -6.30, [1,4], [3], 'Ka_atp')
is0_Kb_mg_atp   = equilibriumConstant(1.514e6,   22.90, [2], [4,2], 'Kb_mg_atp')
is0_Kb_mg_hatp  = equilibriumConstant(4.266e3,   16.90, [1], [2,3], 'Kb_mg_hatp')
is0_Ka_adp      = equilibriumConstant(6.607e-08, -5.60, [1,3], [2], 'Ka_adp')
is0_Kb_mg_adp   = equilibriumConstant(4.466e4,   19.0,  [1], [2,3], 'Kb_mg_adp')
is0_Kb_mg_hadp  = equilibriumConstant(3.163e2,   12.50, [None], [2,2], 'Kb_mg_hadp')
is0_Ka_pho      = equilibriumConstant(6.026e-8,  12.2,  [2], [None], 'Ka_pho')
is0_Kb_mg_pho   = equilibriumConstant(5.128e2,   8.19,  [None], [2,2], 'Kb_mg_pho')
is0_Ka_pcr      = equilibriumConstant(8.854e-6,  2.66,  [2], [None], 'Ka_pcr')
is0_Kb_mg_pcr   = equilibriumConstant(2.320e2,   8.19,  [None], [2,2], 'Kb_mg_pcr')

Kref_CK  = equilibriumConstant(2.58e8,  -17.55, [4], [3,2,1], 'Kref_CK')
Kref_ATP = equilibriumConstant(2.946e-1,-20.50, [3,2,1], [4], 'Kref_ATP')


def gatp(pcr_additions_mM):

    import numpy as np
    import scipy.constants as const

    #for Mg Valuves
    ref_pcr_mM = np.array([1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
    ref_pcr = ref_pcr_mM / 1000
    # Temperature
    temp37 = 273.15 + 37
    temp25 = 273.15 + 25

    # Mg (reference, will interpolate)
    freeMg_ref = [
        0.000407972, 0.000395923, 0.000377038, 0.000364626,
        0.0003487,   0.00033737,  0.000323291, 0.000309397,
        0.000299109, 0.000292253, 0.000281548,
    ]

    phosphate = 0.010
    pH = 7.2
    basal_IS = 0.168

    # ---- convert PCr ----
    pcr_series = np.array(pcr_additions_mM) / 1000
    """
    # MAYBE NOT ACCURATE
    ref_steps = np.linspace(0, 1, len(freeMg_ref))
    target_steps = np.linspace(0, 1, len(pcr_series))
    freeMg = np.interp(pcr_series, ref_pcr, freeMg_ref)
    
    CHANGED
    """
    ref_pcr = np.array([1,3,6,9,12,15,18,21,24,27,30]) / 1000
    mg_lookup = dict(zip(ref_pcr, freeMg_ref))

    current_pcr = 0
    freeMg = []

    for p_add in pcr_series:
        current_pcr += p_add

        if current_pcr in mg_lookup:
            freeMg.append(mg_lookup[current_pcr])
       #DOESNT WORK WITHOUT     
        else:
            freeMg.append(np.interp(current_pcr, ref_pcr, freeMg_ref))

    # ---- ionic strength ----
    ionicStrengths = []

    basal_IS = 0.168

    atp = 0.005
    tris_atp = 2 * atp
    atp_is = 0.5 * (atp * (-4)**2 + tris_atp * (1)**2)

    # Step 1: initial PCr
    current_pcr = pcr_series[0]

    tris = 2 * current_pcr
    pcr_is = 0.5 * (current_pcr * (-2)**2 + tris * (1)**2)

    ionicStrengths.append(basal_IS + atp_is + pcr_is)

    # Remaining steps
    for pcr_add in pcr_series[1:]:
        current_pcr += pcr_add  # accumulate correctly

        tris = 2 * current_pcr
        pcr_is = 0.5 * (current_pcr * (-2)**2 + tris * (1)**2)

        ionicStrengths.append(basal_IS + atp_is + pcr_is)
    print("IONIC STRENGTH")
    for idx, step in enumerate(ionicStrengths):
        print('{0: <8}{1: <12.6}'.format(idx+1, step*1000))

    # ---- CK Keq ----
    CK_k2 = vant_hoff(temp25, temp37, Kref_CK.kref, Kref_CK.dH)

    CK_kref_titration = []
    for ion_str in ionicStrengths:
        gamma = solve_gamma(Kref_CK.prod, Kref_CK.react, temp37, ion_str)
        CK_kref_titration.append(CK_k2 / gamma)

    Keq_CK = calc_CK_Keq_steps(
        ionicStrengths, temp37, CK_kref_titration, pH, freeMg
    )
    print("Creatine Kinase Krefs")
    for idx, step in enumerate(CK_kref_titration):
        print('{0: <8}{1: <#12.6}'.format(idx+1, step))
    print("Keq CK")
    for idx, step in enumerate(Keq_CK):
        print('{0: <8}{1: <#12.6}'.format(idx+1, step))

    # ---- ATP Keq ----
    ATP_k2 = vant_hoff(temp25, temp37, Kref_ATP.kref, Kref_ATP.dH)

    ATP_kref_titration = []
    for ion_str in ionicStrengths:
        gamma = solve_gamma(Kref_ATP.prod, Kref_ATP.react, temp37, ion_str)
        ATP_kref_titration.append(ATP_k2 / gamma)

    Keq_ATP = calc_ATP_Keq_steps(
        ionicStrengths, temp37, ATP_kref_titration, pH, freeMg
    )
    print("ATP hydrolysis Krefs")
    for idx, step in enumerate(ATP_kref_titration):
        print('{0: <8}{1: <#12.6}'.format(idx+1, step))


    Keq_ATP = calc_ATP_Keq_steps(ionicStrengths, temp37, ATP_kref_titration, pH, freeMg)

    # Print the Results
    print('{0: <8}{1: <16}'.format('Step', 'ATP Hydrolysis K`'))
    print('{:-^25}'.format(''))

    for idx, step in enumerate(Keq_ATP):
        print('{0: <8}{1: <#12.6}'.format(idx+1, step))
    # ---- ΔG° ----
    dG_std_ATP = []
    # Temperature of the assay
    t37 = 273.15 + 37
# iterate through each step of the assay 
# and calculate the G`° of ATP Hyd.
    for keq in Keq_ATP:
        
        # solve the equation and store the value in a temporary variable
        dG_std = -const.R * t37 * np.log(keq)
        
        # Store the value on our list
        dG_std_ATP.append(dG_std)

    # Print the Results
    print('{0: <8}{1: <16}'.format('Step', 'ΔG°` of ATP Hydrolysis (kJ/mol)'))
    print('{:-^39}'.format(''))

    for idx, step in enumerate(dG_std_ATP):
        print('{0: <8}{1: <#12.6}'.format(idx+1, step/1000))

    # ---- concentrations (by what is added) ----
  # Starting concentrations
    ATP = 0.005
    Cr  = 0.005

    pcr_additions = np.array(pcr_additions_mM) / 1000

    step_concentrations = []

    # Step 1: initial PCr (first element)
    PCr_initial = pcr_additions[0]

    step_concentrations.append(
        calc_step_one_conc(Keq_CK[0], ATP, Cr, PCr_initial)
    )

    # Remaining steps: use additions directly
    for i in range(1, len(pcr_additions)):
        prev = step_concentrations[i-1]

        step_concentrations.append(
            calc_step_conc(
                Keq_CK[i],
                prev.atp,
                prev.cr,
                prev.adp,
                prev.pcr,
                pcr_additions[i]   # <-- key change
            )
        )
        # Print the Results
    print('{: ^60}'.format('Concentrations over the CK Clamp (mM)'))
    print('{0: ^12}{1: ^12.6}{2: ^12.6}{3: ^12.6}{4: ^12.6}'.format('Step', 'ATP', 'ADP', 'PCr', 'Cr'))
    print('{:-^60}'.format(''))

    for idx, step in enumerate(step_concentrations):
        print('{0: ^12}{1: ^12.6}{2: ^12.6}{3: ^12.6}{4: ^12.6}'.format(idx+1, 
                                                                        step.atp*1000, 
                                                                        step.adp*1000, 
                                                                        step.pcr*1000, 
                                                                        step.cr*1000))
    # ---- ΔG′ ----
    dG_ATP = []

    for i, step in enumerate(step_concentrations):
        dG = dG_std_ATP[i] + const.R * temp37 * np.log(
            step.adp * phosphate / step.atp
        )
        dG_ATP.append(dG)

    return dG_ATP
def solve_gamma(products, reactants, temp, ionic_strength):
    '''
    products        =  list of all charged product ions
    reactants       =  list of all charged reactant ions
    temp            =  temperature in Kelvins
    ionic_strength  =  ionic strength in Molarity 
    
    Returns the calculated Γ 
    '''

    # Function to solve for Γ 
    # Debye–Hückel
    Am = 3*(-16.39023 + (261.3371/temp) + 3.3689633*np.log(temp) - 1.437167*(temp/100) + 0.111995*(temp/100)**2)
    # constant with units of kg**1/2 mol**-1/2
    B = 1.6
    
    # reactants = list of the charge carried by each reactant ionic species
    reactant_gammas = []
    # If there isn't a charged species, pass in None to ensure the returned value divides by 1
    if reactants[0] == None:
        reactant_gammas = [1]
    else:
        for react in reactants:
            reactant_gammas.append(np.e**((-Am * np.sqrt(ionic_strength) * react**2)/(1 + B*np.sqrt(ionic_strength))))
    
    
    # products = list of the charge carried by each product ionic species    
    product_gammas = []
    # if there isn't a charged species, pass in None to ensure the returned value divides by 1
    if products[0] == None:
        product_gammas = [1]
    else:
        for prod in products:
            product_gammas.append(np.e**((-Am * np.sqrt(ionic_strength) * prod**2)/(1 + B*np.sqrt(ionic_strength))))
    
    return np.prod(product_gammas)/np.prod(reactant_gammas)

def vant_hoff(temp1, temp2, Kref1, dH):
    
    '''
    temp1  =   temperature for constant Kref1 (Kelvins)
    Kref1  =   constant at temp1 and ionic strength 0 
                 [the original constant from which Kref2 will be adjusted]
    temp2  =   temperature for constant Kref2 (Kelvins)
                 [the new temperature to which Kref1 is to be adjusted] 
    dH     =   ΔH° associated with Kref1 at temp1 and ionic strength 0 (kilojoules)
    
    Returns the modified constant at ionic strength 0, temp2
    '''
    
    # note, constants stored in the data table are in kJ; they are converted inline.
    
    return np.e**((-1000*dH/const.R)*(1/temp2 - 1/temp1) + np.log(Kref1))


def calc_step_one_conc(Keq_CK, ATP, Cr, PCr):
    a = (Keq_CK - 1)
    b = (Keq_CK*PCr + ATP + Cr)
    c = -ATP*Cr
    
    x = (-1*b + np.sqrt((b**2) - 4*(a*c))) / (2*a)
    
    atp = ATP-x
    cr  = Cr-x
    adp = x
    pcr = PCr+x
    
    concentrations = CK_concentration(atp, cr, adp, pcr)
    
    return concentrations
class CK_concentration:
    def __init__(self, atp, cr, adp, pcr):
        self.atp  =  atp
        self.cr   =  cr
        self.adp  =  adp
        self.pcr  =  pcr
def compute_ionic_strengths(pcr_series):
    ionicStrengths = []
    basal_IS = 0.168

    atp = 0.005
    tris_atp = 2 * atp
    atp_is = 0.5 * (atp*(-4)**2 + tris_atp*(1)**2)

    for pcr in pcr_series:
        tris = 2 * pcr
        pcr_is = 0.5 * (pcr*(-2)**2 + tris*(1)**2)

        total_IS = basal_IS + atp_is + pcr_is
        ionicStrengths.append(total_IS)

    return ionicStrengths
def calc_step_conc(Keq_CK, ATP, Cr, ADP, PCr_pre, PCr_add):
    
    a = (Keq_CK - 1)
    b = Keq_CK*(ADP - PCr_pre - PCr_add) - ATP - Cr
    c = Keq_CK*ADP*PCr_pre + Keq_CK*ADP*PCr_add - ATP*Cr
    
    disc = b**2 - 4*a*c
    if disc < 0:
        disc = 0  # numerical safety
    
    x = (-b - np.sqrt(disc)) / (2*a)
    
    atp = ATP + x
    cr  = Cr + x
    adp = ADP - x
    pcr = PCr_pre + (PCr_add - x)
    
    return CK_concentration(atp, cr, adp, pcr)
def calc_step_concentrations(Keq_CK, pcr_series):
    ATP = 0.005
    Cr  = 0.005

    step_concentrations = []

    # first step
    step_concentrations.append(
        calc_step_one_conc(Keq_CK[0], ATP, Cr, pcr_series[0])
    )

    # subsequent steps
    for i in range(1, len(pcr_series)):
        prev = step_concentrations[i-1]

        PCr_target = pcr_series[i]
        PCr_prev   = prev.pcr

        PCr_add = PCr_target - PCr_prev

        step_concentrations.append(
            calc_step_conc(
                Keq_CK[i],
                prev.atp,
                prev.cr,
                prev.adp,
                prev.pcr,
                PCr_add
            )
        )

    return step_concentrations
def calc_ATP_Keq_steps(ionicStrengths, tempK, ATP_kref_titration, pH, freeMg):
    '''
    ionicStrengths =   list of ionic strength values over titration. Unit = Molarity
    temp           =   temperature of the assay in Kelvin
    pH             =   the pH of the assay
    freeMg         =   list of free Magnesium concentration over the titration. Unit = Molarity.
    
    Returns a list of the apparent ATP Hydrolysis Eq. Constants (Keq_ATP)
    '''
    
    # Convert input variables into necessary units
    proton = 10**-pH
    
    # Temp. of all reference constants
    temp25 = 273.15 + 25
    
    # List to store ATP Equilibrium Constants calculated for each step of the ATP-Clamp titration
    Keq_ATP = []
    
    # Make sure the 2 lists are equal
    if len(ionicStrengths) != len(freeMg):
        raise ValueError('Size mismatch: the length of ionicStrengths and freeMg are not equal')
        
    # A list of the constants contained in the ATP Hydrolysis reaction
    atp_Ks = [is0_Ka_atp,
              is0_Kb_mg_atp,
              is0_Kb_mg_hatp,
              is0_Ka_adp,
              is0_Kb_mg_adp,
              is0_Kb_mg_hadp,
              is0_Ka_pho,
              is0_Kb_mg_pho]
    
    # iterate through each ionic strength change in the assay
    for index, ion_str in enumerate(ionicStrengths):
        # a dictionary to store the modified equilibrium constants at each step in the titration
        new_Kabs = {}
        
        # Free Magnesium at the specific ionic strength
        mg = freeMg[index]
        
        # iterate through the acid and Mg-binding Krefs in the ATP Eq reaction
        for k1 in atp_Ks:
            
            # Use Van't Hoff 
            # new kref (k2) adjusted from I=0, T=25°C to I=0, T=tempK
            k2 = vant_hoff(temp25, tempK, k1.kref, k1.dH)
            
            # Use Debye–Hückel
            # new kref (k3) adjusted from I=0,T=25°C to new I, T=tempK
            # the k1 object contains the product and reactant charges
            k3 = k2/solve_gamma(k1.prod, k1.react, tempK, ion_str)
            
            # Store the new equilibrium constant 
            new_Kabs[k1.name] = k3
        
        # Modified Equilibrium Constant for the ATP Hydrolysis Reaction
        react = (1 + proton/new_Kabs['Ka_atp'] + new_Kabs['Kb_mg_atp']*mg 
                 + new_Kabs['Kb_mg_hatp']*proton*mg/new_Kabs['Ka_atp'])
        
        prod_adp = (1 + proton/new_Kabs['Ka_adp'] + new_Kabs['Kb_mg_adp']*mg 
                    + new_Kabs['Kb_mg_hadp']*proton*mg/new_Kabs['Ka_adp'])
        
        prod_pho = (1 + proton/new_Kabs['Ka_pho'] + new_Kabs['Kb_mg_pho']*mg)

        Kf_ATP = (ATP_kref_titration[index] * prod_adp * prod_pho)/(react * proton)
        
        Keq_ATP.append(Kf_ATP)
    
    return Keq_ATP
def calc_CK_Keq_steps(ionicStrengths, tempK, CK_kref_titration, pH, freeMg):
    '''
    ionicStrengths =   list of ionic strength values over titration. (Molarity)
    temp           =   temperature of the assay in Kelvins
    pH             =   the pH of the assay
    freeMg         =   list of free Magnesium concentration over the titration. (Molarity)
    
    Returns a list of the apparent CK Eq. Constants (Keq_CK)
    '''
    # Convert input variables into necessary units
    proton = 10**-pH
    temp25 = 273.15 + 25
    
    # List to store CK Equilibrium Constants calculated for each step of the CK-Clamp titration
    Keq_CK = []
    
    # Make sure the 2 lists are equal
    if len(ionicStrengths) != len(freeMg):
        raise ValueError('Size mismatch: the length of ionicStrengths and freeMg are not equal')
        
    # A list of the constants contained in the Creatine Kinase reaction
    ck_Ks =  [is0_Ka_atp,
              is0_Kb_mg_atp,
              is0_Kb_mg_hatp,
              is0_Ka_adp,
              is0_Kb_mg_adp,
              is0_Kb_mg_hadp,
              is0_Ka_pcr,
              is0_Kb_mg_pcr]
    
    # iterate through each ionic strength change in the CK Clamp
    for index, ion_str in enumerate(ionicStrengths):
        # a dictionary to store the modified equilibrium constants at each step in the titration
        new_Kabs = {}
        
        # Free Magnesium at the specific ionic strength
        mg = freeMg[index]
        
        # iterate through the acid and Mg-binding Krefs in the CK Eq reaction
        for k1 in ck_Ks:
            
            # Use Van't Hoff 
            # new kref (k2) adjusted from I=0, T=25°C to I=0, T=tempK
            k2 = vant_hoff(temp25, tempK, k1.kref, k1.dH)
            
            # Use Debye–Hückel
            # new kref (k3) adjusted from I=0,T=25°C to new I, T=tempK
            # the k1 object contains the product and reactant charges
            k3 = k2/solve_gamma(k1.prod, k1.react, tempK, ion_str)
            
            # Store the new equilibrium constant 
            new_Kabs[k1.name] = k3
        
        # Modified Equilibrium Constant for the Creatine Kinase Reaction
        products = (1 + proton/new_Kabs['Ka_atp'] + new_Kabs['Kb_mg_atp']*mg 
                    + new_Kabs['Kb_mg_hatp']*proton*mg/new_Kabs['Ka_atp'])
        
        react_adp = (1 + proton/new_Kabs['Ka_adp'] + new_Kabs['Kb_mg_adp']*mg 
                     + new_Kabs['Kb_mg_hadp']*proton*mg/new_Kabs['Ka_adp'])
        
        react_pcr = (1 + proton/new_Kabs['Ka_pcr'] + new_Kabs['Kb_mg_pcr']*mg)
    
        Kf_CK = CK_kref_titration[index] * proton * products/(react_adp * react_pcr)
        
        # Append the apparent Keq to our holding list 
        Keq_CK.append(Kf_CK)

    
    return Keq_CK
    
def main():
    print(gatp([1, 2, 3, 6, 3, 15]))

if __name__ == "__main__":
    main()
