# Sean Dougherty
# 8/8/2022
# calibrating the control selection technique for the convolution AGN merger

# import libraries
import numpy as np
from numpy import random
np.seterr(all="ignore")
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import scipy
from scipy.stats import distributions

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
gamma = 1.4 # for k correction calculation

from time import sleep
from tqdm import tqdm

import pickle

from multiprocessing import Pool, freeze_support, RLock
from functools import partial


# INPUTS
which_bins = 1 # only match out to this kpc bin
iso_method = 'none' # 'high_prob', 'proj_mask' # different cuts to the isolated dataframe
iso_min_prob = 0.2 # look at distribution and decide
dif_lim = 0.5
iters = 100

# define main function
def main():
    # load in convolution output
    PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/conv_prob/'
    with open(PATH+'ALL_FIELDS_HIGH_MASS_8.06.pkl', 'rb') as f: # MASS_PARTNER > 10
        pam_df = pickle.load(f)
    
    # this is a dictionary of dataframes for kpc separation bin, pairs have been unscrambled to simply have 
    # a df of galaxies with their own properties and pair probability

    # kpc projected separation bins I want:
    keys = list(pam_df.keys())
    bin_dfs = { key: pam_df[key] for key in keys[:which_bins] }
        
    
    # have a dictionary to hold info for each bin, inside of it will be however many dictionaries for:
    bin_storage = {}
    for key in keys:
        bin_storage[key] = []
    
# def method(): # a proxy function 
    # %%%%%% Parallelise here %%%%%%% #
    # %%%%%% return 'iter_df' from each one %%%%%%% #
    for i in range(iters):
        controls = orchestrate_control(bin_dfs, iso_method)

        # calculate p value from distribution
        for key in controls.keys():
            storage = {}
            pair_df = bin_dfs[key]
            iso_df = controls[key]
            # calculate a p value for the entire sample:
            n_controls = len(iso_df)
            all_Dn_m, all_p_m = ks_weighted( pair_df['mass'], iso_df['mass'], pair_df['pair_prob'], np.ones_like(iso_df['mass']) )
            all_Dn_z, all_p_z = ks_weighted( pair_df['z'], iso_df['z'], pair_df['pair_prob'], np.ones_like(iso_df['z']) )
            all_Dn_s, all_p_s = ks_weighted( pair_df['2sig'], iso_df['2sig'], pair_df['pair_prob'], np.ones_like(['2sig']) )
            storage['all_Dn_m'] = all_Dn_m
            storage['all_Dn_z'] = all_Dn_z
            storage['all_Dn_s'] = all_Dn_s
            storage['all_p_m'] = all_p_m
            storage['all_p_z'] = all_p_z
            storage['all_p_s'] = all_p_s
            storage['all_n'] = n_controls

            # also calculate an AGN fraction:
            iso_df['iXAGN'] = [0]*len(iso_df)
            iso_df['iDoAGN'] = [0]*len(iso_df)
            iso_df['iStAGN'] = [0]*len(iso_df)
            LX_AGN = 42
            iso_df.loc[ np.log10(iso_df['LX']) > LX_AGN, 'iXAGN' ] = 1
            iso_df.loc[ iso_df['IR_AGN_DON'] == 1, 'iDoAGN' ] = 1
            iso_df.loc[ iso_df['IR_AGN_STR'] == 1, 'iStAGN' ] = 1
            iX_frac = np.sum( iso_df['iXAGN'] ) / len( iso_df['iXAGN'] )
            iDo_frac = np.sum( iso_df['iDoAGN'] ) / len( iso_df['iDoAGN'] )
            iSt_frac = np.sum( iso_df['iStAGN'] ) / len( iso_df['iStAGN'] )
            storage['X_fAGN'] = iX_frac
            storage['DIR_fAGN'] = iDo_frac
            storage['SIR_fAGN'] = iSt_frac

            # now p values for each field
            for field in controls[key]['field'].unique():
                n_controls = len(iso_df.loc[iso_df['field']==field])
                Dn_m, p_m = ks_weighted( pair_df.loc[pair_df['field']==field, 'mass'], 
                                    iso_df.loc[iso_df['field']==field, 'mass'],
                                    pair_df.loc[pair_df['field']==field, 'pair_prob'], 
                                    np.ones_like(iso_df.loc[iso_df['field']==field, 'mass']) )
                Dn_z, p_z = ks_weighted( pair_df.loc[pair_df['field']==field, 'z'], 
                                    iso_df.loc[iso_df['field']==field, 'z'],
                                    pair_df.loc[pair_df['field']==field, 'pair_prob'], 
                                    np.ones_like(iso_df.loc[iso_df['field']==field, 'z']) )
                Dn_s, p_s = ks_weighted( pair_df.loc[pair_df['field']==field, '2sig'], 
                                    iso_df.loc[iso_df['field']==field, '2sig'],
                                    pair_df.loc[pair_df['field']==field, 'pair_prob'], 
                                    np.ones_like(iso_df.loc[iso_df['field']==field, '2sig']) )
                storage[field+'_Dn_m'] = all_Dn_m
                storage[field+'_Dn_z'] = all_Dn_z
                storage[field+'_Dn_s'] = all_Dn_s
                storage[field+'_p_m'] = all_p_m
                storage[field+'_p_z'] = all_p_z
                storage[field+'_p_s'] = all_p_s
                storage[field+'_n'] = n_controls

            # convert storage into a df
            iter_df = pd.DataFrame.from_dict(storage)
            # append it to the appropritiate storage dictionary:
            bin_storage[key].append(iter_df) # this way I can just concatenate at the end
            
    # now concat iteration dfs:
    for key in bin_storage.keys():
        key_final_df = pd.DataFrame( np.concatenate(bin_storage[key]), columns=pd.DataFrame(bin_storage[key][0]).columns )
        # and save:
        key_final_df.to_csv(PATH+'control_cal/NAME.csv', index=False)
        print('Iteration data saved in bin {}'.format(key))
        
    return         
    
    
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def orchestrate_control(bin_dfs, iso_method): # may want to write this to do one bin at a time
    
    all_bin_df = pd.concat([all_df[key] for key in all_df]) # going to start with just one bin, but for flexibility later
    
    # generate the isolated df based on paired sources to exclude:
    iso_df = generate_iso(all_bin_df, iso_method) ### ~~~ UNSURE IF THIS WOULD WORK PAST THE FIRST BIN ~~~ ####
    
    iso_dict = {}
        
    # generate PDFs for each bin and field
    # 10 bin dict -> 6 field dicts in each
    zms = {}
    Pzms = {}
    
    controls = {}
    
    bin_field_count_distr = {}
    
    # initialize control ID selection list (to work across all bins)
    control_ID = {}
    drawn_ID = {}
    
    # count fractional pairs in each bin
    pair_count = []
    
    print('generating PDFs')
    for i, key in enumerate(bin_dfs):
        print(key)
        iso_dict[key] = iso_df
        zms[key] = {}
        Pzms[key] = {}
        controls[key] = []
        bin_field_count_distr[key] = {}
        pair_count.append( np.sum(bin_dfs[key].loc[ :, 'pair_prob']) )
        control_ID[key] = [] # append control IDs to this...
        drawn_ID[key] = []
            
        for j, field in enumerate(list(bin_dfs[key]['field'].unique())): # may break if a bin is missing a field for some reason...
            probs = bin_dfs[key].loc[ bin_dfs[key]['field'] == field, 'pair_prob' ]
            props_df = bin_dfs[key].loc[ bin_dfs[key]['field'] == field, ['z', 'mass', '2sig'] ]
            gal_props, gal_PDF = gengalPDF(props_df, probs, sensitivity=50)

            zms[key][field] = gal_props
            Pzms[key][field] = gal_PDF
        
            # generate PDF for each field within the bins based on counts
            bin_field_count_distr[key][field] = np.sum(bin_dfs[key].loc[ bin_dfs[key]['field'] == field, 'pair_prob'] ) #/ pair_count[i]
    
    # generate PDF for bins based on counts
    bin_count_distr = pair_count / np.sum(pair_count)
    
    total_frac_pairs = np.sum(pair_count)
    
    
    for key in list(zms.keys()): ### ~~~ EASILY PARALLELIZIABLE ~~~ ###
        for field in list(zms[key].keys()):
            print('selecting controls in {0}, {1}'.format(field, bin_field_count_distr[key][field]))
            drawn_z, drawn_m, drawn_s = gen_gal(zms[key][field], Pzms[key][field], n=round(1000*bin_field_count_distr[key][field]))
            # get iso
            control = get_control(iso_df.loc[ (iso_df['field'] == field) ], drawn_z, drawn_m, drawn_s, dif_lim=dif_lim)
            controls[key].append(control)
            
    # get control selection into a df for each bin
    ccontrols = {}
    for key in controls.keys():
        ccontrols[key] = pd.DataFrame( np.concatenate(controls[key]), columns=pd.DataFrame(controls[key][0]).columns )
        
    return ccontrols

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def generate_iso(all_df, iso_method):
    
    print('assembling isolated sources')

    # load in all fields
    fields = list(all_df['field'].unique())
    field_dfs = {}
    PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/Pair Project - Updated Data/'
    cPATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/'
    mass_lo = 8.5

    # load in 

    iso_df = {}

    for field in fields:

        # for each loop, load in the master catalog
        if field == 'COSMOS':
            df = pd.read_csv(cPATH+'select_COSMOS2020.csv')
            df['SIG_DIFF'] = df['lp_zPDF_u68'] - df['lp_zPDF_l68']
            df = df[ (df['lp_type'] != 1) & (df['lp_type'] != -99) & (df['lp_mass_med'] > (mass_lo-1)) & 
               (df['FLAG_COMBINED'] == 0) ]

            df = df.rename(columns={'ALPHA_J2000':'RA', 'DELTA_J2000':'DEC', 'lp_mass_med':'MASS', 'lp_zPDF':'ZPHOT_PEAK',
                                   'F0.5-10_2015':'FX', 'UVISTA_H_MAG_APER2':'HMAG'})
            df = df.reset_index(drop=True)
        else:
            df = pd.read_csv(PATH+'CANDELS_Catalogs/CANDELS.'+field+'.1018.Lx_best.wFx_AIRD.csv')
            # add the zphot data
            df_phot = pd.read_csv(PATH+'redshift_catalogs.full/zcat_'+field+'_v2.0.cat', names=['file','ID','RA','DEC','z_best',
                        'z_best_type','z_spec','z_spec_ref','z_grism','mFDa4_z_peak','mFDa4_z_weight','mFDa4_z683_low',
                        'mFDa4_z683_high','mFDa4_z954_low','mFDa4_z954_high','HB4_z_peak','HB4_z_weight','HB4_z683_low',
                        'HB4_z683_high','HB4_z954_low','HB4_z954_high','Finkelstein_z_peak','Finkelstein_z_weight',
                        'Finkelstein_z683_low','Finkelstein_z683_high','Finkelstein_z954_low','Finkelstein_z954_high',
                        'Fontana_z_peak','Fontana_z_weight','Fontana_z683_low','Fontana_z683_high','Fontana_z954_low',
                        'Fontana_z954_high','Pforr_z_peak','Pforr_z_weight','Pforr_z683_low','Pforr_z683_high',
                        'Pforr_z954_low','Pforr_z954_high','Salvato_z_peak','Salvato_z_weight','Salvato_z683_low',
                        'Salvato_z683_high','Salvato_z954_low','Salvato_z954_high','Wiklind_z_peak','Wiklind_z_weight',
                        'Wiklind_z683_low','Wiklind_z683_high','Wiklind_z954_low','Wiklind_z954_high','Wuyts_z_peak',
                        'Wuyts_z_weight','Wuyts_z683_low','Wuyts_z683_high','Wuyts_z954_low','Wuyts_z954_high'],
                           delimiter=' ', comment='#')
            # match based on ID as GDN has ID weirdness
            df_phot = df_phot.loc[ (df_phot['ID'].isin(df['ID']) == True) ]
            df_phot = df_phot.reset_index(drop=True)
            df['ZPHOT_PEAK'] = df_phot['mFDa4_z_peak'] # might want to use weight for consistency with COSMOS
            df['SIG_DIFF'] = df_phot['mFDa4_z683_high'] - df_phot['mFDa4_z683_low']
            df = df[ (df['CLASS_STAR'] < 0.9) & (df['PHOTFLAG'] == 0) & (df['MASS'] > (mass_lo-1)) ]
            df = df.reset_index(drop=True)
            
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # calculate LX
        df['LX'] = ( df['FX'] * 4 * np.pi * ((cosmo.luminosity_distance(df['ZPHOT_PEAK']).to(u.cm))**2) * 
                                                                    ((1+df['ZPHOT_PEAK'])**(gamma-2)) )

        # Flag IR AGN based on Donley and Stern
        # look at IR luminosities
        df['IR_AGN_DON'] = [0]*len(df)
        df['IR_AGN_STR'] = [0]*len(df)
        if field == 'GDN':
            df = df.rename(columns={'IRAC_CH1_SCANDELS_FLUX':'IRAC_CH1_FLUX', 'IRAC_CH1_SCANDELS_FLUXERR':'IRAC_CH1_FLUXERR',
                                            'IRAC_CH2_SCANDELS_FLUX':'IRAC_CH2_FLUX', 'IRAC_CH2_SCANDELS_FLUXERR':'IRAC_CH2_FLUXERR'})

        df.loc[ (np.log10(df['IRAC_CH3_FLUX']/df['IRAC_CH1_FLUX']) >= 0.08) &
                   (np.log10(df['IRAC_CH4_FLUX']/df['IRAC_CH2_FLUX']) >= 0.15) &
                   (np.log10(df['IRAC_CH4_FLUX']/df['IRAC_CH2_FLUX']) >= (1.21*np.log10(df['IRAC_CH3_FLUX']/df['IRAC_CH1_FLUX']))-0.27) &
                   (np.log10(df['IRAC_CH4_FLUX']/df['IRAC_CH2_FLUX']) <= (1.21*np.log10(df['IRAC_CH3_FLUX']/df['IRAC_CH1_FLUX']))+0.27) &
                   (df['IRAC_CH2_FLUX'] > df['IRAC_CH1_FLUX']) &
                   (df['IRAC_CH3_FLUX'] > df['IRAC_CH2_FLUX']) &
                   (df['IRAC_CH4_FLUX'] > df['IRAC_CH3_FLUX']), 'IR_AGN_DON'] = 1

        # zero magnitude fluxes:
        F03p6 = 280.9 #±4.1 Jy
        F04p5 = 179.7 #±2.6 Jy
        F05p8 = 115.0 #±1.7 Jy
        F08p0 = 64.9 #±0.9 Jy 
        df.loc[ (2.5*np.log10(F05p8 / (df['IRAC_CH3_FLUX']/1e6)) - 2.5*np.log10(F08p0 / (df['IRAC_CH4_FLUX']/1e6)) > 0.6) &
                   (2.5*np.log10(F03p6 / (df['IRAC_CH1_FLUX']/1e6)) - 2.5*np.log10(F04p5 / (df['IRAC_CH2_FLUX']/1e6)) > 
                   0.2 * (2.5*np.log10(F05p8 / (df['IRAC_CH3_FLUX']/1e6)) - 2.5*np.log10(F08p0 / (df['IRAC_CH4_FLUX']/1e6))) + 0.18) &
                   (2.5*np.log10(F03p6 / (df['IRAC_CH1_FLUX']/1e6)) - 2.5*np.log10(F04p5 / (df['IRAC_CH2_FLUX']/1e6)) > 
                    2.5 * (2.5*np.log10(F05p8 / (df['IRAC_CH3_FLUX']/1e6)) - 2.5*np.log10(F08p0 / (df['IRAC_CH4_FLUX']/1e6))) - 3.5),
                   'IR_AGN_STR'] = 1

        # set the ones with incomplete data back to 0: POTENTIALLY UNECESSARY NOW (BELOW)
        df.loc[ (df['IRAC_CH1_FLUX'] <= 0) | (df['IRAC_CH2_FLUX'] <= 0) |
                   (df['IRAC_CH3_FLUX'] <= 0) | (df['IRAC_CH4_FLUX'] <= 0), ['IR_AGN_DON', 'IR_AGN_STR'] ] = 0
        df.loc[ (df['IRAC_CH1_FLUX']/df['IRAC_CH1_FLUXERR'] < 3) | (df['IRAC_CH2_FLUX']/df['IRAC_CH2_FLUXERR'] < 3) |
                   (df['IRAC_CH3_FLUX']/df['IRAC_CH3_FLUXERR'] < 3) | (df['IRAC_CH4_FLUX']/df['IRAC_CH4_FLUXERR'] < 3),
                  ['IR_AGN_DON', 'IR_AGN_STR'] ] = 0
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # count isolated galaxies in each field:

        # all IDs in field
        all_ID = np.array(df.loc[(df['ZPHOT_PEAK'] > 0.3) & (df['ZPHOT_PEAK'] < 6.5), 'ID'])
        # get array of all IDs in a pair:
        
        if iso_method == 'high_prob':
            pair_ID = np.array(all_df.loc[ (all_df['field'] == field) & (all_df['pair_prob'] > iso_min_prob), 'ID' ])
            # only keep the elements that aren't in true pair:
            mask = np.isin(all_ID, pair_ID, invert=True)
            isos = all_ID[mask]
        elif iso_method == 'proj_mask':        
            pair_ID = np.array(all_df.loc[ all_df['field'] == field, 'ID' ])
            # only keep the elements that aren't in true pair:
            mask = np.isin(all_ID, pair_ID, invert=True)
            isos = all_ID[mask]
        else: # 'none'
            isos = all_ID

        iso_ID = np.array(isos)
        iso_z = np.array(df.loc[ (df['ID'].isin(iso_ID) == True), 'ZPHOT_PEAK'])
        iso_mass = np.array(df.loc[ (df['ID'].isin(iso_ID) == True), 'MASS'])
        iso_sig = np.array(df.loc[ (df['ID'].isin(iso_ID) == True), 'SIG_DIFF'])
        iso_LX = np.array(df.loc[ (df['ID'].isin(iso_ID) == True), 'LX'] )
        iso_Don = np.array(df.loc[ (df['ID'].isin(iso_ID) == True), 'IR_AGN_DON'] )
        iso_Str = np.array(df.loc[ (df['ID'].isin(iso_ID) == True), 'IR_AGN_STR'] )

        idf = pd.DataFrame( { 'ID':iso_ID, 'field':[field]*len(iso_ID), 'z':iso_z, 
                            'mass':iso_mass, '2sig':iso_sig, 'LX':iso_LX, 
                            'IR_AGN_DON':iso_Don, 'IR_AGN_STR':iso_Str} )

        iso_df[field] = idf
        
    iso_df = pd.concat([iso_df[key] for key in iso_df])
    iso_df = iso_df.reset_index(drop=True)
    
    return iso_df

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def gengalPDF(prop_df, probs, sensitivity=10): # sensitivity is the number of bins per delta 1 of z, mass, and 2sig
    
    props = np.array(prop_df)
    
    # redefine smaller bin sizes... sticking with 5 bins / range(1) for each
    # input is in order of z, mass, 2sig
    # redshift bins from 0-4
    zbins = np.linspace(0,4,((4-0)*sensitivity)+1)
    # mass from 8-12
    mbins = np.linspace(8,12,((12-8)*sensitivity)+1)
    # 2sig from 0-10
    sbins = np.linspace(0,10,((10-0)*sensitivity)+1)
    # (zbins, mbins, sbins) => SHOULD FIX THIS <= ### ~~~ CURRENT DIMENSION ERROR BELOW ~~~ ###
    
    H, edges = np.histogramdd(props, bins=(zbins, mbins, sbins), weights=probs, density=True)
    edges = np.array(edges, dtype=object)
    
    # create an array for the grid centers
    grid_centers0 = np.zeros(len(edges[0][:-1]))
    grid_centers1 = np.zeros(len(edges[1][:-1]))
    grid_centers2 = np.zeros(len(edges[2][:-1]))
    grid_centers = np.array([grid_centers0, grid_centers1, grid_centers2], dtype=object)
    # grid_centers = np.zeros(edges[:,:-1].shape)
    for i in range(0,len(edges)):
        for j in range(1,len(edges[i])):
            # grid_centers[i,j-1] = (edges[i,j-1] + edges[i,j]) / 2
            grid_centers[i][j-1] = (edges[i][j-1] + edges[i][j]) / 2

    z, m, sig = np.meshgrid(grid_centers[0], grid_centers[1], grid_centers[2], indexing='ij')

    a = np.stack((z,m,sig), axis=3) # yeah this what we want
    # now every coordinate value of a has some sort of proabbility
    # best to flatten these arrays down to the last axis
    # a.shape
    aa = np.reshape(a, (a.shape[0]*a.shape[1]*a.shape[2], 3) ) ####
    aa_idx = list(range(0,len(aa))) # need to do this for each of numpy choice
    Ha = np.reshape(H, (H.shape[0]*H.shape[1]*H.shape[2]) ) ####
    Ha = Ha / np.sum(Ha)
    gal_props = aa
    gal_PDF = Ha
    
    return gal_props, gal_PDF

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def gen_gal(gal_props, gal_PDF, n):
    
    aa_idx = list(range(0,len(gal_props)))
    choice_idx = random.choice(aa_idx, size=n, p=gal_PDF)
    drawn_gal = gal_props[choice_idx]

    drawn_z = drawn_gal[:,0]
    drawn_m = drawn_gal[:,1]
    drawn_s = drawn_gal[:,2]
    
    return drawn_z, drawn_m, drawn_s

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def get_control(iso_df, z_arr, m_arr, sig_arr, dif_lim=0.5):
    
    bad = False
    
    milestones = np.linspace(500, 500000, 1000, dtype=int)
    
    matches = [] # IDs to drop
    
    for i, (z, m, sig) in enumerate(zip(z_arr, m_arr, sig_arr)):
        
        cmatch_df = iso_df

        # create columns for difference between z/mass/2sig control and pair z/mass/2sig  
        cmatch_df['dif'] = (cmatch_df['z'] - z)**2 + (cmatch_df['mass'] - m)**2 + (cmatch_df['2sig'] - sig)**2

        # need to sort dataframe based on two columns THEN continue
        cmatch_df.sort_values(by=['dif'], inplace = True, ascending = True)

        # top one is the match
        match = cmatch_df.iloc[0]
        if match['dif'] > dif_lim:
            print('difference exceeds {0} on run {1}'.format(dif_lim, i))
            return matches
        else:
            matches.append(match)
            iso_df = cmatch_df.drop( cmatch_df[ (cmatch_df['ID'] == match['ID']) ].index ).reset_index(drop=True)
            if np.any(milestones==len(matches)) == True:
                print('{0} matches /// dif = {1} /// {2} left'.format(len(matches), match['dif'], len(cmatch_df)))
                # how to round to 2 sgfgs?
        
    return matches
    

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def ks_weighted(data1, data2, wei1, wei2, alternative='two-sided'):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
    cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
    d = np.max(np.abs(cdf1we - cdf2we))
    # calculate p-value
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = m * n / (m + n)
    if alternative == 'two-sided':
        prob = distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the larger of (n1, n2)
        expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
        prob = np.exp(expt)
    return d, prob

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


if __name__ == '__main__':
    main()