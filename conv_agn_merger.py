# Sean Dougherty
# 6/6/2022
# weighting pairs using the new convolution method

# import libraries
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' -> could change to None

import numpy as np
from numpy import random
np.seterr(divide = 'ignore') #'warn' <- division issues in log10, no problem these nans are replaced later
np.seterr(all="ignore")

import time
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
import math as m
import multiprocessing
from multiprocessing import Pool, freeze_support, RLock, RawArray, Manager
from functools import partial

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from scipy import signal
from scipy.interpolate import interp1d

import collections

import sys
import os, psutil

PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/Pair Project - Updated Data/'
cPATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/'
mPATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/MODEL/Input_data/'
conv_PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/conv_prob/'

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
# determine conservative ang separation correspondong to 150 kpc at z = 0.5
R_kpc = cosmo.arcsec_per_kpc_proper(0.5) # arcsec/kpc at z=0.5
max_R_kpc = 100*u.kpc * R_kpc # in arcseconds ### this is the bug right here

mass_lo = 9.4 # lower mass limit of the more massive galaxy in a pair that we want to consider
gamma = 1.4 # for k correction calculation

# max_sep = 100 # 150 kpc <== should make farther out so as to not get varying sep measurements based on prime/partner z that don't cut them
max_dv = 1000 

sigma_cut = 100 # for individual PDF broadness
zp_cut = 0 # for pairs that will negligently contribute to the final AGN fractions
hmag_cut = 100 # essentially no cut <- not important 
select_controls = False
duplicate_pairs = False
apple_bob = True
save = True
t_run = False
z_type = 'p' # ['p', 'ps' ,'s']
date = '12.13' ### can automate this you know ### COSMOS version of 11.11 didn't work...
num_procs = 25
num_PDFprocs = 20
min_pp = 0.1
pp_cut = 0 ### ~~~ NEW ~~~ ###
ch_size = 10

# initialize global dictionaries for iso_pools and PDF_array to help with multiprocesing
iso_pool_dict = {}
controlled_IDs = {}
PDF_dict = {}
prime_zt_dict, partner_zt_dict = {},{}
chunked_idx_dict = {}
base_dz=0.05
base_dM=0.05
base_dE=2
base_dS=0.05 # and make an incremental log difference...
dP=0.01
N_controls=3 ###
sig_det = 5
# initialize worker dict:
var_dict = {}
bob_type = 'full' # 'full' or 'randbob' <== only works if duplicate pairs is true.
# COSMOS_pt = 'ez' # 'ez', 'lp'
full_SPLASH = False
if bob_type == 'randbob':
    duplicate_pairs = True
    N_controls=1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# main function
def main():
    print('beginning main()')
    
    # we want to parallelize the data by fields, so:
    all_fields = ['GDS','EGS','COS','GDN','UDS','COSMOS'] # COS is for CANDELS COSMOS
    # all_fields = ['GDN']
    # all_fields = ['COSMOS']
    # process_samples('COSMOS')
    
    for field in all_fields:
        process_samples(field)
#     # Create a multiprocessing Pool
#     pool = Pool(num_proc) 
#     pool.map(process_samples, all_fields)
            
#     print('Done!')
    
#     # close pool
#     pool.close()
#     pool.join()
    
    return
    
def process_samples(field):
    print('beginning process_samples() for {}'.format(field))
    
    # load in catalogs: <== specify column dtypes
    if field == 'COSMOS':
        df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/CANDELS_COSMOS_CATS/'+field+'_data3.csv',
                        dtype={'ZSPEC_R':object})
        df = df.loc[ (df['LP_TYPE'] != 1) & (df['LP_TYPE'] != -99) & (df['MASS'] > (mass_lo)) & # (mass_lo-1)
            (df['FLAG_COMBINED'] == 0) & (df['SIG_DIFF'] > 0) & #(df['HSC_i_MAG_AUTO'] < 27) & (df['HSC_i_MAG_AUTO'] > 0)
                    (df['ZPHOT_PEAK'] > 0) & (df['CANDELS_FLAG'] == False) ]
        if full_SPLASH == True:
            df = df.drop(columns={'IRAC_CH1_FLUX', 'IRAC_CH1_FLUXERR', 'IRAC_CH2_FLUX', 'IRAC_CH2_FLUXERR'})
            df = df.rename(columns={'SPLASH_CH1_FLUX':'IRAC_CH1_FLUX', 'SPLASH_CH1_FLUXERR':'IRAC_CH1_FLUXERR',
                                    'SPLASH_CH2_FLUX':'IRAC_CH2_FLUX', 'SPLASH_CH2_FLUXERR':'IRAC_CH2_FLUXERR',
                                    'SPLASH_CH3_FLUX':'IRAC_CH3_FLUX', 'SPLASH_CH3_FLUXERR':'IRAC_CH3_FLUXERR',
                                    'SPLASH_CH4_FLUX':'IRAC_CH4_FLUX', 'SPLASH_CH4_FLUXERR':'IRAC_CH4_FLUXERR'})
        elif full_SPLASH == False:
            df = df.rename(columns={'SPLASH_CH3_FLUX':'IRAC_CH3_FLUX', 'SPLASH_CH3_FLUXERR':'IRAC_CH3_FLUXERR',
                                    'SPLASH_CH4_FLUX':'IRAC_CH4_FLUX', 'SPLASH_CH4_FLUXERR':'IRAC_CH4_FLUXERR'})
            
        # calculate AB mags:
        df['IRAC_CH1_ABMAG'] = F2m(df['IRAC_CH1_FLUX'], 1)
        df['IRAC_CH2_ABMAG'] = F2m(df['IRAC_CH2_FLUX'], 2)
        df['IRAC_CH3_ABMAG'] = F2m(df['IRAC_CH3_FLUX'], 3)
        df['IRAC_CH4_ABMAG'] = F2m(df['IRAC_CH4_FLUX'], 4)
        
        # # try the FARMER EZ set:
        # df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/COSMOS_FARMER_select.csv',
        #                 dtype={'ZSPEC_R':object})
        # if COSMOS_pt == 'ez':
        #     df = df.loc[ (df['VALID_SOURCE'] == True) & (df['ez_MASS'] > mass_lo) & (df['ez_MASS'] < 12) &
        #         (df['FLAG_COMBINED'] == 0) & (df['ez_SIG_DIFF'] > 0) & (df['HSC_i_MAG'] < 26) ]
        #     df = df.rename(columns={'ez_MASS':'MASS','ez_ZPHOT_PEAK':'ZPHOT_PEAK'})
        # elif COSMOS_pt == 'lp':
        #     df = df.loc[ (df['lp_type'] != 1) & (df['lp_type'] != -99) & (df['lp_MASS'] > mass_lo) & (df['lp_MASS'] < 12) &
        #         (df['FLAG_COMBINED'] == 0) & (df['lp_SIG_DIFF'] > 0) & (df['HSC_i_MAG'] < 26) ]
        #     df = df.rename(columns={'lp_MASS':'MASS','lp_ZPHOT_PEAK':'ZPHOT_PEAK'})
            
    else:
        df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/CANDELS_COSMOS_CATS/'+field+'_data3.csv',
                        dtype={'ZSPEC_R':object})
        df = df[ (df['CLASS_STAR'] < 0.9) & (df['PHOTFLAG'] == 0) & (df['MASS'] > mass_lo) & (df['MASS'] < 15) &
            (df['SIG_DIFF'] > 0) & (df['ZPHOT_PEAK'] > 0) ] 
        #& (df['MAG_AUTO_F606W'] < 30) & (df['MAG_AUTO_F606W'] > 0) ]
    df = df.reset_index(drop=True)
        
    # ### ~~~ TEST RUN ~~~ ###
    if t_run == True:
        df = df.iloc[:500]        
    
    # there is no data draw in this method, go straight to getting projected pairs based on prime
    determine_pairs(df, field)
    
    return
    

def determine_pairs(df, field):
    print('beginning determine_pairs() for ', field)
    
    # first thing is change df based on z_type
    df['z'] = df['ZPHOT_PEAK']
    if z_type == 'ps':
        df.loc[ df['ZBEST_TYPE'] == 's', 'z' ] = df['ZSPEC']
        df.loc[ df['ZBEST_TYPE'] == 's', 'SIG_DIFF' ] = 0.01
    
    # make definite redshift cut:
    all_df = df.loc[ (df['z'] > 0.5) & (df['z'] < 3.0) ]  ### ~~~ GOTTA BE CAREFUL WITH EDGES ~~~ ###
    print('all_df length', field, len(all_df))
    all_df = all_df.reset_index(drop=True)
    
    # calculate LX
    # ### ~~~ SHUFFLE FX... ~~~ ###
    # all_df['FX'] = np.random.permutation(all_df['FX'])
    # print('FX shuffles')
    all_df['LX'] = ( all_df['FX'] * 4 * np.pi * ((cosmo.luminosity_distance(all_df['z']).to(u.cm))**2).value * 
                                                                ((1+all_df['z'])**(gamma-2)) )
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # Flag IR AGN based on Donley and Stern
    # look at IR luminosities
    all_df['IR_AGN_DON'] = [0]*len(all_df)
    all_df['IR_AGN_STR'] = [0]*len(all_df)
    
    np.seterr(divide = 'ignore')

    all_df.loc[ (np.log10(all_df['IRAC_CH3_FLUX']/all_df['IRAC_CH1_FLUX']) >= 0.08) &
               (np.log10(all_df['IRAC_CH4_FLUX']/all_df['IRAC_CH2_FLUX']) >= 0.15) &
               (np.log10(all_df['IRAC_CH4_FLUX']/all_df['IRAC_CH2_FLUX']) >= (1.21*np.log10(all_df['IRAC_CH3_FLUX']/all_df['IRAC_CH1_FLUX']))-0.27) &
               (np.log10(all_df['IRAC_CH4_FLUX']/all_df['IRAC_CH2_FLUX']) <= (1.21*np.log10(all_df['IRAC_CH3_FLUX']/all_df['IRAC_CH1_FLUX']))+0.27) &
               (all_df['IRAC_CH2_FLUX'] > all_df['IRAC_CH1_FLUX']) &
               (all_df['IRAC_CH3_FLUX'] > all_df['IRAC_CH2_FLUX']) &
               (all_df['IRAC_CH4_FLUX'] > all_df['IRAC_CH3_FLUX']), 'IR_AGN_DON'] = 1
    
    # zero magnitude fluxes:
    F03p6 = 280.9 #±4.1 Jy
    F04p5 = 179.7 #±2.6 Jy
    F05p8 = 115.0 #±1.7 Jy
    F08p0 = 64.9 #±0.9 Jy 
    all_df.loc[ (2.5*np.log10(F05p8 / (all_df['IRAC_CH3_FLUX']/1e6)) - 2.5*np.log10(F08p0 / (all_df['IRAC_CH4_FLUX']/1e6)) > 0.6) &
               (2.5*np.log10(F03p6 / (all_df['IRAC_CH1_FLUX']/1e6)) - 2.5*np.log10(F04p5 / (all_df['IRAC_CH2_FLUX']/1e6)) > 
               0.2 * (2.5*np.log10(F05p8 / (all_df['IRAC_CH3_FLUX']/1e6)) - 2.5*np.log10(F08p0 / (all_df['IRAC_CH4_FLUX']/1e6))) + 0.18) &
               (2.5*np.log10(F03p6 / (all_df['IRAC_CH1_FLUX']/1e6)) - 2.5*np.log10(F04p5 / (all_df['IRAC_CH2_FLUX']/1e6)) > 
                2.5 * (2.5*np.log10(F05p8 / (all_df['IRAC_CH3_FLUX']/1e6)) - 2.5*np.log10(F08p0 / (all_df['IRAC_CH4_FLUX']/1e6))) - 3.5),
               'IR_AGN_STR'] = 1
    
    # set the ones with incomplete data back to 0: POTENTIALLY UNECESSARY NOW (BELOW)
    all_df.loc[ (all_df['IRAC_CH1_FLUX'] <= 0) | (all_df['IRAC_CH2_FLUX'] <= 0) |
               (all_df['IRAC_CH3_FLUX'] <= 0) | (all_df['IRAC_CH4_FLUX'] <= 0), ['IR_AGN_DON', 'IR_AGN_STR'] ] = 0
    all_df.loc[ (all_df['IRAC_CH1_FLUX']/all_df['IRAC_CH1_FLUXERR'] < sig_det) | (all_df['IRAC_CH2_FLUX']/all_df['IRAC_CH2_FLUXERR'] < sig_det) |
               (all_df['IRAC_CH3_FLUX']/all_df['IRAC_CH3_FLUXERR'] < sig_det) | (all_df['IRAC_CH4_FLUX']/all_df['IRAC_CH4_FLUXERR'] < sig_det),
              ['IR_AGN_DON', 'IR_AGN_STR'] ] = 0
    
    # CHANGE NUMPY ERROR TO TRY TO FIND BUG
    np.seterr(divide = 'raise')
        
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#     print('Applying COSMOS mag cut...')
#     all_df.loc[ (all_df['IRAC_CH1_ABMAG'] > 26) | (all_df['IRAC_CH2_ABMAG'] > 26) |
#            (all_df['IRAC_CH3_ABMAG'] > 21) | (all_df['IRAC_CH4_ABMAG'] > 21),  # try the CANDELS COSMOS limits...
#            ['IR_AGN_DON', 'IR_AGN_STR'] ] = 0
    
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # match catalogs:
    df_pos = SkyCoord(all_df['RA'],all_df['DEC'],unit='deg')
    idxc, idxcatalog, d2d, d3d = df_pos.search_around_sky(df_pos, max_R_kpc)
    # place galaxy pairs into a df and get rid of duplicate pairs:
    matches = {'prime_index':idxc, 'partner_index':idxcatalog, 'arc_sep': d2d.arcsecond}
    match_df = pd.DataFrame(matches)
    
    # all galaxies will match to themselves, so if they match to another they are another potential pair
    pair_df = match_df[ (match_df['arc_sep'] != 0.00) ]
    
    # calculate mass ratio
    pair_df['mass_ratio'] = (np.array(all_df.loc[pair_df['prime_index'], 'MASS']) - 
                             np.array(all_df.loc[pair_df['partner_index'],'MASS']) )
    
    # get isolated galaxy samples
    iso_df = match_df[ (match_df['arc_sep'] == 0.00) ]
    iso_conf_id = np.array(iso_df['prime_index'])
    pair_ear_id = np.array(pair_df['prime_index'])
    mask_conf = np.isin(iso_conf_id, pair_ear_id, invert=True)
    iso_conf = iso_conf_id[mask_conf]
    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    if duplicate_pairs == False:
        # get rid of duplicates in pair sample
        pair_df = pair_df[ (pair_df['mass_ratio'] >= 0) ]
        
        sorted_idx_df = pd.DataFrame(np.sort((pair_df.loc[:,['prime_index','partner_index']]).values, axis=1), 
                                        columns=(pair_df.loc[:,['prime_index','partner_index']]).columns).drop_duplicates()
        
        pair_df = pair_df.reset_index(drop=True)
        pair_df = pair_df.iloc[sorted_idx_df.index]
        # we only want pairs where the mass ratio is within 10
        pair_df = pair_df[ (pair_df['mass_ratio'] <= 1) ] 
        # calculate projected separation at z
        pair_df['kpc_sep'] = (pair_df['arc_sep']) / (cosmo.arcsec_per_kpc_proper(all_df.loc[pair_df['prime_index'], 'z']).value)
        # get complete list of projected pairs -> no need to calculate dv in this method
        # true_pairs = pair_df[ (pair_df['kpc_sep'] <= max_sep*u.kpc) ] # <== I wouldn't actually even make this cut
        true_pairs = pair_df
        # we only want to consider pairs where the prime index is above our threshold <== now what do you do...
        # true_pairs = true_pairs.iloc[ np.where( np.array(all_df.loc[ true_pairs['prime_index'], 'MASS' ] > mass_lo) == True ) ]
        
    elif duplicate_pairs == True:
        # we only want pairs where the mass ratio is within 10
        pair_df = pair_df[ (np.abs(pair_df['mass_ratio']) <= 1) ] 
        pair_df['kpc_sep'] = (pair_df['arc_sep']) / (cosmo.arcsec_per_kpc_proper(all_df.loc[pair_df['prime_index'], 'z']).value)
        # true_pairs = pair_df[ (pair_df['kpc_sep'] <= max_sep*u.kpc) ]
        true_pairs = pair_df
        # in this case, we want to ask if either mass is above mass_lo
        true_pairs = true_pairs.iloc[ np.where( (np.array(all_df.loc[ true_pairs['prime_index'], 'MASS' ] > mass_lo) == True) |
                                    (np.array(all_df.loc[ true_pairs['partner_index'], 'MASS' ] > mass_lo) == True )) ]

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
    
#     # add galaxies that aren't pairs into the isolated sample:
#     iso_add = pair_df[ (pair_df['kpc_sep'] > max_sep) ]

#     # just stack prime and partner indices into massive array:
#     iso_add_idx = np.concatenate( (np.array(iso_add['prime_index']), np.array(iso_add['partner_index'])), axis=0)
#     # return unique indices
#     iso_add_uniq = np.unique(iso_add_idx)
#     # get rid of cases where those indices appear elsewhere, so create array for true pair indices
#     true_pair_idx = np.concatenate( (np.array(true_pairs['prime_index']), np.array(true_pairs['partner_index'])), axis=0)
#     # only keep the elements that aren't in true pair:
#     mask = np.isin(iso_add_uniq, true_pair_idx, invert=True)
#     iso_unq = iso_add_uniq[mask]
#     all_iso = np.concatenate( (iso_conf, iso_unq), axis=0)
    
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print('PDF width cut? ', sigma_cut)
    # print(field, len(all_df), len(np.unique(np.concatenate((true_pairs['prime_index'], true_pairs['partner_index'])))), len(iso_unq))
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                         
    # # calculate pair fraction for each projected pair:
    # pair_probs, pair_PdA = load_pdfs(all_df.loc[ true_pairs['prime_index'], 'ID'], all_df.loc[ true_pairs['partner_index'], 'ID'],
    #                                  all_df.loc[ true_pairs['prime_index'], 'z'], all_df.loc[ true_pairs['partner_index'], 'z'],
    #                                 all_df.loc[ true_pairs['prime_index'], 'ZBEST_TYPE'], 
    #                                  all_df.loc[ true_pairs['partner_index'], 'ZBEST_TYPE'],
    #                                  true_pairs['arc_sep'], field)
    
    # ### ~~~ save input data for the model then end the program ~~~ ###
    # load_pdfs(all_df.loc[ true_pairs['prime_index'], 'ID'], all_df.loc[ true_pairs['partner_index'], 'ID'], 
    #                        true_pairs['arc_sep'], field)
    # model_df = pd.DataFrame( {'Field': field*len(true_pairs),
    #                                   'ID1': np.array(all_df.loc[ true_pairs['prime_index'], 'ID']),
    #                                   'ID2': np.array(all_df.loc[ true_pairs['partner_index'], 'ID']),
    #                                   'M1': np.array(all_df.loc[ true_pairs['prime_index'], 'MASS' ]),
    #                                   'M2': np.array(all_df.loc[ true_pairs['partner_index'], 'MASS' ]),
    #                                   'fAGN1_X': np.array(all_df.loc[ true_pairs['prime_index'], 'LX' ]),
    #                                   'fAGN2_X': np.array(all_df.loc[ true_pairs['partner_index'], 'LX' ]),
    #                                   'fAGN1_IR': np.array(all_df.loc[ true_pairs['prime_index'], 'IR_AGN_DON' ]),
    #                                   'fAGN2_IR': np.array(all_df.loc[ true_pairs['partner_index'], 'IR_AGN_DON' ])} )
    # # do the fine tuning of this in post
    # model_df.to_csv(mPATH+'cat_'+field+'.csv', index=False)
    # print('Data saved! now exiting {}'.format(field))
    # return
    # ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
    
    # print('pair probability calculated in ', field)
    # true_pairs['pair_prob'] = pair_probs
    
    gtrue_pairs = true_pairs.reset_index(drop=True)
    
    # add back control galaxies that are only included in pairs where pair_prob < 0
    # gtrue_pairs = true_pairs[ true_pairs['pair_prob'] >= zp_cut ] ### MAY NEED TO RESET INDEX ###
    # print(gtrue_pairs.index, len(pair_PdA))
    # pair_PdA_gt0 = pair_PdA[gtrue_pairs.index]
    # # save PdA arrays
    # hdu_dA = fits.PrimaryHDU(pair_PdA_gt0)
    # hdul_dA = fits.HDUList([hdu_dA])
    # if save == True:
    #     hdul_dA.writeto(conv_PATH+'PdA_output/PdA_ztype-'+z_type+'_'+field+'_'+date+'.fits', overwrite=True)
    #     print('PdA array saved in {}'.format(field))
    # gtrue_pairs = gtrue_pairs.reset_index(drop=True)
    
    
#     # add galaxies that aren't pairs into the isolated sample:
#     iso_add = true_pairs[ true_pairs['pair_prob'] < zp_cut ]
#     # just stack prime and partner indices into massive array:
#     iso_add_idx = np.concatenate( (np.array(iso_add['prime_index']), np.array(iso_add['partner_index'])), axis=0)
#     # return unique indices
#     iso_add_uniq = np.unique(iso_add_idx)
#     # get rid of cases where those indices appear elsewhere, so create array for true pair indices
#     true_pair_idx = np.concatenate( (np.array(gtrue_pairs['prime_index']), np.array(gtrue_pairs['partner_index'])), axis=0)
#     # only keep the elements that aren't in true pair:
#     mask = np.isin(iso_add_uniq, true_pair_idx, invert=True)
#     iso_unq = iso_add_uniq[mask]
#     all_iso = np.concatenate( (all_iso, iso_unq), axis=0)
    
#     print('{0}: pair_count = {1}, iso count = {2}'.format(field, len(gtrue_pairs), len(all_iso)))
    
    
    # # pick out control galaxies
    # iso_idx = all_iso
    # iso_mass = all_df.loc[all_iso, 'MASS']
    # iso_z = all_df.loc[all_iso, 'z']
    # iso_sig = all_df.loc[all_iso, 'SIG_DIFF']
    # pair_idx = np.concatenate( (gtrue_pairs['prime_index'], gtrue_pairs['partner_index']) )
    # pair_mass = np.concatenate( (all_df.loc[ gtrue_pairs['prime_index'], 'MASS' ], all_df.loc[ gtrue_pairs['partner_index'], 'MASS' ]) )
    # pair_z = np.concatenate( (all_df.loc[ gtrue_pairs['prime_index'], 'z' ], all_df.loc[ gtrue_pairs['partner_index'], 'z' ]) )
    # pair_sig = np.concatenate( (all_df.loc[ gtrue_pairs['prime_index'], 'SIG_DIFF' ], all_df.loc[ gtrue_pairs['partner_index'], 'SIG_DIFF' ]) )
    
    gtrue_pairs['prime_ID'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'ID' ])
    gtrue_pairs['partner_ID'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'ID' ])
    gtrue_pairs['prime_z'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'z' ])       ### ~~~ append redshift type ~~~ ###
    gtrue_pairs['prime_zt'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'ZBEST_TYPE' ])
    gtrue_pairs['partner_z'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'z' ])
    gtrue_pairs['partner_zt'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'ZBEST_TYPE' ])
    gtrue_pairs['prime_M'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'MASS' ])
    gtrue_pairs['partner_M'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'MASS' ])
    gtrue_pairs['prime_SFR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'LOGSFR_MED' ])
    gtrue_pairs['partner_SFR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'LOGSFR_MED' ])
    gtrue_pairs['prime_LX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'LX' ])
    gtrue_pairs['partner_LX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'LX' ])
    gtrue_pairs['prime_PDFsig'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'SIG_DIFF'])
    gtrue_pairs['partner_PDFsig'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'SIG_DIFF'])
    
    gtrue_pairs['prime_CH1_FLUX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH1_FLUX'])
    gtrue_pairs['prime_CH2_FLUX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH2_FLUX'])
    gtrue_pairs['prime_CH3_FLUX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH3_FLUX'])
    gtrue_pairs['prime_CH4_FLUX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH4_FLUX'])
    gtrue_pairs['partner_CH1_FLUX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH1_FLUX'])
    gtrue_pairs['partner_CH2_FLUX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH2_FLUX'])
    gtrue_pairs['partner_CH3_FLUX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH3_FLUX'])
    gtrue_pairs['partner_CH4_FLUX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH4_FLUX'])
    
    gtrue_pairs['prime_CH1_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH1_FLUXERR'])
    gtrue_pairs['prime_CH2_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH2_FLUXERR'])
    gtrue_pairs['prime_CH3_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH3_FLUXERR'])
    gtrue_pairs['prime_CH4_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH4_FLUXERR'])
    gtrue_pairs['partner_CH1_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH1_FLUXERR'])
    gtrue_pairs['partner_CH2_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH2_FLUXERR'])
    gtrue_pairs['partner_CH3_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH3_FLUXERR'])
    gtrue_pairs['partner_CH4_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH4_FLUXERR'])
    
    gtrue_pairs['prime_CH1_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH1_ABMAG'])
    gtrue_pairs['prime_CH2_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH2_ABMAG'])
    gtrue_pairs['prime_CH3_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH3_ABMAG'])
    gtrue_pairs['prime_CH4_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH4_ABMAG'])
    gtrue_pairs['partner_CH1_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH1_ABMAG'])
    gtrue_pairs['partner_CH2_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH2_ABMAG'])
    gtrue_pairs['partner_CH3_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH3_ABMAG'])
    gtrue_pairs['partner_CH4_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH4_ABMAG'])
    
    gtrue_pairs['prime_IR_AGN_DON'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'IR_AGN_DON'])
    gtrue_pairs['prime_IR_AGN_STR'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'IR_AGN_STR'])
    gtrue_pairs['partner_IR_AGN_DON'] = np.array(all_df.loc[gtrue_pairs['partner_index'], 'IR_AGN_DON'])
    gtrue_pairs['partner_IR_AGN_STR'] = np.array(all_df.loc[gtrue_pairs['partner_index'], 'IR_AGN_STR'])
    
    gtrue_pairs['prime_env'] = np.array(all_df.loc[gtrue_pairs['prime_index'], z_type+'_env'])
    gtrue_pairs['partner_env'] = np.array(all_df.loc[gtrue_pairs['partner_index'], z_type+'_env'])
    gtrue_pairs['field'] = [field] * len(gtrue_pairs)
    
    
    #### %%%%%%%%%%%%%%%%%%%%%%%% ####
    #### %%% EDIT %%%%%%%%%%%%%%% ####
    #### %%%%%%%%%%%%%%%%%%%%%%%% ####
    
    # # if indeed the bug is here, check how many cases partner_M > prime_mass and exit
    # print('DRUMROLL PLEASE')
    # print( len(gtrue_pairs.loc[ gtrue_pairs['partner_M'] > gtrue_pairs['prime_M'] ]) )
    # sys.exit()
    
    
    gtrue_pairs = gtrue_pairs.loc[ gtrue_pairs['prime_M'] > 10 ].reset_index(drop=True)
    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    # I say just do the load_pdf logic here
        # calculate pair fraction for each projected pair:
    # pair_probs, pair_PdA = load_pdfs(all_df.loc[ true_pairs['prime_index'], 'ID'], all_df.loc[ true_pairs['partner_index'], 'ID'],
    #                                  all_df.loc[ true_pairs['prime_index'], 'z'], all_df.loc[ true_pairs['partner_index'], 'z'],
    #                                 all_df.loc[ true_pairs['prime_index'], 'ZBEST_TYPE'], 
    #                                  all_df.loc[ true_pairs['partner_index'], 'ZBEST_TYPE'],
    #                                  true_pairs['arc_sep'], field)
    
    pair_probs, pair_PdA = load_pdfs(gtrue_pairs, all_df)
    gtrue_pairs['pair_prob'] = pair_probs
    if bob_type == 'full':
        gtrue_pairs = gtrue_pairs.loc[ gtrue_pairs['pair_prob'] > pp_cut ] # > 0
    elif bob_type == 'randbob':
        gtrue_pairs = gtrue_pairs.loc[ gtrue_pairs['pair_prob'] >= 0 ]
    
    pair_PdA_gt0 = pair_PdA[gtrue_pairs.index]
    # save PdA arrays
    hdu_dA = fits.PrimaryHDU(pair_PdA_gt0)
    hdul_dA = fits.HDUList([hdu_dA])
    if save == True:
        hdul_dA.writeto(conv_PATH+'PdA_output/PdA_Pp-'+str(min_pp)+'_bob-'+bob_type+'_M-'+str(mass_lo)+'_ztype-'+z_type+'_'+field+'_'+date+'.fits', overwrite=True)
        print('PdA array saved in {}'.format(field))
        
    gtrue_pairs = gtrue_pairs.reset_index(drop=True)
    
    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    # need to assign all galaxies in all_df a pair probability for any pair within max_R_kpc:
    # WILL ONLY WORK IF I AM USING THE DUPLICATE PAIR LIST, probably need a for loop:
    print('Determing all pair prob for all galaxies in {}'.format(field))
    all_df['all_pp'] = [0]*len(all_df)
    for i, ID in tqdm(enumerate(all_df['ID']), miniters=100):
        # get parts of true_pairs where ID is prime ID
        if duplicate_pairs == False:
            gal_match_probs = gtrue_pairs.loc[ (gtrue_pairs['prime_ID'] == ID) | (gtrue_pairs['partner_ID'] == ID), 'pair_prob' ]
        elif duplicate_pairs == True:
            gal_match_probs = gtrue_pairs.loc[ (gtrue_pairs['prime_ID'] == ID), 'pair_prob' ]
        if len(gal_match_probs) == 0:
            all_df.loc[ i, 'all_pp'] = 0
        elif len(gal_match_probs) != 0: 
            all_df.loc[ i, 'all_pp'] = 1 - np.prod(1-gal_match_probs)
            
    # print('saving all_df to look at all_pp')
    # all_df.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/test_output/'+field+'_all_df.csv', index=False)
    # sys.exit()
        
    # print('%%%%%%%%%%%%%%%%%%%%%')
    # print(len(all_df))
    # print(len(all_df.loc[ all_df['all_pp'] < 0.1 ]))
    # print(i)
    # save for dask demo:
    # all_df.to_parquet('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/all_df.parquet.gzip', index=False)
    # print('saved to parquet')
#     GDS_PDF_array = PDF_dict['GDS']
#     hdu_PDF = fits.PrimaryHDU(GDS_PDF_array)
#     hdul_PDF = fits.HDUList([hdu_PDF])
#     hdul_PDF.writeto('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/GDS_PDF.fits', overwrite=True)
#     print('PDF array saved in {}'.format(field))
#     sys.exit()
    
    # plt.hist(all_df['all_pp'], bins=1000)
    # plt.xscale('log')
    # plt.show()
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
    if save == True:
        gtrue_pairs.to_csv(conv_PATH+'conv_output/PAIRS_Pp-'+str(min_pp)+'_bob-'+bob_type+'_M-'+str(mass_lo)+'_ztype-'+z_type+'_'+field+'_'+date+'.csv', index=False)
        print('saved in ', field)
    
    # let's experiment with The Bobbing Approach
    # write a function for it
    if apple_bob == True:
        gtrue_pairs['prime_RA'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'RA'])
        gtrue_pairs['prime_DEC'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'DEC']) # <== only relevant for apple bobbing
        conv_apples(gtrue_pairs, all_df)

                    
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
        # okay trying to see what the long times are from, 
        # seems just in GDS we have some processes taking anywhere between 40-500 seconds (up to min)
        # try making chunk sizes smaller and experiment without the spawn bit...
        # also no need to be loading in the PDF arrays constantly, especailly burdonsome in ps mode...
        
    
    return
    
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

# def load_pdfs(gal1, gal2, z1, z2, zt1, zt2, theta, field):
def load_pdfs(pair_df, iso_pool_df): # just load in the same iso_pool_df to make the initiliation of the arrays easier
    field = pair_df['field'].unique()[0]
    print('beginning conv_prob() for ', field)
    
    all_prob = []
    all_PdA = []
    
    print('number of pairs in load_pdfs', field, len(pair_df))
    
    dA = np.linspace(0, 200, num=2001)
    # define array sizes to save distributions as
    PdA_2sav = np.zeros((len(pair_df), len(dA)+2)) # so I can add the IDs and field as a check
    Pzz_2sav = np.zeros((len(pair_df), 1001)) # length of z array
    
    
    # load in the PDFs:
    if field == 'COSMOS':
        with fits.open(cPATH+'COSMOS2020_R1/PZ/COSMOS2020_CLASSIC_R1_v2.0_LEPHARE_PZ.fits') as data:
        # if COSMOS_pt == 'ez':
        #     with fits.open(cPATH+'COSMOS2020_R1/PZ/COSMOS2020_FARMER_R1_v2.0_EAZY_PZ.fits') as data:
        #         # fix big endian buffer error:
        #         COSMOS_PZ_arr = np.array(data[0].data)
        # elif COSMOS_pt == 'lp':
        #     with fits.open(cPATH+'COSMOS2020_R1/PZ/COSMOS2020_FARMER_R1_v2.0_LEPHARE_PZ.fits') as data:
        #         # fix big endian buffer error:
            COSMOS_PZ_arr = np.array(data[0].data)
        COSMOS_PZ_arrf = COSMOS_PZ_arr.byteswap().newbyteorder()
        COSMOS_PZ = pd.DataFrame(COSMOS_PZ_arrf)
        z_01 = COSMOS_PZ.loc[0,1:].to_numpy()
        PDF_array = np.array(COSMOS_PZ) # becomes an array in the column case

    else:
        with fits.open(PATH+'CANDELS_PDFs/'+field+'_mFDa4.fits') as data:
            CANDELS_PZ_arr = np.array(data[0].data)
        CANDELS_PZ_arrf = CANDELS_PZ_arr.byteswap().newbyteorder()
        CANDELS_PZ = pd.DataFrame(CANDELS_PZ_arrf)
        z_01 = CANDELS_PZ.loc[0,1:].to_numpy()
        PDF_array = np.array(CANDELS_PZ)
        
    # also work just adjusting the arrays for spectroscopic redshifts here:
    if z_type != 'p':
        # if we are working with zspecs, we need to interpolate to a finer grid:
        z_fine = np.linspace(0,10,10001).round(3)
        # we wanna do this column wise hmmmmm... recreate an appropriate sized array:
        PDF_array_ps = np.zeros((len(PDF_array),len(z_fine)+1))
        # add the fine redshift as the first row:
        PDF_array_ps[0,1:] = z_fine
        # add the IDs on the left hand side below:
        PDF_array_ps[:,0] = PDF_array[:,0]
        # fill the phot-zs first: need the IDs of zbest_type = 'p'
        fintp1 = interp1d(z_01, PDF_array[np.array(iso_pool_df.loc[ iso_pool_df['ZBEST_TYPE'] == 'p', 'ID' ]),1:], kind='linear')
        # actually do them all at once first...
        PDF_array_ps[ np.array(iso_pool_df.loc[ iso_pool_df['ZBEST_TYPE'] == 'p', 'ID' ]), 1: ] = fintp1(z_fine)
        # now for spec-zs
        # find where in the PDF_array_ps the z-value is our spec-z value and fill then normalize:
        spec_IDs = np.array(iso_pool_df.loc[ iso_pool_df['ZBEST_TYPE'] == 's', 'ID' ])
        spec_zs = np.array(iso_pool_df.loc[ iso_pool_df['ZBEST_TYPE'] == 's', 'z' ]).round(3)
        # get the spec-z's to the right values
        y = spec_zs
        x = PDF_array_ps[0,:]
        xsorted = np.argsort(x)
        ypos = np.searchsorted(x[xsorted], y)
        indices = xsorted[ypos]
        PDF_array_ps[spec_IDs, indices ] = 1
        # now normalize
        PDF_array_ps[spec_IDs,1:] = ( PDF_array_ps[spec_IDs,1:] / 
                                     np.array([np.trapz(PDF_array_ps[spec_IDs,1:], x=z_fine)]*PDF_array_ps[:,1:].shape[1]).T )
        z_01 = z_fine
        PDF_array = PDF_array_ps
        
    # add PDF array to the PDF_dict for use later:
    PDF_dict[field] = PDF_array

#     ### ~~~ PARALELLIZE HERE ~~~ ### ~~~~~~~~~~~~~~~~~~~~~~~~ ### ~~~~~~~~~~~~~~~~~~~~~~~~ ###
#     print('Creating pool for Cp and PdA...')
#     # could create a list of 'regions' of the df to call, and could even shange them to an array like below
#     # create an array of chunk sizes:
#     idxs = np.linspace(0, len(pair_df), len(pair_df)+1, dtype=int)
#     split_lines = np.linspace(0, 1e7, 1001, dtype=int)
#     idx_split = np.split(idxs, split_lines)
#     # will need to loop through idx_splitter to remove from the list empty arrays
#     chunked_idx = []
#     for ob in idx_split:
#         if np.any(ob.shape) != 0:
#             chunked_idx.append(ob) # list of indices array to split into:
#     chunked_idx_dict[field] = chunked_idx
    
#     # we are going to want to split up pair_df the same as in bobbing:
#     # will have to remove prime/partner_zt and put as a separate dict
#     # may try this time just using a dict for pair_df and calling parts of it based on the parallelized chunked_idx...
#     # eh nah I'll just do it the same way anyway, probably easiest on the universal memory that way anyway
#     # make separate dicts for prime and partner zt to add in post:
#     prime_zt_dict[field] = np.array(pair_df['prime_zt'])
#     partner_zt_dict[field] = np.array(pair_df['partner_zt'])
#     pair_df = pair_df.drop(columns={'prime_zt','partner_zt','field'})
#     P_shape = np.array(pair_df).shape
#     # Randomly generate some data
#     P_data = np.array(pair_df)
#     P_cols = np.array(pair_df.columns)
#     P_field = field
#     P = RawArray('d', P_shape[0] * P_shape[1]) # 'd' is for double, perhaps will fail with mixed data types...
#     # Wrap X as an numpy array so we can easily manipulates its data.
#     P_np = np.frombuffer(P).reshape(P_shape)
#     # Copy data to our shared array.
#     np.copyto(P_np, P_data)
    
#     # Start the process p
#     # Start the process pool and do the computation.
#     with Pool(processes=num_PDFprocs, initializer=init_worker, initargs=(P, P_cols, P_shape, P_field)) as pool:
#         Cv_result, PdA_result = pool.map(pdf_pll, range(len(chunked_idx)))
        
#     pool.close()
#     pool.join()
#     prime_zt_dict.clear()
#     partner_zt_dict.clear()
#     chunked_idx_dict.clear()
#     var_dict.clear()
    
#     # combine the result lists ---> will need to check order is okay, but I believe map returns things ordered
#     Cv_prob = np.concatenate(Cv_result)
#     PdA_2sav = np.sum(PdA_result, axis=1)
    
#     ### ~~~~~~~~~~~~~~~~~~~~~~~~ ### ~~~~~~~~~~~~~~~~~~~~~~~~ ### ~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
    
    # and no for loop necessary here: <== also just loaf in the match df instead
    Cv_prob = Convdif(z_01, PDF_array[pair_df['prime_ID'],1:], PDF_array[pair_df['partner_ID'],1:], dv_lim=max_dv)
    # wanna put these all in as arrays:
    PdA, comb_PDF = PdA_prob(PDF_array[pair_df['prime_ID'],1:], PDF_array[pair_df['partner_ID'],1:], 
                             np.array(pair_df['prime_zt']), np.array(pair_df['partner_zt']), 
                             np.array(pair_df['arc_sep']), z_01, dA)

# fill the PdA_2sav array
    PdA_2sav[:,0] = np.array(pair_df['prime_ID'])
    PdA_2sav[:,1] = np.array(pair_df['partner_ID'])
    PdA_2sav[:,2:] = PdA
    
    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
    
    return Cv_prob, PdA_2sav

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def pdf_pll(i): # not used
        
    field = var_dict['field']
    idxs = chunked_idx_dict[field]
    idxs = idxs[i]
    PDF_array = PDF_dict[field]
    z_01 = PDF_array[0,1:]
    zt1_arr = prime_zt_dict[field]
    zt2_arr = partner_zt_dict[field]
    
    print('Getting Pc and PdA for sources {0}-{1}'.format(idxs[0], idxs[-1]))

    # can recover pair_df data from var_dict:
    pair_arr = np.frombuffer(var_dict['P']).reshape(var_dict['Pshape']) # what about the i!
    
    ID1 = np.array(pair_arr[idxs, np.where(var_dict['Pcols'] == 'prime_ID')][0], dtype=int)
    ID2 = np.array(pair_arr[idxs, np.where(var_dict['Pcols'] == 'partner_ID')][0], dtype=int)
    arc_sep = pair_arr[idxs, np.where(var_dict['Pcols'] == 'arc_sep')][0]
    zt1 = zt1_arr[ idxs ]
    zt2 = zt2_arr[ idxs ]
        
    dA = np.linspace(0, 200, num=2001)
    PdA_2sav = np.zeros((len(PDF_array), len(dA)+2))
    
    # get the Cv_prob and PdA for this processors chunk:
    Cv_prob = Convdif(z_01, PDF_array[ID1,1:], PDF_array[ID2,1:], dv_lim=max_dv)
    # wanna put these all in as arrays:
    PdA, comb_PDF = PdA_prob(PDF_array[ID1,1:], PDF_array[ID2,1:], np.array(zt1), np.array(zt2), np.array(arc_sep), z_01, dA)
    
    # now how do I return these...
    # fill part of the PdA_2sav array
    PdA_2sav[idxs,0] = np.array(ID1)
    PdA_2sav[idxs,1] = np.array(ID2)
    PdA_2sav[idxs,2:] = PdA
    # and return Cv_prob simply as an array:
    # the idea for PdA_2sav is to return them all together then just add them, and there should be no overlapping values
          
    print('Pc and PdA for sources {0}-{1} Secure'.format(idxs[0], idxs[-1]))
    
    return Cv_prob, PdA_2sav 

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def dzdv(v):
    c = 2.998e5 # km/s
    dzdv = (1/c) * ((1- (v/c))**(-1.5)) * ((1+ (v/c))**(-0.5))
    return dzdv

def radvel(z):
    c = 2.998e5 # km/s
    v = c * ( ((z+1)**2 - 1) / ((z+1)**2 + 1) )
    return v

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def Convdif(z_all, Pz1, Pz2, dv_lim=1000): ### ~~~ should edit this so if there are two spec-zs just calculate it directly ~~~ ###
        
    # perform a change of variables into velocity space
    v_all = radvel(z_all)
    Pv1 = Pz1 * dzdv(v_all)
    Pv2 = Pz2 * dzdv(v_all)
    
    # interpolate the velocities to get evenly spaced points like in redshift space
    v_new = np.linspace(0,radvel(10),num=10000)
    fintp1 = interp1d(v_all, Pv1, kind='linear')
    fintp2 = interp1d(v_all, Pv2, kind='linear')
    
    # extend the inteprolated array
    all_v_neg = -1*v_new[::-1]
    all_ve = np.concatenate((all_v_neg[:-1], v_new))
    
    # convolve with the symmetrical interpolation values
    if len(fintp2(v_new).shape) == 1:
        v_conv = signal.fftconvolve(fintp2(v_new), fintp1(v_new)[::-1], mode='full')
    else:
        v_conv = signal.fftconvolve(fintp2(v_new), fintp1(v_new)[:,::-1], mode='full', axes=1) ### THE BUG IS HERE ###
    
    # clip out negative values before clipping: <== Check is this works
    v_conv = np.clip(v_conv, 0, None)
    try:
        v_conv = v_conv / np.trapz(v_conv, x=all_ve) # normalize area
    except:
        # error is in the division
        v_conv = v_conv / np.array([np.trapz(v_conv, x=all_ve)]*v_conv.shape[1]).T # normalize area

    # integrate velocity convolution to find probability dv within dv_lim
    rnge = tuple(np.where( (all_ve > -dv_lim) & (all_ve < dv_lim)))
    try:
        prob = np.trapz(v_conv[rnge], x=all_ve[rnge])
    except:
        # print('HERE1', v_conv.shape, all_ve.shape)
        prob = np.trapz(v_conv[:,rnge], x=all_ve[rnge])
    
    return prob

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def PdA_prob(PDF1, PDF2, zt1, zt2, theta, z, dA):
    
    comb_PDF = PDF1*PDF2
    # get a middle_z for all rows:
    mid_z_arr = np.vstack( (z[np.argmax(PDF1, axis=1)], z[np.argmax(PDF2, axis=1)]) ).T
    middle_z = np.mean( mid_z_arr, axis=1 )
    
    # go into comb_PDF where zt1 and zt2 are both 's'
    if z_type != 'p':
        s2_idx = np.where((zt1=='s') & (zt2=='s'))[0]
        y = middle_z[s2_idx]
        x = z
        xsorted = np.argsort(x)
        ypos = np.searchsorted(x[xsorted], y)
        indices = xsorted[ypos]
        comb_PDF[ s2_idx, indices] = 1
        
    # normalize first
    comb_PDF = (comb_PDF / np.array([np.trapz(comb_PDF, x=z)]*len(z)).T) # and normalize ### EXPECT ERROR HERE ###
    comb_PDF = np.nan_to_num(comb_PDF) # if they don't ever overlap we get badness
    
    # so split 0-1.61 (1) and 1.62-10 (2)
    dA1 = ang_diam_dist( z[np.where(z <= 1.61)], theta )
    dA2 = ang_diam_dist( z[np.where(z > 1.61)], theta )
    z1 = z[np.where(z <= 1.61)]
    z2 = z[np.where(z > 1.61)]
    comb_PDF1 = comb_PDF[:,np.where(z <= 1.61)[0]] # some np where weirdness
    comb_PDF2 = comb_PDF[:,np.where(z > 1.61)[0]] 

    PdA11 = comb_PDF1 * np.abs(dzdA(dA1, z1, theta)) # bug here
    PdA11 = np.nan_to_num(PdA11) # fill the nan casued by division ny zero
    PdA12 = comb_PDF2 * np.abs(dzdA(dA2, z2, theta))
    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    # concatenate 1 and 2 just to find the max
    dA_comb = np.concatenate((dA1, dA2), axis=1)
    PdA_comb = np.concatenate((PdA11, PdA12), axis=1)
    max_dA = dA_comb[np.arange(len(dA_comb)),np.argmax(PdA_comb, axis=1)] # this was wrong bc every dA is gonna be different...
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    dA_new = dA
    ## Oh this may be tricky bc each one is gonna have a different dA1 due to different thetas...
    ## may need to look into a numpy interpolate OR loop here and fill (gross) ##
    
    # need to get a far as intr_PdA1 and 2 in the loop, so fill a numpy array-style:
    print('Running interpolation loop, please wait.')
    intr_PdA1 = np.zeros((len(dA1),len(dA_new)))
    intr_PdA2 = np.zeros((len(dA2),len(dA_new)))
    for i in tqdm(range(0,len(dA1)), miniters=100):
        fintp1 = interp1d(dA1[i,:], PdA11[i,:], kind='linear', bounds_error=False, fill_value=0)
        fintp2 = interp1d(dA2[i,:], PdA12[i,:], kind='linear', bounds_error=False, fill_value=0)
        intr_PdA1[i,:] = fintp1(dA_new)
        intr_PdA2[i,:] = fintp2(dA_new)

    PdA = intr_PdA1+intr_PdA2
    
    # okay, need to do a similar insertion
    # find value in dA_new that is closest to max_dA in every row:
    # gotta stretch out to get proper shapes:
    dA_new2 = np.array([dA_new]*len(max_dA))
    max_dA2 = np.array([max_dA]*len(dA_new)).T
    if z_type != 'p':
        s3_idx = np.where((zt1=='s') | (zt2=='s'))[0]
        max_dA_idxs = np.argmin(np.abs(dA_new2[s3_idx,:]-max_dA2[s3_idx,:]), axis=1)
        PdA[s3_idx, max_dA_idxs] = np.max(PdA_comb[s3_idx,:], axis=1)
        
    PdA = PdA / np.array([np.trapz(PdA, x=dA_new)]*PdA.shape[1]).T
    
    return np.nan_to_num(PdA), comb_PDF


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def F2m(F, ch): # goes to Vega then to AB
    if ch == 1:
        F0 = 280.9
        K = 2.788
    elif ch == 2:
        F0 = 179.7
        K = 3.255
    elif ch == 3:
        F0 = 115
        K = 3.743
    elif ch == 4:
        F0 = 64.9
        K = 4.372
    return 2.5*np.log10(F0/(F*1e-6)) + K

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


# def angular diameter function:
def ang_diam_dist(z, theta, H0=70, Om_m=0.3, Om_rel=0, Om_lam=0.7, Om_0=1):
    
    c = 2.998e5 # km/s
    try:
        zs = np.linspace(0,z,10, endpoint=True, axis=1) # numerically integrate
    except:
        zs = np.linspace(0,z,10, endpoint=True)
    dA = ( c / (H0*(1+z)) ) * np.trapz( ( 1 / np.sqrt( Om_m*(1+zs)**3 + Om_rel*(1+zs)**4 + Om_lam + (1-Om_0)*(1+zs)**2 ) ), x=zs )
    
    try: # add another flag that will lead to the correct logic with an array
        return dA * theta * 1000 / ((180/np.pi)*3600)
    except:
        exp_dA = np.array([dA]*len(theta))
        exp_theta = np.array([theta]*exp_dA.shape[1]).T
        return exp_dA * exp_theta * 1000 / ((180/np.pi)*3600) # CORRECT

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def dzdA(A, z, theta, H0=70, Om_m=0.3, Om_rel=0, Om_lam=0.7, Om_0=1):
    
    c = 2.998e5 # km/s
    
    try:
        zs = np.linspace(0,z,10, endpoint=True, axis=1) # numerically integrate
    except:
        zs = np.linspace(0,z,10, endpoint=True)
    
    dzdA = - (c * np.trapz( ( 1 / np.sqrt( Om_m*(1+zs)**3 + Om_rel*(1+zs)**4 + Om_lam + (1-Om_0)*(1+zs)**2 ) ), x=zs )) / (H0*A**2)
    
    # dAdz = ( ( c/(H0*(1+z))) * ( 1 / np.sqrt( Om_m*(1+z)**3 + Om_rel*(1+z)**4 + Om_lam + (1-Om_0)*(1+z)**2 ) ) +
    #         (np.trapz( ( 1 / np.sqrt( Om_m*(1+zs)**3 + Om_rel*(1+zs)**4 + Om_lam + (1-Om_0)*(1+zs)**2 ) ), x=zs ) * (-c/H0)*((1+z)**-2) ))
    try:
        dzdA = dzdA * theta * 1000 / ((180/np.pi)*3600) # convert to kpc/"
    except:
        exp_theta = np.array([theta]*dzdA.shape[1]).T
        dzdA = dzdA * exp_theta * 1000 / ((180/np.pi)*3600) # convert to kpc/"
    return dzdA

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def conv_apples(pair_df, iso_pool_df):
    # so for each pair, assemble a df from the iso pool for both it's prime and partner properties
    
    iso_pool_df = iso_pool_df.loc[ iso_pool_df['all_pp'] <= min_pp ].reset_index(drop=True)
    
    field = pair_df['field'].unique()[0]    ### DOESN'T WORK FOR SPEC Z'S BC I NEED TO INTERPOLATE REMEMBER ###
    if field == 'COSMOS':
        base_dz = 0.02
        base_dM = 0.02
    print('Bobbing for apples in {}'.format(field))
    print('Length of pair_df = {}'.format(len(pair_df)))
    
    # split COSMOS into quadrants... hopefully this will ease the convolution calculations:
    iso_pool_df['Quadrant'] = [0]*len(iso_pool_df)
    pair_df['Quadrant'] = [0]*len(pair_df)
        
    # load in the PDFs:
    
    if field == 'COSMOS':
        print('Breaking up into quadrants...')
        percs = np.linspace(0,100,4) # will split COSMOS up into 16 pieces  ##### think of when defaulting to CANDELS #####
        RA_perc = np.percentile(iso_pool_df['RA'], percs)
        DEC_perc = np.percentile(iso_pool_df['DEC'], percs)
        for i in range(len(RA_perc)-1):
            for j in range(len(DEC_perc)-1):
                iso_pool_df.loc[ (iso_pool_df['RA'] >= RA_perc[i]) & (iso_pool_df['RA'] <= RA_perc[i+1]) & 
                       (iso_pool_df['DEC'] >= DEC_perc[j]) & (iso_pool_df['DEC'] <= DEC_perc[j+1]), 'Quadrant'] =  int(str(i)+str(j))
                pair_df.loc[ (pair_df['prime_RA'] >= RA_perc[i]) & (pair_df['prime_RA'] <= RA_perc[i+1]) & 
                       (pair_df['prime_DEC'] >= DEC_perc[j]) & (pair_df['prime_DEC'] <= DEC_perc[j+1]), 'Quadrant'] =  int(str(i)+str(j))
        
    # throw the iso_pool df onto the universal dictionary for mp use:
    iso_pool_dict[field] = iso_pool_df
    controlled_IDs[field] = np.zeros(len(pair_df))
    
    # begin the pooling
    print('Filling apple pool in {}'.format(field)) 
    start = time.perf_counter()
    # prepare the pair_df for pooling <== consider changing ztypes from strings to ints...
    # but remove them for now:
    pair_df = pair_df.drop(columns={'prime_zt','partner_zt','field'})
    P_shape = np.array(pair_df).shape
    # Randomly generate some data
    P_data = np.array(pair_df)
    P_cols = np.array(pair_df.columns)
    P_field = field
    P = RawArray('d', P_shape[0] * P_shape[1]) # 'd' is for double, perhaps will fail with mixed data types...
    # Wrap X as an numpy array so we can easily manipulates its data.
    P_np = np.frombuffer(P).reshape(P_shape)
    # Copy data to our shared array.
    np.copyto(P_np, P_data)
    # Start the process p
    
    # bug is in this pair:
    # 242030 - 238673... memory error
    # when you get to dM > 0.5 and DON'T have enough even in apple df, you then ask the code to search
    # in a massive parameter space, to the point where to convolve 1 million pairs, for example, is not
    # possible for memory...
    # nothing found for 673025-687360
    
    # # STEPS: run this case alone and see how large that merger would be...
    # # SOLUTION: when we reach the 'oops' warning, reset mass and z back to something much lower...
    # var_dict['field'] = P_field
    # var_dict['P'] = P
    # var_dict['Pshape'] = P_shape
    # var_dict['Pcols'] = P_cols
    # # get the index of that pair:
    # bad_idx = pair_df.index[ (pair_df['prime_ID'] == 673025) & (pair_df['partner_ID'] == 687360) ].to_list()[0]
    # print('bad idx:', bad_idx)
    # test = bobbing_pll(bad_idx)
    # print('SUCCESSFUL TEST PASS......')
    # ...
    # ...
    # Still a memory error... solution is just to change the Cp matching function to also adjust when it hits the reset
    # but hopefully the cut_pp > 0.01 will work...............

    ### ~~~ just try runing COSMOS in serial... ~~~ ###
    if bob_type == 'full':
        # Start the process pool and do the computation.
        # add a manager for checklist:
        # manager = multiprocessing.Manager()
        # check_list_ug = manager.list(np.zeros(len(pair_df)))
        
        with Pool(processes=num_procs, initializer=init_worker, initargs=(P, P_cols, P_shape, P_field)) as pool:
        # with Pool(processes=num_procs, initargs=(var_dict, check_list)) as pool:
            # add a manager for checklist:
            result = pool.map(bobbing_pll, range(P_shape[0]))
            # result.get() # see if this catches an error -> map has get built in...
    elif bob_type == 'randbob':
        # Start the process pool and do the computation.
        with Pool(processes=num_procs, initializer=init_worker, initargs=(P, P_cols, P_shape, P_field)) as pool:
            result = pool.map(randbob_pll, np.array(pair_df['prime_ID'].unique()))
            
    # join pool and clear all universal dictionaries used per field:
    pool.close()
    pool.join()
    ### ~~~ just try runing COSMOS in serial... ~~~ ###

    # for serialization
    # result = []
    # for i in tqdm(range(0,len(range(P_shape[0])))):
    #     result.append(bobbing_pll(i))
    # ... If this takes ages, just cut all pairs with Pp < 0.01 - well justified enough...

    iso_pool_dict.clear()
    PDF_dict.clear()
    var_dict.clear()
    # controlled_IDs.clear()
    
    print('CHECK HERE DONE')
    
    # sent the list of dfs to function to better combine them with pickle functions below
    control_df = byhand(result)
    # add field and unique pair_str in post?
    control_df['field'] = [field]*len(control_df)
    control_df['pair_ID'] = control_df['field']+'_'+(control_df['ID1'].astype(int)).astype(str)+'+'+(control_df['ID2'].astype(int)).astype(str)
    
    # combine result df:
    # control_df = pd.concat(result, ignore_index=True)
    
    print('Draining pool.')
    
    print('FINAL TIME', time.perf_counter() - start)
    
#     ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
#     ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
#     ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
#     print('Filling apple pool in {}'.format(field))   
    
#     start = time.perf_counter()
    
#     chunksize = ch_size
    
#     all_controls = []
#     sort_idx = []
#     pbar = tqdm(total=len(pair_df))
    
#     # ctx = multiprocessing.get_context('spawn') # for helping with memory... spawn vs fork allows the spawned procs to not have
#     # pool = ctx.Pool(num_proc)                    # access to their parent's entire memorey
    
#     pool = Pool(num_proc)#, maxtasksperchild=1) ## ^ could be the cause of the COSMOS broken pipes
#     # try different configurations to see what's fastest
#     # all_controls = pool.imap(partial(bobbing_pll, #iso_pool_df=iso_pool_df, 
#     #                                 base_dz=base_dz, base_dM=base_dM, dP=dP, N_controls=N_controls), 
#     #                         df_chunking(pair_df, chunksize))
#     # print('PID done', os.getpid())
    
#     ### potential issues with splitting the df, the for loop, child processses types, maxtasksperchild
#     for _ in pool.imap_unordered(partial(bobbing_pll, #iso_pool_df=iso_pool_df, 
#                                     base_dz=base_dz, base_dM=base_dM, dP=dP, N_controls=N_controls), np.array(pair_df),10):
#                             # df_chunking(pair_df, chunksize)):
#         pbar.update(int(len(_)/2))
#         sort_idx.append(_.loc[0, 'P_ID1'])
#         all_controls.append(_)
#         print('cache length:', len(pool._cache))
#         process = psutil.Process(os.getpid())
#         pass
        
#     # close pool
#     pool.close()
#     pool.join()
#     # print(collections.Counter(all_controls))
    
#     ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
#     ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
#     ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
#     print('Draining pool.')
    
#     print('FINAL TIME', time.perf_counter() - start)
#     print('Chunksize = {}'.format(chunksize))
#     # seems that it takes some time to throw iso_pool_df and PDF_arr on separate processors
#     ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
#     ### ~~~ see that things get put in the right place ~~~ ### it's not but we can sort by P_ID1 in each
#     order = np.argsort(sort_idx)
#     control_order = [all_controls[i] for i in order]
#     control_df = pd.concat(control_order, ignore_index=True)
    
    # and save:
    if save == True:
        # save the field df as a csv
        control_df.to_csv(conv_PATH+'control_output/APPLES_Pp-'+str(min_pp)+'_bob-'+bob_type+'_M-'+str(mass_lo)+'_N-'+str(N_controls)+'_ztype-'+z_type+'_'+field+'_'+date+'.csv', index=False)
        print('Control df saved in {}'.format(field))
    
    return
    
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# build worker initilization function:
def init_worker(P, P_cols, P_shape, P_field):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['field'] = P_field
    var_dict['P'] = P
    var_dict['Pshape'] = P_shape
    var_dict['Pcols'] = P_cols
    
    # global check_list
    # check_list = check_list_ug
    
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def byhand(dfs):
    mtot=0    # save this in storage  /nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/conv_prob/storage
    with open(conv_PATH+'storage/df_all.bin','wb') as f:
        for df in dfs:
            m,n =df.shape
            mtot += m
            f.write(df.values.tobytes())
            typ=df.values.dtype                
    #del dfs
    with open(conv_PATH+'storage/df_all.bin','rb') as f:
        buffer=f.read()
        data=np.frombuffer(buffer,dtype=typ).reshape(mtot,n)
        df_all=pd.DataFrame(data=data,columns=dfs[0].columns) 
    os.remove(conv_PATH+'storage/df_all.bin')
    return df_all

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def PP_diff_func(x):
    # input the value of Pp...
    # I have already tuned what this function should be...
    A = 5.24002529e-05
    C = 1.54419221e-04
    
    return C*(1/(x+A)) + 0.05

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


def bobbing_pll(i): # seems that a log10 warning is raised when no isos are found at all
    
    # CHANGE NUMPY ERROR TO TRY TO FIND BUG # just in case...
    np.seterr(divide = 'raise')
    
    # print('IN', i)
    # st = time.perf_counter()
    
    # recover some variables:
    field = var_dict['field']
    
    if field == 'COSMOS':
        dM_lim = 0.5
    else:
        dM_lim = 0.5
    iso_pool_df = iso_pool_dict[field]
    PDF_array = PDF_dict[field]
    z_01 = PDF_array[0,1:]
    # milestones = np.linspace(100, 100000, 1000, dtype=int)
    
    # can recover pair_df data from var_dict:
    pair_arr = np.frombuffer(var_dict['P']).reshape(var_dict['Pshape']) # what about the i!

    # controls = [] <== should just return one df, not alist at a time, so ignore for now
    # best to get out of the for loop for now, though it is recoverable in github
    # pair_df = pd.DataFrame(pair_arr, columns=var_dict['Pcols'])
    # ID1 = np.array(pair_df.loc[i, 'prime_ID'])
    # z1 = np.array(pair_df.loc[i, 'prime_z'])
    # M1 = np.array(pair_df.loc[i, 'prime_M'])
    # ID2 = np.array(pair_df.loc[i, 'partner_ID'])
    # z2 = np.array(pair_df.loc[i, 'partner_z'])
    # M2 = np.array(pair_df.loc[i, 'partner_M'])
    # Pp = np.array(pair_df.loc[i, 'pair_prob'])
    # Qd = np.array(pair_df.loc[i, 'Quadrant'])
    
    ID1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_ID')][0][0]
    z1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_z')][0][0]
    M1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_M')][0][0]
    SIG1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_PDFsig')][0][0]
    E1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_env')][0][0]
    ID2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_ID')][0][0]
    z2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_z')][0][0]
    M2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_M')][0][0]
    SIG2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_PDFsig')][0][0]
    E2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_env')][0][0]
    Pp = pair_arr[i, np.where(var_dict['Pcols'] == 'pair_prob')][0][0]
    Qd = pair_arr[i, np.where(var_dict['Pcols'] == 'Quadrant')][0][0]
    all_Qds = np.unique(pair_arr[:, np.where(var_dict['Pcols'] == 'Quadrant')][0])
    
    if Pp == 0:
        print('PP = 0', ID1, ID2)
                       
    got = False
    good_old = False
    report = False
    catch_flag = False
    dz = base_dz
    dM = base_dM
    dE = base_dE
    dS = base_dS
    tried_arr = []
    tries = 0
    # start = time.perf_counter()
    
    ### %%%%%%%%%% try to get this error message %%%%%%%%%%%% ###

    while got == False:
        tries+=1
        iso1 = iso_pool_df.loc[ (np.abs(iso_pool_df['z']-z1) < dz) & (np.abs(iso_pool_df['MASS']-M1) < dM) &
                              (iso_pool_df['ID'] != ID1) & (iso_pool_df['ID'] != ID2) & (iso_pool_df['Quadrant'] == Qd) &
                               (np.abs(iso_pool_df[z_type+'_env']-E1) < dE) & 
                               (np.abs(np.log10(iso_pool_df['SIG_DIFF']) - np.log10(SIG1)) < dS), 
                              ['ID','RA','DEC','MASS','z', 'SIG_DIFF','LX','IR_AGN_DON', z_type+'_env'] ] 
        iso1 = iso1.rename(columns={'RA':'RA1', 'DEC':'DEC1', 'z':'z1', 'SIG_DIFF':'SIG1', 'MASS':'MASS1',
                                    'LX':'LX1', 'IR_AGN_DON':'IR_AGN_DON1', z_type+'_env':'ENV1'})
        iso2 = iso_pool_df.loc[ (np.abs(iso_pool_df['z']-z2) < dz) & (np.abs(iso_pool_df['MASS']-M2) < dM) &
                              (iso_pool_df['ID'] != ID1) & (iso_pool_df['ID'] != ID2) & (iso_pool_df['Quadrant'] == Qd) & 
                               (np.abs(iso_pool_df[z_type+'_env']-E2) < dE) & 
                               (np.abs(np.log10(iso_pool_df['SIG_DIFF']) - np.log10(SIG2)) < dS),
                              ['ID','RA','DEC','MASS','z', 'SIG_DIFF','LX','IR_AGN_DON', z_type+'_env'] ]
        iso2 = iso2.rename(columns={'RA':'RA2', 'DEC':'DEC2', 'z':'z2', 'SIG_DIFF':'SIG2', 'MASS':'MASS2',
                                    'LX':'LX2', 'IR_AGN_DON':'IR_AGN_DON2', z_type+'_env':'ENV2'})

        # print(iso1.loc[:,['MASS1', 'z1', 'SIG_DIFF','ENV1']])
        # print(M1, z1, SIG1, E1)
        # sys.exit()
        # somehow combine the iso1 and iso2 df into a M x N long df that has all possible combinations of them
        # useful to calculate conv_prob columns-wise
        apple_df = iso1.merge(iso2, how='cross').rename(columns={'ID_x':'ID1','ID_y':'ID2'})
        # if len(apple_df) == 0:
        #     dz = dz + 0.03
        #     dM = dM + 0.03
        #     continue

        if catch_flag == True:
            print('CAUGHT', len(apple_df))
        # create unique pair strings:
        apple_df['pair_str'] = (apple_df['ID1'].astype(int)).astype(str)+'+'+(apple_df['ID2'].astype(int)).astype(str)
        # AND a flag on if it passes the tests below...
        apple_df['reuse_flag'] = [0]*len(apple_df)
        # make a cut based on apple_df str
        apple_df = apple_df.loc[ (apple_df['pair_str'].isin(tried_arr) == False) ].reset_index(drop=True)
        # safeguard when tried_arr takes out everything: <== just may also need the one above...
        if len(apple_df) == 0:
            dz = dz + 0.03
            dM = dM + 0.03
            dE = dE + 1
            dS = dS + 0.05
            continue

        # make sure a separation cut is adequite:
        xCOR = SkyCoord(apple_df['RA1'], apple_df['DEC1'], unit='deg')
        yCOR = SkyCoord(apple_df['RA2'], apple_df['DEC2'], unit='deg')
        apple_df['arc_sep'] = xCOR.separation(yCOR).arcsecond

        # if we are working with spec-z's, we gotta interpolate these...
        # count the length of Cp and if its more than 50,000, chunk into 10,000 bits
        merge_length = len(apple_df)
        if merge_length < 50000:
            apple_df['Cp'] = Convdif(z_01, PDF_array[apple_df['ID1'],1:], PDF_array[apple_df['ID2'],1:], dv_lim=max_dv)
        else:
            # loop through and calculate Cp in chunks
            print('Chunking up Cp calculation for pair {0}-{1} into {2} chunks'.format( ID1, ID2, (merge_length//10000)+1 ))
            # initialize an empty array to put Cps into
            Cp_chunks = np.zeros( merge_length )
            Ch_chunks_check = np.zeros( merge_length )
            N_chunks = merge_length // 10000
            for j in range(N_chunks+1):
                Cp_ch = Convdif(z_01, PDF_array[apple_df.loc[ j*10000:(j+1)*10000 , 'ID1'],1:], 
                                PDF_array[apple_df.loc[ j*10000:(j+1)*10000, 'ID2'],1:], dv_lim=max_dv)
                Cp_chunks[j*10000:((j+1)*10000)+1] = Cp_ch[:,0] #shape = [10001,1]
                ### BEWARE LOC INDEX ISSUES AT END OF ARRAY ###
                Ch_chunks_check[j*10000:((j+1)*10000)+1] = 1
            # quick check everyone was got:
            if np.any(Ch_chunks_check) == 0:
                print('CHUNK MISSED SOME')
            # now add the chunked Cp values into the df
            apple_df['Cp'] = Cp_chunks
            # final step is to pray

        ### --- WHAT IS GOING ON WITH LOG --- ### 
        # I think it's an issue in Convdif?
        # check if apple_df returns a NaN value?
        if np.any(np.isinf(apple_df['Cp'])) == True:
            print('Convdif produced in infinite for pair {0}, {1}'.format([ID1, ID2]))
            print(apple_df)
            report=True
        if np.any(np.isnan(apple_df['Cp'])) == True:
            print('Convdif produced in infinite for pair {0}, {1}'.format([ID1, ID2]))
            print(apple_df)
            report=True

        apple_df = apple_df.loc[ apple_df['Cp'] > 0 ]

        apple_df2 = apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < PP_diff_func(Pp)) &
                                                 (apple_df['arc_sep'] > max_R_kpc) & 
                                                 (apple_df['ID1'] != apple_df['ID2']) ].reset_index(drop=True)

        apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < PP_diff_func(Pp)) & 
                                                 (apple_df['arc_sep'] > max_R_kpc) &
                                                 (apple_df['ID1'] != apple_df['ID2']), 'reuse_flag' ] = 1

#             if np.log10(Pp) >= -2: ### ~~~ WILL HAVE TO MESS AROUND WITH THESE PARAMETERS ~~~ ###
#                 apple_df2 = apple_df.loc[ ((np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 0.05) |
#                                            (np.abs(apple_df['Cp'] - Pp) < 0.01)) & (apple_df['arc_sep'] > max_R_kpc) &
#                                          (apple_df['ID1'] != apple_df['ID2']) ].reset_index(drop=True)
#                 # add a second tier here to loosen who gets chosen, that way after a certain amount of attempts we just choose the best
#                 apple_df.loc[  ((np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 0.5) | ## * ##
#                                            (np.abs(apple_df['Cp'] - Pp) < 0.01)) & (apple_df['arc_sep'] > max_R_kpc) &
#                                          (apple_df['ID1'] != apple_df['ID2']), 'reuse_flag' ] = 1
#             # add another tier for extremely small probabilities
#             elif np.log10(Pp) < -8:
#                 apple_df2 = apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 4) & 
#                                          (apple_df['arc_sep'] > max_R_kpc) &
#                                          (apple_df['ID1'] != apple_df['ID2']) ].reset_index(drop=True)
#                 apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 4) & (apple_df['arc_sep'] > max_R_kpc) &
#                                          (apple_df['ID1'] != apple_df['ID2']), 'reuse_flag' ] = 1
#             elif np.log10(Pp) < -2 and np.log10(Pp) > -3:
#                 apple_df2 = apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 0.5) & 
#                                          (apple_df['arc_sep'] > max_R_kpc) &
#                                          (apple_df['ID1'] != apple_df['ID2']) ].reset_index(drop=True)
#                 apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 0.5) & (apple_df['arc_sep'] > max_R_kpc) &
#                                          (apple_df['ID1'] != apple_df['ID2']), 'reuse_flag' ] = 1
#             else:                                                                          # maybe too strict...
#                 apple_df2 = apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 1) & 
#                                          (apple_df['arc_sep'] > max_R_kpc) &
#                                          (apple_df['ID1'] != apple_df['ID2']) ].reset_index(drop=True)
#                 apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 1) & (apple_df['arc_sep'] > max_R_kpc) &
#                                          (apple_df['ID1'] != apple_df['ID2']), 'reuse_flag' ] = 1

        # add pair information:
        apple_df2['P_ID1'] = np.array([ID1]*len(apple_df2), dtype=int)
        apple_df2['P_ID2'] = np.array([ID2]*len(apple_df2), dtype=int)
        apple_df2['Pp'] = np.array([Pp]*len(apple_df2), dtype=float)
                                                # Cp could theoretically be 0... causing log10 of 0
        # organize to get best fit:  ### may need to think how the Cp dif is made (log or not), doubt it'd matter tho ###
        apple_df2['dif'] = (10*(np.log10(apple_df2['Cp']) - np.log10(Pp))**2 + 
                                (apple_df2['z1'] - z1)**2 + (apple_df2['MASS1'] - M1)**2 + 0.1*(apple_df2['ENV1'] - E1)**2 +
                            (np.log10(apple_df2['SIG1']) - np.log10(SIG1))**2 +
                            (apple_df2['z2'] - z2)**2 + (apple_df2['MASS2'] - M2)**2 + 0.1*(apple_df2['ENV2'] - E2)**2 +
                            (np.log10(apple_df2['SIG2']) - np.log10(SIG2))**2)
    
        # # now sort on this
        # # add good values to this if they exist:
        # if good_old == True:
        #     apple_df2 = good_old_df
        #     apple_df2 = pd.concat(apple_df2, good_old_df, ignore_index=True) 

        apple_df2.sort_values(by=['dif'], inplace=True, ascending=True, ignore_index=True) # this resets the index

        # if we have repeat ID1s or ID2s for examples, we want to ignore these # essentially just like chooseing the same 
        apple_df2 = apple_df2.drop_duplicates(['ID1'])
        apple_df2 = apple_df2.drop_duplicates(['ID2'])
        apple_df2 = apple_df2.reset_index(drop=True)
        # add a simple integer count
        # apple_df2['NUM'] = np.linspace(1, len(apple_df2), len(apple_df2), dtype=int)

        # finally, if we have inverse pairs we need to get rid of these...
        # apple_df3 = pd.DataFrame(np.sort((apple_df2.loc[:,['ID1','ID2']]).values, axis=1), 
        #                             columns=(apple_df2.loc[:,['ID1','ID2']]).columns).drop_duplicates()
        
        sort_idx = np.argsort((apple_df2.loc[:,['ID1','ID2']]).values, axis=1)
        arr = np.array(apple_df2.loc[:, ['ID1','ID2']].values)
        arr_ysort = np.take_along_axis(arr, sort_idx, axis=1)
        # unique at axis 0 to get rid of duplicates
        uniq_arr, uniq_idx = np.unique(arr_ysort, return_index=True, return_inverse=False, axis=0)
        # return the unique pairing
        uniq_p = arr[np.sort(uniq_idx)]
        apple_df3 = pd.DataFrame( uniq_p, columns=(apple_df2.loc[:,['ID1','ID2']]).columns )
        apple_df3['pair_str'] = (apple_df3['ID1'].astype(int)).astype(str)+'+'+(apple_df3['ID2'].astype(int)).astype(str)
        apple_df2 = apple_df2.loc[ apple_df2['pair_str'].isin(apple_df3['pair_str']) == True ].reset_index(drop=True)
                
        # good luck...

        # take the top pair and add it to an array:
        if len(apple_df2) >= N_controls:   ### ~~~ POSSIBLE THE TWO CHOSEN PAIRS HAVE OVERLAPPING GALAXIES ~~~ ###
            control_df = apple_df2.iloc[:N_controls]
            # print('Success at attempt', tries, Pp)
            # controls.append(apple_df2.iloc[:N_controls])
            # give me a progress message every 500 iterations
            # if np.any(milestones==(i+1)) == True:
            #     print('{0}/{1} controls selected for {2}'.format(i+1, len(pair_df), field))
            # finish = time.perf_counter()
            # if finish - start > 150:
            #     print('time exceeded', os.getpid())
            #     print(M1, z1, M2, z2, dM, dz, Pp)
            got = True
        else:
            # get an array of str IDs that have been tried already
            tried_arr = np.concatenate( (tried_arr, apple_df.loc[ apple_df['reuse_flag'] == 0, 'pair_str' ]) ) 
            dz = dz + 0.03
            dM = dM + 0.03 ####
            dE = dE + 1
            dS = dS + 0.05
            # add some kind of flag if there really aren't any good matches in Cp...
            if dM > 0.6:
                print('dM exceeded 0.6 for pair {0}-{1}'.format(ID1, ID2))
                report = True

#                 if dM > dM_lim:
#                     print('search has exceeded appropriate mass threshold...')
#                     print('to avoid bias, we are going to select the closest matches thus far...')
#                     print('BUG OUT IN', os.getpid(), ID1, M1, z1, E1, ID2, M2, z2, E2, Pp)
#                     # print('but we tried...', np.sort(apple_df['Cp'])[-3:])
#                     # print('good selections in apple_df2 =', apple_df2)
#                     # just take them...
#                     print('just gonna take the top choices for now...')
#                     # add pair information:
#                     apple_df['P_ID1'] = np.array([ID1]*len(apple_df), dtype=int)
#                     apple_df['P_ID2'] = np.array([ID2]*len(apple_df), dtype=int)
#                     apple_df['Pp'] = np.array([Pp]*len(apple_df), dtype=float)

#                     apple_df['dif'] = (10*(np.log10(apple_df['Cp']) - np.log10(Pp))**2 + 
#                                     (apple_df['z1'] - z1)**2 + (apple_df['MASS1'] - M1)**2 + 0.5*(apple_df['ENV1'] - E1)**2 +
#                                 (apple_df['z2'] - z2)**2 + (apple_df['MASS2'] - M2)**2 + 0.5*(apple_df['ENV2'] - E2)**2)
#                     # now sort on this
#                     apple_df = apple_df.loc[ (apple_df['ID1'] != apple_df['ID2']) & 
#                                             (apple_df['arc_sep'] > max_R_kpc) ].reset_index(drop=True)
#                     apple_df.sort_values(by=['dif'], inplace=True, ascending=True, ignore_index=True) # this resets the index
#                     print('settled for')
#                     # print(apple_df.loc[ :10, ['pair_str','z1','z2', 'Cp','dif', 'P_ID1','P_ID2', 'Pp'] ])
#                     apple_df = apple_df.drop_duplicates(['ID1'])
#                     apple_df = apple_df.drop_duplicates(['ID2'])
#                     print(apple_df.loc[ :, ['ID1', 'ID2', 'Cp', 'P_ID1', 'P_ID2', 'Pp'] ])

#                     if len(apple_df) < N_controls:
#                         print('oops...')
#                         report = True
#                         # sys.exit()
#                         dS = dS + 0.5
#                         dz = base_dz # need to once with base masses then need to groe exponentially...
#                         dM = base_dM # if I chunk up the convdif then it shouldn't be a problem regardless... 
#                         dE = base_dE # unless memory issue is merge itself.. 
#                         # catch_flag = True
#                     else:
#                         print('got it!', tries, Pp)
#                         control_df = apple_df.iloc[:N_controls]
#                         got = True

#                 # should save some that actually match the criteria before switching regions:
#                 N_good = len(apple_df2)
#                 if N_good > 0:
#                     good_old_df = apple_df2.iloc[:N_good]
#                     print('TAKING {} INTO NEXT REGION'.format(N_good))
#                     print(good_old_df)
#                     # N_controls = N_controls - N_good
#                     good_old = True

#                 # take out this Qd from the list
#                 Qd_old = Qd
#                 n_Qds = all_Qds[np.where(all_Qds != Qd)[0]]
#                 print('%%%%')
#                 print('TEST', n_Qds, all_Qds[np.where(all_Qds != Qd)[0]])
#                 print(Qd_old, all_Qds)
#                 print('%%%%')
#                 Qd = random.choice(n_Qds)
#                 print('moving from Q{0} to Q{1}'.format(Qd_old, Qd)) 
#                 # print('length test:', len(n_Qds))
#                 st = time.perf_counter()

            # # tell me if it's searched over everything and found nothing:
            # if ((z1 + dz > 3) and (z1 - dz < 0.5)) or ((z2 + dz > 3) and (z2 - dz < 0.5)):
            #     print('z FAIL -> NOTHING IN RANGE', ID1, M1, z1, E1, ID2, M2, z2, E2)
            # if ((M1 + dM > 12) and (M1 - dM < mass_lo)) or ((M2 + dM > 12) and (M2 - dM < mass_lo)):
            #     print('M FAIL -> NOTHING IN RANGE', ID1, M1, z1, E1, ID2, M2, z2, E2)
            # # if dz > 0.5:
            # #     print('dz exceeds 0.5', field, ID1, M1, z1, ID2, M2, z2, Pp, np.max(apple_df['Cp']), len(apple_df), len(apple_df2))


    # # combine all control_dfs into one and save for this field:
    # # controls is a list of a 2 index dfs, so need to combine these all\
    # control_df = pd.DataFrame( np.concatenate(controls), columns=pd.DataFrame(controls[0]).columns )
    # # add field information

    # print(dM)
    # try removing pait str and add in post: #
    control_df = control_df.drop(columns={'pair_str'})
    # control_df['field'] = [field]*len(control_df)

    # print('Process {} complete. Returning...'.format(os.getpid()))
#     if save == True:
#         # save the field df as a csv
#         control_df.to_csv(conv_PATH+'control_output/control_N-'+str(N_controls)+'_ztype-'+z_type+'_'+field+'_'+date+'.csv', index=False)
#         print('Control df saved in {}'.format(field))

       # if I parallelize this, I would need to return each processor's controls and join them IN ORDER      

    # print('Apples Bobbed:', i, ID1, ID2)

    if report == True:
        print('~~ Made it for Cp = 0 ~~',  ID1, ID2, Pp)
        print(control_df)

    # cache successful ID:
    # check_list[i] = 1
    # print('heo')
    # check_list_db = list(check_list)
    # print('heo2')
    # print(check_list_db)
    # sys.exit()
    # print(sum(check_list_db))
    # run a test on successful IDs - how many are completed?
    # print('NUMBER COMPLETED:', np.sum(controlled_IDs[field] / len(controlled_IDs[field])))
    # remaining_IDXs = np.where(controlled_IDs[field] == 0)[0]
#         print('hi.........')
#         num_left = len(check_list) - np.sum(check_list) # don't like to do the calculation...
#         print('hi...')
#         print(np.sum(check_list), len(check_list))
#         if num_left < 100:

#             UN_PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/conv_prob/UNMATCHED/'
#             print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#             print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#             # simply save the remaining controlled_IDs:
#             # save np array
#             np.savetxt(UN_PATH+field+'_'+str(num_left)+'_UNCONTROLLED.txt', controlled_IDs[field], fmt='%d')
#             # print(ID1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_ID')][0][0])
#             print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SAVED ~~~~~~~~~~~~~~~~~~~~~~~~~')
#             print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    return control_df

    # except:
    #     # Put all exception text into an exception and raise that
    #     print('FAILED')
    #     print("".join(traceback.format_exception(*sys.exc_info())))
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     raise Exception("".join(traceback.format_exception(*sys.exc_info())))
        

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def randbob_pll(ID): ### beware this will only work for the duplicate pairs
    
    field = var_dict['field']
    iso_pool_df = iso_pool_dict[field]
    PDF_array = PDF_dict[field]
    z_01 = PDF_array[0,1:]
    # milestones = np.linspace(100, 100000, 1000, dtype=int)
    
    # can recover pair_df data from var_dict:
    pair_arr = np.frombuffer(var_dict['P']).reshape(var_dict['Pshape'])
    
    # # how can I skip this step... seems all I need are the IDs, so just pull these out and modify fidf accordingly
    # pair_df = pd.DataFrame(pair_arr, columns=var_dict['Pcols'])
    # pam_df = pair_df
    
    ### ~~~~~~
    # ID1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_ID')][0][0] # remember input here is prime_ID
    # get the rows where prime ID == this:
    ID1_col = np.where(var_dict['Pcols'] == 'prime_ID')[0][0]
    ID2_col = np.where(var_dict['Pcols'] == 'partner_ID')[0][0]
    pID_rows = np.where( pair_arr[:, ID1_col] == ID )[0]
    all_IDs = np.unique(np.concatenate((pair_arr[pID_rows, ID1_col], pair_arr[pID_rows, ID2_col]), axis=0))
    
    # also need z and mass for the prime_ID:
    ID1 = pair_arr[pID_rows, ID1_col]
    ID2 = pair_arr[pID_rows, ID2_col]
    z1_col = np.where(var_dict['Pcols'] == 'prime_z')[0][0]
    M1_col = np.where(var_dict['Pcols'] == 'prime_M')[0][0]
    LX1_col = np.where(var_dict['Pcols'] == 'prime_LX')[0][0]
    DON1_col = np.where(var_dict['Pcols'] == 'prime_IR_AGN_DON')[0][0]
    STR1_col = np.where(var_dict['Pcols'] == 'prime_IR_AGN_STR')[0][0]
    z1 = pair_arr[pID_rows, z1_col]
    M1 = pair_arr[pID_rows, M1_col]
    LX1 = pair_arr[pID_rows, LX1_col]
    DON1 = pair_arr[pID_rows, DON1_col]
    STR1 = pair_arr[pID_rows, STR1_col]
    # print(ID1,'|', ID2,'|', z1,'|', M1)
    
    ### ~~~~~~
    
    controls = []

    # shuffle the catalog
    
    idf = iso_pool_df.sample(frac=1, random_state=ID).reset_index(drop=True) # random generator may be bad for mp remember...

    # initilize beginning shuffled df idx:
    iidx = 0

    # match_list = []
    
    # go through all the prime IDs: in each field:

    # obj_df = pam_df.loc[ (pam_df['prime_ID'] == ID) ].reset_index(drop=True)
    # # remove and
    # all_IDs = np.unique(np.concatenate( (obj_df['prime_ID'], obj_df['partner_ID']) ))
    fidf = idf.drop( idf[idf['ID'].isin(all_IDs) == True].index )

    try: # to see if the iidx would go out of bounds for idf
        iidx_check = fidf.iloc[iidx+len(pID_rows)]
    except: # if it does, simply reshuffle and restart iidx
        idf = iso_pool_df.sample(frac=1).reset_index(drop=True)
        fidf = idf.drop( idf[idf['ID'].isin(all_IDs) == True].index )
        # print('{0} df shuffled, iidx = {1}'.format(field, iidx))
        iidx = 0

    # get index of last taken
    match_df = fidf.iloc[iidx:iidx+len(pID_rows)] # multiply by 2 and get rid of any negative Cps below:
    if len(match_df) == 0:
        print(iidx)
    iidx = match_df.index[-1]+1
    match_df = match_df.loc[:, ['ID','z','MASS','LX','IR_AGN_DON','IR_AGN_STR'] ].reset_index(drop=True) 
    match_df['ID1'] = np.array(ID1, dtype=int)
    match_df['z1'] = np.array(z1)
    match_df['MASS1'] = np.array(M1) 
    match_df['LX1'] = np.array(LX1)
    match_df['IR_AGN_DON1'] = np.array(DON1)
    match_df['IR_AGN_STR1'] = np.array(STR1)
    match_df = match_df.rename(columns={'ID': 'ID2', 'z':'z2', 'MASS':'MASS2', 'LX':'LX2',
                                       'IR_AGN_DON':'IR_AGN_DON2', 'IR_AGN_STR':'IR_AGN_STR2'})
    # add other prime ID information
    match_df['P_ID1'] = np.array(ID1)
    match_df['P_ID2'] = np.array(ID2)
    match_df['Cp'] = Convdif(z_01, PDF_array[match_df['ID1'],1:], PDF_array[match_df['ID2'],1:], dv_lim=max_dv)
    
    # match_list.append(match_df)
    # # concat all the match dfs and add to the dict:
    # control_df = pd.DataFrame( np.concatenate(match_list), columns=pd.DataFrame(match_list[0]).columns )

    print('ID Bobbed:', ID)
    
    return match_df

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


if __name__ == '__main__':
    main()