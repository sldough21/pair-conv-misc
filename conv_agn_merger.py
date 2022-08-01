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

from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
import math as m
from multiprocessing import Pool, freeze_support, RLock

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from scipy import signal
from scipy.interpolate import interp1d

import sys

PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/Pair Project - Updated Data/'
cPATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/'

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
# determine conservative ang separation correspondong to 150 kpc at z = 0.5
R_kpc = cosmo.arcsec_per_kpc_proper(0.5) # arcsec/kpc at z=0.5
max_R_kpc = (150*u.kpc * R_kpc) # in arcseconds ### this is the bug right here

mass_lo = 8.5 # lower mass limit of the more massive galaxy in a pair that we want to consider
gamma = 1.4 # for k correction calculation

max_sep = 150 # kpc
max_dv = 1000 # DUDE #

sigma_cut = 100 # for individual PDF broadness
zp_cut = 0 # for pairs that will negligently contribute to the final AGN fractions
hmag_cut = 100 # essentially no cut <- not important 
select_controls = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# main function
def main():
    print('beginning main()')
    
    # we want to parallelize the data by fields, so:
    all_fields = ['GDS','EGS','COS','GDN','UDS']#,'COSMOS'] # COS is for CANDELS COSMOS
    # all_fields = ['COSMOS']
    # process_samples('COSMOS')
    
    # Create a multiprocessing Pool
    pool = Pool() 
    pool.map(process_samples, all_fields)
            
    print('Done!')
    
    # close pool
    pool.close()
    pool.join()
    
def process_samples(field):
    print('beginning process_samples() for {}'.format(field))
    
    # load in catalogs
    if field == 'COSMOS':
        df = pd.read_csv(cPATH+'select_COSMOS2020.csv')
        df['SIG_DIFF'] = df['lp_zPDF_u68'] - df['lp_zPDF_l68']
        df = df[ (df['lp_type'] != 1) & (df['lp_type'] != -99) & (df['lp_mass_med'] > (mass_lo-1)) & 
           (df['FLAG_COMBINED'] == 0) & (df['SIG_DIFF'] < sigma_cut)]
        
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
        df = df[ (df['CLASS_STAR'] < 0.9) & (df['PHOTFLAG'] == 0) & (df['MASS'] > (mass_lo-1)) & (df['SIG_DIFF'] < sigma_cut) &
               (df['HMAG'] < hmag_cut) ]
        df = df.reset_index(drop=True)

        
    # df = df.iloc[:500]        
    
    # there is no data draw in this method, go straight to getting projected pairs based on prime
    results = determine_pairs(df, field)
    
    return
    

def determine_pairs(df, field):
    print('beginning determine_pairs() for ', field)
    # get preliminary list of pairs and isolated galaxies
    # make definite redshift cut:
    all_df = df[ (df['ZPHOT_PEAK'] >= 0.5) & (df['ZPHOT_PEAK'] <= 6.5) ]
    print(field, len(all_df))
    all_df = all_df.reset_index(drop=True)
    
    # calculate LX
    all_df['LX'] = ( all_df['FX'] * 4 * np.pi * ((cosmo.luminosity_distance(all_df['ZPHOT_PEAK']).to(u.cm))**2) * 
                                                                ((1+all_df['ZPHOT_PEAK'])**(gamma-2)) )
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # Flag IR AGN based on Donley and Stern
    # look at IR luminosities
    all_df['IR_AGN_DON'] = [0]*len(all_df)
    all_df['IR_AGN_STR'] = [0]*len(all_df)
    if field == 'GDN':
        all_df = all_df.rename(columns={'IRAC_CH1_SCANDELS_FLUX':'IRAC_CH1_FLUX', 'IRAC_CH1_SCANDELS_FLUXERR':'IRAC_CH1_FLUXERR',
                                        'IRAC_CH2_SCANDELS_FLUX':'IRAC_CH2_FLUX', 'IRAC_CH2_SCANDELS_FLUXERR':'IRAC_CH2_FLUXERR'})

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
    all_df.loc[ (all_df['IRAC_CH1_FLUX']/all_df['IRAC_CH1_FLUXERR'] < 3) | (all_df['IRAC_CH2_FLUX']/all_df['IRAC_CH2_FLUXERR'] < 3) |
               (all_df['IRAC_CH3_FLUX']/all_df['IRAC_CH3_FLUXERR'] < 3) | (all_df['IRAC_CH4_FLUX']/all_df['IRAC_CH4_FLUXERR'] < 3),
              ['IR_AGN_DON', 'IR_AGN_STR'] ] = 0
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    if field == 'GDS':
        all_df['DALE_AGN_FLAG'] = all_df['AGN_FLAG']
    elif field == 'COSMOS':
        all_df['DALE_AGN_FLAG'] = [0]*len(all_df)
    elif field == 'GDN':
        all_df['DALE_AGN_FLAG'] = all_df['X_RAY_FLAG']
    else:
        all_df['DALE_AGN_FLAG'] = all_df['AGNFLAG']
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    
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
    
    # get rid of duplicates in pair sample
    pair_df = pair_df[ (pair_df['mass_ratio'] >= 0) ]
    sorted_idx_df = pd.DataFrame(np.sort((pair_df.loc[:,['prime_index','partner_index']]).values, axis=1), 
                                    columns=(pair_df.loc[:,['prime_index','partner_index']]).columns).drop_duplicates()
    pair_df = pair_df.reset_index(drop=True)
    pair_df = pair_df.iloc[sorted_idx_df.index]
    
    # we only want pairs where the mass ratio is within 10
    pair_df = pair_df[ (pair_df['mass_ratio'] <= 1) ] 
    
    # calculate projected separation at z
    pair_df['kpc_sep'] = (pair_df['arc_sep']) / cosmo.arcsec_per_kpc_proper(all_df.loc[pair_df['prime_index'], 'ZPHOT_PEAK'])
    
    # get complete list of projected pairs -> no need to calculate dv in this method
    true_pairs = pair_df[ (pair_df['kpc_sep'] <= max_sep*u.kpc) ]
    
    # we only want to consider pairs where the prime index is above our threshold
    true_pairs = true_pairs.iloc[ np.where( np.array(all_df.loc[ true_pairs['prime_index'], 'MASS' ] > mass_lo) == True ) ]
    
    # add galaxies that aren't pairs into the isolated sample:
    iso_add = pair_df[ (pair_df['kpc_sep'] > max_sep) ]

    # just stack prime and partner indices into massive array:
    iso_add_idx = np.concatenate( (np.array(iso_add['prime_index']), np.array(iso_add['partner_index'])), axis=0)
    # return unique indices
    iso_add_uniq = np.unique(iso_add_idx)
    # get rid of cases where those indices appear elsewhere, so create array for true pair indices
    true_pair_idx = np.concatenate( (np.array(true_pairs['prime_index']), np.array(true_pairs['partner_index'])), axis=0)
    # only keep the elements that aren't in true pair:
    mask = np.isin(iso_add_uniq, true_pair_idx, invert=True)
    iso_unq = iso_add_uniq[mask]
    all_iso = np.concatenate( (iso_conf, iso_unq), axis=0)
    
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print('PDF width cut? ', sigma_cut)
    # print(field, len(all_df), len(np.unique(np.concatenate((true_pairs['prime_index'], true_pairs['partner_index'])))), len(iso_unq))
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                         
    # calculate pair fraction for each projected pair:
    pair_probs, pair_PdA = load_pdfs(all_df.loc[ true_pairs['prime_index'], 'ID'], all_df.loc[ true_pairs['partner_index'], 'ID'], 
                           true_pairs['arc_sep'], field)
    print('pair probability calculated in ', field)
    true_pairs['pair_prob'] = pair_probs
    
    np.save('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/conv_prob/'+field+'_arr_6.30', pair_PdA)
        
    print('{0}: pair_count = {1}, iso count = {2}'.format(field, len(true_pairs), len(all_iso)))
    
    # add back control galaxies that are only included in pairs where pair_prob < 0
    gtrue_pairs = true_pairs[ true_pairs['pair_prob'] >= zp_cut ] ### MAY NEED TO RESET INDEX ###
    gtrue_pairs = gtrue_pairs.reset_index(drop=True)
    
    # add galaxies that aren't pairs into the isolated sample:
    iso_add = true_pairs[ true_pairs['pair_prob'] < zp_cut ]
    # just stack prime and partner indices into massive array:
    iso_add_idx = np.concatenate( (np.array(iso_add['prime_index']), np.array(iso_add['partner_index'])), axis=0)
    # return unique indices
    iso_add_uniq = np.unique(iso_add_idx)
    # get rid of cases where those indices appear elsewhere, so create array for true pair indices
    true_pair_idx = np.concatenate( (np.array(gtrue_pairs['prime_index']), np.array(gtrue_pairs['partner_index'])), axis=0)
    # only keep the elements that aren't in true pair:
    mask = np.isin(iso_add_uniq, true_pair_idx, invert=True)
    iso_unq = iso_add_uniq[mask]
    all_iso = np.concatenate( (all_iso, iso_unq), axis=0)
    
    print('{0}: pair_count = {1}, iso count = {2}'.format(field, len(gtrue_pairs), len(all_iso)))
    
    
    # pick out control galaxies
    iso_idx = all_iso
    iso_mass = all_df.loc[all_iso, 'MASS']
    iso_z = all_df.loc[all_iso, 'ZPHOT_PEAK']
    iso_sig = all_df.loc[all_iso, 'SIG_DIFF']
    pair_idx = np.concatenate( (gtrue_pairs['prime_index'], gtrue_pairs['partner_index']) )
    pair_mass = np.concatenate( (all_df.loc[ gtrue_pairs['prime_index'], 'MASS' ], all_df.loc[ gtrue_pairs['partner_index'], 'MASS' ]) )
    pair_z = np.concatenate( (all_df.loc[ gtrue_pairs['prime_index'], 'ZPHOT_PEAK' ], all_df.loc[ gtrue_pairs['partner_index'], 'ZPHOT_PEAK' ]) )
    pair_sig = np.concatenate( (all_df.loc[ gtrue_pairs['prime_index'], 'SIG_DIFF' ], all_df.loc[ gtrue_pairs['partner_index'], 'SIG_DIFF' ]) )
    
    gtrue_pairs['prime_ID'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'ID' ])
    gtrue_pairs['partner_ID'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'ID' ])
    gtrue_pairs['prime_z'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'ZPHOT_PEAK' ])
    gtrue_pairs['partner_z'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'ZPHOT_PEAK' ])
    gtrue_pairs['prime_M'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'MASS' ])
    gtrue_pairs['partner_M'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'MASS' ])
    gtrue_pairs['prime_LX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'LX' ])
    gtrue_pairs['partner_LX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'LX' ])
    gtrue_pairs['prime_PDFsig'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'SIG_DIFF'])
    gtrue_pairs['partner_PDFsig'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'SIG_DIFF'])
    gtrue_pairs['prime_IR_AGN_DON'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'IR_AGN_DON'])
    gtrue_pairs['prime_IR_AGN_STR'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'IR_AGN_STR'])
    gtrue_pairs['partner_IR_AGN_DON'] = np.array(all_df.loc[gtrue_pairs['partner_index'], 'IR_AGN_DON'])
    gtrue_pairs['partner_IR_AGN_STR'] = np.array(all_df.loc[gtrue_pairs['partner_index'], 'IR_AGN_STR'])
    gtrue_pairs['field'] = [field] * len(gtrue_pairs)
    
    if select_controls == True: ### DONT FORGET ABOUT SHUFFLING LATER ON
        controls, c_flag = get_control(iso_idx, iso_mass, iso_z, iso_sig, pair_idx, pair_mass, pair_z, pair_sig)
        # let's output the matched fraction based on c_flag
        c_flag_all = np.concatenate(c_flag)
        tp = len(c_flag_all)
        tm = len(c_flag_all[ np.where(c_flag_all == 0) ])
        tr = len(c_flag_all[ np.where(c_flag_all == 1) ])
        tf = len(c_flag_all[ np.where(c_flag_all == 2) ])
        print('UNIQUE MATCH FRACTION IN {} = {}'.format(field, tm/tp))
        print('RECYCLED FRACTION IN {} = {}'.format(field, tr/tp))
        print('MATCH FRACTION WITH RECYCLING IN {} = {}'.format(field, (tm+tr)/tp)) 
        print('FAIL FRACTION IN {} = {}'.format(field, tf/tp)) 
    else:
        controls = np.full((len(pair_idx), 2), -99)
        c_flag = np.full((len(pair_idx), 2), -99)
    
    print('made it here ', field)
    
    middle_idx = len(controls)//2
    prime_controls = controls[:middle_idx]
    # get a c1prime list without -99
    c1prime_no99 = prime_controls[:,0][np.where(prime_controls[:,0] != -99)]
    c2prime_no99 = prime_controls[:,1][np.where(prime_controls[:,1] != -99)]
    prime_flags = c_flag[:middle_idx]
    
    partner_controls = controls[middle_idx:]
    c1partner_no99 = partner_controls[:,0][np.where(partner_controls[:,0] != -99)]
    c2partner_no99 = partner_controls[:,1][np.where(partner_controls[:,1] != -99)]
    partner_flags = c_flag[middle_idx:]
    
    # add pair data to the dataframe: (all empty values are -99)
    gtrue_pairs['i1prime_idx'] = prime_controls[:,0]
    gtrue_pairs['i2prime_idx'] = prime_controls[:,1]
    gtrue_pairs['i1partner_idx'] = partner_controls[:,0]
    gtrue_pairs['i2partner_idx'] = partner_controls[:,1]
        
    gtrue_pairs['c1prime_ID'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1prime_idx'] != -99, 'c1prime_ID' ] = np.array(all_df.loc[ c1prime_no99, 'ID' ])    
    gtrue_pairs['c1prime_z'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1prime_idx'] != -99, 'c1prime_z' ] = np.array(all_df.loc[ c1prime_no99, 'ZPHOT_PEAK' ])
    gtrue_pairs['c1prime_M'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1prime_idx'] != -99, 'c1prime_M' ] = np.array(all_df.loc[ c1prime_no99, 'MASS' ])
    gtrue_pairs['c1prime_sig'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1prime_idx'] != -99, 'c1prime_sig' ] = np.array(all_df.loc[ c1prime_no99, 'SIG_DIFF' ])
    gtrue_pairs['c1prime_LX'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1prime_idx'] != -99, 'c1prime_LX' ] = np.array(all_df.loc[ c1prime_no99, 'LX' ])
    gtrue_pairs['c1prime_IR_AGN_DON'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1prime_idx'] != -99, 'c1prime_IR_AGN_DON' ] = np.array(all_df.loc[ c1prime_no99, 'IR_AGN_DON' ]) 
    gtrue_pairs['c1prime_IR_AGN_STR'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1prime_idx'] != -99, 'c1prime_IR_AGN_STR' ] = np.array(all_df.loc[ c1prime_no99, 'IR_AGN_STR' ]) 
    
    gtrue_pairs['c2prime_ID'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2prime_idx'] != -99, 'c2prime_ID' ] = np.array(all_df.loc[ c2prime_no99, 'ID' ])
    gtrue_pairs['c2prime_z'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2prime_idx'] != -99, 'c2prime_z' ] = np.array(all_df.loc[c2prime_no99, 'ZPHOT_PEAK' ])
    gtrue_pairs['c2prime_M'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2prime_idx'] != -99, 'c2prime_M' ] = np.array(all_df.loc[ c2prime_no99, 'MASS' ])
    gtrue_pairs['c2prime_sig'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2prime_idx'] != -99, 'c2prime_sig' ] = np.array(all_df.loc[ c2prime_no99, 'SIG_DIFF' ])
    gtrue_pairs['c2prime_LX'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2prime_idx'] != -99, 'c2prime_LX' ] = np.array(all_df.loc[ c2prime_no99, 'LX' ])
    gtrue_pairs['c2prime_IR_AGN_DON'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2prime_idx'] != -99, 'c2prime_IR_AGN_DON' ] = np.array(all_df.loc[ c2prime_no99, 'IR_AGN_DON' ]) 
    gtrue_pairs['c2prime_IR_AGN_STR'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2prime_idx'] != -99, 'c2prime_IR_AGN_STR' ] = np.array(all_df.loc[ c2prime_no99, 'IR_AGN_STR' ]) 
    
    gtrue_pairs['c1partner_ID'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1partner_idx'] != -99, 'c1partner_ID' ] = np.array(all_df.loc[ c1partner_no99, 'ID' ])
    gtrue_pairs['c1partner_z'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1partner_idx'] != -99, 'c1partner_z' ] = np.array(all_df.loc[ c1partner_no99, 'ZPHOT_PEAK' ])
    gtrue_pairs['c1partner_M'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1partner_idx'] != -99, 'c1partner_M' ] = np.array(all_df.loc[ c1partner_no99, 'MASS' ])
    gtrue_pairs['c1partner_sig'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1partner_idx'] != -99, 'c1partner_sig' ] = np.array(all_df.loc[ c1partner_no99, 'SIG_DIFF' ])
    gtrue_pairs['c1partner_LX'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1partner_idx'] != -99, 'c1partner_LX' ] = np.array(all_df.loc[ c1partner_no99, 'LX' ])
    gtrue_pairs['c1partner_IR_AGN_DON'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1partner_idx'] != -99, 'c1partner_IR_AGN_DON' ] = np.array(all_df.loc[ c1partner_no99, 'IR_AGN_DON' ]) 
    gtrue_pairs['c1partner_IR_AGN_STR'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1partner_idx'] != -99, 'c1partner_IR_AGN_STR' ] = np.array(all_df.loc[ c1partner_no99, 'IR_AGN_STR' ]) 
    
    gtrue_pairs['c2partner_ID'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2partner_idx'] != -99, 'c2partner_ID' ] = np.array(all_df.loc[ c2partner_no99, 'ID' ])
    gtrue_pairs['c2partner_z'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2partner_idx'] != -99, 'c2partner_z' ] = np.array(all_df.loc[ c2partner_no99, 'ZPHOT_PEAK' ])
    gtrue_pairs['c2partner_M'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2partner_idx'] != -99, 'c2partner_M' ] = np.array(all_df.loc[ c2partner_no99, 'MASS' ])
    gtrue_pairs['c2partner_sig'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2partner_idx'] != -99, 'c2partner_sig' ] = np.array(all_df.loc[ c2partner_no99, 'SIG_DIFF' ])
    gtrue_pairs['c2partner_LX'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2partner_idx'] != -99, 'c2partner_LX' ] = np.array(all_df.loc[ c2partner_no99, 'LX' ])
    gtrue_pairs['c2partner_IR_AGN_DON'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2partner_idx'] != -99, 'c2partner_IR_AGN_DON' ] = np.array(all_df.loc[ c2partner_no99, 'IR_AGN_DON' ]) 
    gtrue_pairs['c2partner_IR_AGN_STR'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2partner_idx'] != -99, 'c2partner_IR_AGN_STR' ] = np.array(all_df.loc[ c2partner_no99, 'IR_AGN_STR' ]) 
    
    # just tack on counts for total isolated galaxies:
    # gtrue_pairs['iso_count'] = [len(all_iso)] * len(gtrue_pairs)
    
    gtrue_pairs['c1prime_flag'] = prime_flags[:,0]
    gtrue_pairs['c2prime_flag'] = prime_flags[:,1]
    gtrue_pairs['c1partner_flag'] = partner_flags[:,0]
    gtrue_pairs['c2partner_flag'] = partner_flags[:,1]
    
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    
    gtrue_pairs['prime_DALE_AGN_FLAG'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'DALE_AGN_FLAG' ])
    gtrue_pairs['partner_DALE_AGN_FLAG'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'DALE_AGN_FLAG' ])
    
    gtrue_pairs['c1prime_DALE_AGN_FLAG'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1prime_idx'] != -99, 'c1prime_DALE_AGN_FLAG' ] = np.array(all_df.loc[ c1prime_no99, 'DALE_AGN_FLAG' ])   
    gtrue_pairs['c2prime_DALE_AGN_FLAG'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2prime_idx'] != -99, 'c2prime_DALE_AGN_FLAG' ] = np.array(all_df.loc[ c2prime_no99, 'DALE_AGN_FLAG' ])  
    gtrue_pairs['c1partner_DALE_AGN_FLAG'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i1partner_idx'] != -99, 'c1partner_DALE_AGN_FLAG' ] = np.array(all_df.loc[ c1partner_no99, 'DALE_AGN_FLAG' ])   
    gtrue_pairs['c2partner_DALE_AGN_FLAG'] = [-99]*len(gtrue_pairs)
    gtrue_pairs.loc[ gtrue_pairs['i2partner_idx'] != -99, 'c2partner_DALE_AGN_FLAG' ] = np.array(all_df.loc[ c2partner_no99, 'DALE_AGN_FLAG' ])  
    
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    
    gtrue_pairs.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/conv_prob/'+field+'_6.30.csv', index=False)
        
    print('saved in ', field)
    
    return gtrue_pairs
    
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def load_pdfs(gal1, gal2, theta, field):
    print('beginning conv_prob() for ', field)
    
    all_prob = []
    all_PdA = []
    
    print(field, len(gal1))
    
    dA = np.linspace(0, 150, num=1500, endpoint=True)
    
    # the COSMOS PDF are all loaded together, so do this outside for loop:
    if field == 'COSMOS':
        with fits.open(cPATH+'COSMOS2020_R1/PZ/COSMOS2020_CLASSIC_R1_v2.0_LEPHARE_PZ.fits') as data:
            # fix big endian buffer error:
            COSMOS_PZ_arr = np.array(data[0].data)
        COSMOS_PZ_arrf = COSMOS_PZ_arr.byteswap().newbyteorder()
        COSMOS_PZ = pd.DataFrame(COSMOS_PZ_arrf)
        z = COSMOS_PZ.loc[0,1:].to_numpy()
        COSMOS_PZ = COSMOS_PZ.T
    
    for ID1, ID2, th in tqdm(zip(gal1, gal2, theta), miniters=100): 
        # load PDFs based on string ID
        ID1_str = str(ID1)
        if len(ID1_str) == 1: id_string1 = '0000'+ID1_str
        if len(ID1_str) == 2: id_string1 = '000'+ID1_str
        if len(ID1_str) == 3: id_string1 = '00'+ID1_str
        if len(ID1_str) == 4: id_string1 = '0'+ID1_str
        if len(ID1_str) == 5: id_string1 = ID1_str
        ID2_str = str(ID2)
        if len(ID2_str) == 1: id_string2 = '0000'+ID2_str
        if len(ID2_str) == 2: id_string2 = '000'+ID2_str
        if len(ID2_str) == 3: id_string2 = '00'+ID2_str
        if len(ID2_str) == 4: id_string2 = '0'+ID2_str
        if len(ID2_str) == 5: id_string2 = ID2_str
        
        if field == "GDS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSS_ID'+id_string1+'.pzd'
            pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSS_ID'+id_string2+'.pzd'
            # read the PDFs
            PDF1_load = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF2_load = pd.read_csv(pdf_filename2, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF1 = np.array(PDF1_load['mFDa4'])
            PDF2 = np.array(PDF2_load['mFDa4'])
            z = np.array(PDF1_load['z'])
            
        elif field == "EGS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/EGS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_EGS_ID'+id_string1+'.pzd'
            pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/EGS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_EGS_ID'+id_string2+'.pzd'
            # read the PDFs
            PDF1_load = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF2_load = pd.read_csv(pdf_filename2, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF1 = np.array(PDF1_load['mFDa4'])
            PDF2 = np.array(PDF2_load['mFDa4'])
            z = np.array(PDF1_load['z'])
            
        elif field == "GDN":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSN_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSN_ID'+id_string1+'.pzd'
            pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSN_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSN_ID'+id_string2+'.pzd'
            # read the PDFs
            PDF1_load = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF2_load = pd.read_csv(pdf_filename2, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF1 = np.array(PDF1_load['mFDa4'])
            PDF2 = np.array(PDF2_load['mFDa4'])
            z = np.array(PDF1_load['z'])
            
        elif field == "COS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/COSMOS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_COSMOS_ID'+id_string1+'.pzd'
            pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/COSMOS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_COSMOS_ID'+id_string2+'.pzd'
            # read the PDFs
            PDF1_load = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF2_load = pd.read_csv(pdf_filename2, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF1 = np.array(PDF1_load['mFDa4'])
            PDF2 = np.array(PDF2_load['mFDa4'])
            z = np.array(PDF1_load['z'])
            
        elif field == "UDS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/UDS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_UDS_ID'+id_string1+'.pzd' 
            pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/UDS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_UDS_ID'+id_string2+'.pzd' 
            # read the PDFs
            PDF1_load = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF2_load = pd.read_csv(pdf_filename2, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                      'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            PDF1 = np.array(PDF1_load['mFDa4'])
            PDF2 = np.array(PDF2_load['mFDa4'])
            z = np.array(PDF1_load['z'])
            
        elif field == 'COSMOS':
            PDF1 = np.array(COSMOS_PZ.loc[ 1:, ID1 ])
            PDF2 = np.array(COSMOS_PZ.loc[ 1:, ID2 ])
        
        Cv_prob = Convdif(z, PDF1, PDF2, dv_lim=max_dv)
        
        # should do the angular diameter distance here and integrate for each bin in post
        PdA = PdA_prob(PDF1, PDF2, th, z, dA)
        
        all_prob.append(Cv_prob)
        all_PdA.append(PdA)
        
        ### WRITE THIS TO SPIT OUT ARRAYS FOR THE ENTIRE GROUP ###
        ### WILL ALSO NEED THIS FROM P(Z1Z2) = NORMALIZED PDF1 * PDF2 ###
    
    return all_prob, np.array(all_PdA)

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

def Convdif(z_all, Pz1, Pz2, dv_lim=1000):
        
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
    v_conv = signal.fftconvolve(fintp2(v_new), fintp1(v_new)[::-1], mode='full')
    v_conv = v_conv / np.trapz(v_conv, x=all_ve) # normalize area

    # integrate velocity convolution to find probability dv within dv_lim
    rnge = tuple(np.where( (all_ve > -dv_lim) & (all_ve < dv_lim)))
    prob = np.trapz(v_conv[rnge], x=all_ve[rnge])
    
    return prob

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def PdA_prob(PDF1, PDF2, theta, z, dA):
    
    comb_PDF = PDF1*PDF2
    comb_PDF = comb_PDF / np.trapz(comb_PDF, x=z)
    
    # # attempt change of variable
    # dA = ( ang_diam_dist(z, theta) )
    # dAdz = np.gradient(dA,z)
    # dzdA_test = dAdz**(-1)

    # # will need to split into two monochromatic parts of dA:
    # split_z = z[np.where(dA==np.max(dA))]

    # so split 0-1.61 (1) and 1.62-10 (2)
    dA1 = ang_diam_dist( z[np.where(z <= 1.61)], theta )
    dA2 = ang_diam_dist( z[np.where(z > 1.61)], theta )
    z1 = z[np.where(z <= 1.61)]
    z2 = z[np.where(z > 1.61)]
    comb_PDF1 = comb_PDF[np.where(z <= 1.61)] 
    comb_PDF2 = comb_PDF[np.where(z > 1.61)] 

    PdA11 = comb_PDF1 * np.abs(dzdA(dA1, z1, theta))
    PdA11 = np.nan_to_num(PdA11) # fill the nan casued by divisiom ny zero
    PdA12 = comb_PDF2 * np.abs(dzdA(dA2, z2, theta))

    dA_new = dA
    fintp1 = interp1d(dA1, PdA11, kind='linear', bounds_error=False, fill_value=0)
    fintp2 = interp1d(dA2, PdA12, kind='linear', bounds_error=False, fill_value=0)

    intr_PdA1 = fintp1(dA_new)
    intr_PdA2 = fintp2(dA_new)

    PdA = intr_PdA1+intr_PdA2
    
    PdA = PdA / np.trapz(PdA, x=dA_new)
    
    return np.nan_to_num(PdA)


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

# def angular diameter function:
def ang_diam_dist(z, theta, H0=70, Om_m=0.3, Om_rel=0, Om_lam=0.7, Om_0=1):
    
    c = 2.998e5 # km/s
    try:
        zs = np.linspace(0,z,10, endpoint=True, axis=1) # numerically integrate
    except:
        zs = np.linspace(0,z,10, endpoint=True)
    dA = ( c / (H0*(1+z)) ) * np.trapz( ( 1 / np.sqrt( Om_m*(1+zs)**3 + Om_rel*(1+zs)**4 + Om_lam + (1-Om_0)*(1+zs)**2 ) ), x=zs )
    
    return dA * theta * 1000 / ((180/np.pi)*3600)

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
    
    dzdA = dzdA * theta * 1000 / ((180/np.pi)*3600) # convert to kpc/"
        
    return dzdA

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


def get_control(control_ID, control_mass, control_z, control_sig, # want to try matching to PDF width
                gal_ID, mass, redshift, sigma, N_control=2, zfactor=0.2, mfactor=2, sigfactor=0.25): 
    
    dz = zfactor
    cg = 0
    cr = 0
    
    ### TRY SHUFFLING PAIR ARRAYS ### ADD DISTANCE METRIC ###
    # added combinatory distance metric, will need to play around with whether I want hard z and mass control limits...

    # create array to store lists of control indices per prime galaxy
    control_all = np.full((len(gal_ID), N_control), -99)
        
    # create a list for all ID's to make sure there are no duplicates
    control_dup = []
    
    # create a flag for each paired galaxy to keep track of what ID's had repeated control galaxies and if it's empty
    control_flag = np.full((len(gal_ID), N_control), 2) # 0 = unique control, 1 = repeated control, 2 = no match
    
    # create a dataframe from the isolated galaxy data
    iso = {'ID':control_ID, 'z':control_z, 'mass':control_mass, 'PDFsig':control_sig}
    all_iso_df = pd.DataFrame( iso )
    # somehow indices were carried with these values, so if we want to index a list of indices we want:
    all_iso_df = all_iso_df.reset_index(drop=True)
    
    for i, (ID, m, z, sig) in enumerate(zip(gal_ID, mass, redshift, sigma)):
        
        control = np.full(2, -99)

        zmin = z - dz
        zmax = z + dz
        mmin = m-np.log10(mfactor)
        mmax = m+np.log10(mfactor)
        smin = sig - sigfactor
        smax = sig + sigfactor

        # create a dataframe for possible matches
        cmatch_df = all_iso_df[ (all_iso_df['z'] >= zmin) & (all_iso_df['z'] <= zmax) & (all_iso_df['mass'] >= mmin) &
                               (all_iso_df['mass'] <= mmax) & (all_iso_df['PDFsig'] >= smin) & (all_iso_df['PDFsig'] <= smax) ]
        
        # create columns for difference between z/mass control and pair z/m
        cmatch_df['dif'] = (cmatch_df['z'] - z)**2 + (cmatch_df['mass'] - m)**2 + 0.5*(cmatch_df['PDFsig'] - sig)**2
                
        # need to sort dataframe based on two columns THEN continue
        cmatch_df.sort_values(by=['dif'], inplace = True, ascending = True)

        # immediately get rid of control galaxies that have already been selected
        cmatch_df = cmatch_df[ ((cmatch_df['ID']).isin(control_dup) == False) ]
        
        mcount = 0

        for iso_ID in cmatch_df['ID']: # could simply append the first two rows... but short enough for loop here

            control[mcount] = iso_ID
            control_dup.append(iso_ID) # issue is here
            control_flag[i,mcount] = 0
            mcount+=1

            if mcount == N_control:
                control_all[i] = control
                break
                
        if mcount == N_control: # if we have two unique controls, move onto the next paired galaxy
            continue
        # if mcount == 1:
        #     print('HERE', i, control)
        
        # if I can't match both, see if the paired galaxy has any previous independent matches and choose from those
        else: 
                        
            # then get the iso ID's from the same pair ID (can appear more than once)
            prev_control = control_all[ np.where(gal_ID[:i] == ID) ]
            aa = np.where(gal_ID[:i] == ID)
            
            # skip if it doesn't appear before
            if len(prev_control) == 0:
                cg += 1
                control_all[i] = control
                continue
            
            all_prev = np.concatenate(prev_control, axis=0)
            all_prev = all_prev[ np.where(all_prev >= 0) ]
            
            # skip of the previous iteration of this index had nothing
            if len(all_prev) == 0:
                cg+=1
                control_all[i] = control
                continue
                            
            # if there are, draw the amount you need
            if len(all_prev) >= N_control-mcount:
                rd_idx = random.choice( np.arange(0,len(all_prev),1), N_control-mcount, replace=False )
            else:
                rd_idx = random.choice( np.arange(0,len(all_prev),1), len(all_prev), replace=False )
            recycled = all_prev[rd_idx]
            
            # mcount is how many have already been chosen
            # rcount is how many we can add
            rcount = len(recycled)
            for add in recycled:
                control[mcount] = add
                control_flag[i, mcount] = 1
                mcount+=1     
            cr+=1
        control_all[i] = control

    
#     print('Failed on ', cg)
#     print('Replaced ', cr)
    
    
    # return np.asarray(control_all, dtype=object)
    return control_all, control_flag

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


if __name__ == '__main__':
    main()