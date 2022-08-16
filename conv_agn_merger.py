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
mPATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/MODEL/Input_data/'
conv_PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/conv_prob/'

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
# determine conservative ang separation correspondong to 150 kpc at z = 0.5
R_kpc = cosmo.arcsec_per_kpc_proper(0.5) # arcsec/kpc at z=0.5
max_R_kpc = 120*u.kpc * R_kpc # in arcseconds ### this is the bug right here

mass_lo = 10 # lower mass limit of the more massive galaxy in a pair that we want to consider
gamma = 1.4 # for k correction calculation

max_sep = 75 # 150 kpc <== should make farther out so as to not get varying sep measurements based on prime/partner z that don't cut them
max_dv = 1000 

sigma_cut = 100 # for individual PDF broadness
zp_cut = 0 # for pairs that will negligently contribute to the final AGN fractions
hmag_cut = 100 # essentially no cut <- not important 
select_controls = False
duplicate_pairs = False
apple_bob = True
save = True
z_type = 'ps' # ['p', 'ps' ,'s']
date = '8.17' ### can automate this you know ###
num_proc = 20
min_pp = 1.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# main function
def main():
    print('beginning main()')
    
    # we want to parallelize the data by fields, so:
    all_fields = ['GDS','EGS','COS','GDN','UDS','COSMOS'] # COS is for CANDELS COSMOS
    # all_fields = ['GDN']
    # all_fields = ['COSMOS']
    # process_samples('COSMOS')
    
    # Create a multiprocessing Pool
    pool = Pool(num_proc) 
    pool.map(process_samples, all_fields)
            
    print('Done!')
    
    # close pool
    pool.close()
    pool.join()
    
    return
    
def process_samples(field):
    print('beginning process_samples() for {}'.format(field))
    
    # load in catalogs: <== specify column dtypes
    if field == 'COSMOS':
        df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/CANDELS_COSMOS_CATS/'+field+'_data.csv',
                        dtype={'ZSPEC_R':object})
        df = df.loc[ (df['LP_TYPE'] != 1) & (df['LP_TYPE'] != -99) & (df['MASS'] > (mass_lo)) & # (mass_lo-1)
            (df['FLAG_COMBINED'] == 0) & (df['SIG_DIFF'] < sigma_cut)]
    else:
        df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/CANDELS_COSMOS_CATS/'+field+'_data.csv',
                        dtype={'ZSPEC_R':object})
        df = df[ (df['CLASS_STAR'] < 0.9) & (df['PHOTFLAG'] == 0) & (df['MASS'] > (mass_lo)) & 
            (df['SIG_DIFF'] < sigma_cut) ]
    df = df.reset_index(drop=True)

        
    # ### ~~~ TEST RUN ~~~ ###
    # df = df.iloc[:1000]        
    
    # there is no data draw in this method, go straight to getting projected pairs based on prime
    determine_pairs(df, field)
    
    return
    

def determine_pairs(df, field):
    print('beginning determine_pairs() for ', field)
    
    # first thing is change df based on z_type
    df['z'] = df['ZPHOT_PEAK']
    if z_type == 'ps':
        df.loc[ df['ZBEST_TYPE'] == 's', 'z' ] = df['ZSPEC']
        df.loc[ df['ZBEST_TYPE'] == 's', 'SIG_DIFF' ] = 0
    
    # make definite redshift cut:
    all_df = df.loc[ (df['z'] >= 0.5) & (df['z'] <= 3.0) ]  ### ~~~ GOTTA BE CAREFUL WITH EDGES ~~~ ###
    print('all_df length', field, len(all_df))
    all_df = all_df.reset_index(drop=True)
    
    # calculate LX
    all_df['LX'] = ( all_df['FX'] * 4 * np.pi * ((cosmo.luminosity_distance(all_df['z']).to(u.cm))**2).value * 
                                                                ((1+all_df['z'])**(gamma-2)) )
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # Flag IR AGN based on Donley and Stern
    # look at IR luminosities
    all_df['IR_AGN_DON'] = [0]*len(all_df)
    all_df['IR_AGN_STR'] = [0]*len(all_df)

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
        true_pairs = true_pairs.iloc[ np.where( np.array(all_df.loc[ true_pairs['prime_index'], 'MASS' ] > mass_lo) == True ) ]
        
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
                         
    # # calculate pair fraction for each projected pair:
    pair_probs, pair_PdA = load_pdfs(all_df.loc[ true_pairs['prime_index'], 'ID'], all_df.loc[ true_pairs['partner_index'], 'ID'],
                                     all_df.loc[ true_pairs['prime_index'], 'z'], all_df.loc[ true_pairs['partner_index'], 'z'],
                                    all_df.loc[ true_pairs['prime_index'], 'ZBEST_TYPE'], 
                                     all_df.loc[ true_pairs['partner_index'], 'ZBEST_TYPE'],
                                     true_pairs['arc_sep'], field)
    
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
    
    
    print('pair probability calculated in ', field)
    true_pairs['pair_prob'] = pair_probs
    true_pairs = true_pairs.reset_index(drop=True)
    
    # add back control galaxies that are only included in pairs where pair_prob < 0
    gtrue_pairs = true_pairs[ true_pairs['pair_prob'] >= zp_cut ] ### MAY NEED TO RESET INDEX ###
    # print(gtrue_pairs.index, len(pair_PdA))
    pair_PdA_gt0 = pair_PdA[gtrue_pairs.index]
    # save PdA arrays
    hdu_dA = fits.PrimaryHDU(pair_PdA_gt0)
    hdul_dA = fits.HDUList([hdu_dA])
    if save == True:
        hdul_dA.writeto(conv_PATH+'PdA_output/PdA_ztype-'+z_type+'_'+field+'_'+date+'.fits', overwrite=True)
        print('PdA array saved in {}'.format(field))
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
    iso_z = all_df.loc[all_iso, 'z']
    iso_sig = all_df.loc[all_iso, 'SIG_DIFF']
    pair_idx = np.concatenate( (gtrue_pairs['prime_index'], gtrue_pairs['partner_index']) )
    pair_mass = np.concatenate( (all_df.loc[ gtrue_pairs['prime_index'], 'MASS' ], all_df.loc[ gtrue_pairs['partner_index'], 'MASS' ]) )
    pair_z = np.concatenate( (all_df.loc[ gtrue_pairs['prime_index'], 'z' ], all_df.loc[ gtrue_pairs['partner_index'], 'z' ]) )
    pair_sig = np.concatenate( (all_df.loc[ gtrue_pairs['prime_index'], 'SIG_DIFF' ], all_df.loc[ gtrue_pairs['partner_index'], 'SIG_DIFF' ]) )
    
    gtrue_pairs['prime_ID'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'ID' ])
    gtrue_pairs['partner_ID'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'ID' ])
    gtrue_pairs['prime_z'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'z' ])       ### ~~~ append redshift type ~~~ ###
    gtrue_pairs['prime_zt'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'ZBEST_TYPE' ])
    gtrue_pairs['partner_z'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'z' ])
    gtrue_pairs['partner_zt'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'ZBEST_TYPE' ])
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
    
    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    # need to assign all galaxies in all_df a pair probability for any pair within max_R_kpc:
    # WILL ONLY WORK IF I AM USING THE DUPLICATE PAIR LIST, probably need a for loop:
    print('Determing all pair prob for all galaxies in {}'.format(field))
    all_df['all_pp'] = [0]*len(all_df)
    for i, ID in enumerate(all_df['ID']):
        # get parts of true_pairs where ID is prime ID
        if duplicate_pairs == False:
            gal_match_probs = gtrue_pairs.loc[ (gtrue_pairs['prime_ID'] == ID) | (gtrue_pairs['partner_ID'] == ID), 'pair_prob' ]
        elif duplicate_pairs == True:
            gal_match_probs = gtrue_pairs.loc[ (gtrue_pairs['prime_ID'] == ID), 'pair_prob' ]
        if len(gal_match_probs) == 0:
            all_df.loc[ i, 'all_pp'] = 0
        elif len(gal_match_probs) != 0: 
            all_df.loc[ i, 'all_pp'] = 1 - np.prod(1-gal_match_probs)
            
    # plt.hist(all_df['all_pp'], bins=1000)
    # plt.xscale('log')
    # plt.show()
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
    
    # let's experiment with The Sean Approach
    # write a function for it
    if apple_bob == True:
        gtrue_pairs['prime_RA'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'RA'])
        gtrue_pairs['prime_DEC'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'DEC']) # <== only relevant for apple bobbing
        conv_apples(gtrue_pairs, all_df)
    
    
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
    gtrue_pairs.loc[ gtrue_pairs['i1prime_idx'] != -99, 'c1prime_z' ] = np.array(all_df.loc[ c1prime_no99, 'z' ])
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
    gtrue_pairs.loc[ gtrue_pairs['i2prime_idx'] != -99, 'c2prime_z' ] = np.array(all_df.loc[c2prime_no99, 'z' ])
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
    gtrue_pairs.loc[ gtrue_pairs['i1partner_idx'] != -99, 'c1partner_z' ] = np.array(all_df.loc[ c1partner_no99, 'z' ])
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
    gtrue_pairs.loc[ gtrue_pairs['i2partner_idx'] != -99, 'c2partner_z' ] = np.array(all_df.loc[ c2partner_no99, 'z' ])
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
    
    if save == True:
        gtrue_pairs.to_csv(conv_PATH+'conv_output/pair_ztype-'+z_type+'_'+field+'_'+date+'.csv', index=False)
        
    print('saved in ', field)
    
    return
    
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def load_pdfs(gal1, gal2, z1, z2, zt1, zt2, theta, field):
    print('beginning conv_prob() for ', field)
    
    all_prob = []
    all_PdA = []
    
    print(field, len(gal1))
    
    dA = np.linspace(0, 200, num=2001)
    # define array sizes to save distributions as
    PdA_2sav = np.zeros((len(gal1), len(dA)+2)) # so I can add the IDs and field as a check
    Pzz_2sav = np.zeros((len(gal1), 1001)) # length of z array
    
    # the COSMOS PDF are all loaded together, so do this outside for loop:
    if field == 'COSMOS':
        with fits.open(cPATH+'COSMOS2020_R1/PZ/COSMOS2020_CLASSIC_R1_v2.0_LEPHARE_PZ.fits') as data:
            # fix big endian buffer error:
            COSMOS_PZ_arr = np.array(data[0].data)
        COSMOS_PZ_arrf = COSMOS_PZ_arr.byteswap().newbyteorder()
        COSMOS_PZ = pd.DataFrame(COSMOS_PZ_arrf)
        z_01 = COSMOS_PZ.loc[0,1:].to_numpy()
        PDF_array = COSMOS_PZ.T
    else:
        with fits.open(PATH+'CANDELS_PDFs/'+field+'_mFDa4.fits') as data:
            CANDELS_PZ = pd.DataFrame(data[0].data)
        z_01 = CANDELS_PZ.loc[0,1:].to_numpy()
        PDF_array = CANDELS_PZ.T
    
    for i, (ID1, ID2, ID1_z, ID2_z, ID1_zt, ID2_zt, th) in tqdm(enumerate(zip(gal1, gal2, z1, z2, zt1, zt2, theta)), miniters=100): 

        if z_type != 'p':
            # if we are working with zspecs, we need to interpolate to a finer grid:
            z_fine = np.linspace(0,10,10001).round(3)
            if ID1_zt == 's':
                zspec1 = round(ID1_z, 3)
                PDF1 = np.zeros(z_fine.shape)
                PDF1[np.where(z_fine == zspec1)] = 1
                PDF1 = PDF1 / np.trapz(PDF1, x=z_fine) # normalize
            elif ID1_zt == 'p':
                # interpolate
                PDF1_01 = np.array(PDF_array.loc[ 1:, ID1 ])  ### BUG ###
                fintp1 = interp1d(z_01, PDF1_01, kind='linear')
                PDF1 = fintp1(z_fine)
                PDF1 = PDF1 / np.trapz(PDF1, x=z_fine)
            if ID2_zt == 's':
                zspec2 = round(ID2_z, 3)
                PDF2 = np.zeros(z_fine.shape)
                PDF2[np.where(z_fine == zspec2)] = 1
                PDF2 = PDF2 / np.trapz(PDF2, x=z_fine) # normalize
            elif ID2_zt == 'p':
                # interpolate
                PDF2_01 = np.array(PDF_array.loc[ 1:, ID2 ])
                fintp2 = interp1d(z_01, PDF2_01, kind='linear')
                PDF2 = fintp2(z_fine)
                PDF2 = PDF2 / np.trapz(PDF2, x=z_fine)
            
            Cv_prob = Convdif(z_fine, PDF1, PDF2, dv_lim=max_dv)
            PdA, comb_PDF = PdA_prob(PDF1, PDF2, ID1_zt, ID2_zt, th, z_fine, dA)
            
        else:
        
            PDF1 = np.array(PDF_array.loc[ 1:, ID1 ])
            PDF2 = np.array(PDF_array.loc[ 1:, ID2 ])
            Cv_prob = Convdif(z_01, PDF1, PDF2, dv_lim=max_dv)
            PdA, comb_PDF = PdA_prob(PDF1, PDF2, ID1_zt, ID2_zt, th, z_01, dA)
        
        
        all_prob.append(Cv_prob)
        # all_PdA.append(PdA)
        
#         ### WRITE THIS TO SPIT OUT ARRAYS FOR THE ENTIRE GROUP ###
        # add the IDs to the first two entries
        PdA_2sav[i,0] = ID1
        PdA_2sav[i,1] = ID2
        PdA_2sav[i,2:] = PdA
#         Pzz_2sav[i,:] = comb_PDF
        
    ### SAVE AS A FITS FILE ###
    # hdu_dA = fits.PrimaryHDU(PdA_2sav)
    # hdul_dA = fits.HDUList([hdu_dA])
    # hdul_dA.writeto(mPATH+'/TEST_PdA_'+field+'_8.09.fits', overwrite=True)
#     hdu_zz = fits.PrimaryHDU(Pzz_2sav)
#     hdul_zz = fits.HDUList([hdu_zz])
#     hdul_zz.writeto(mPATH+'/Pzz_'+field+'.fits', overwrite=True)
    
    # np.savetxt(mPATH+'/PdA_'+field+'.txt', PdA_2sav)
    # np.savetxt(mPATH+'/Pzz_'+field+'.txt', Pzz_2sav)
            
    return all_prob, PdA_2sav #####################

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
    if len(fintp2(v_new).shape) == 1:
        
        v_conv = signal.fftconvolve(fintp2(v_new), fintp1(v_new)[::-1], mode='full')
    else:
        v_conv = signal.fftconvolve(fintp2(v_new), fintp1(v_new)[:,::-1], mode='full', axes=1) ### THE BUG IS HERE ###
    
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
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    if z_type != 'p' and zt1 == 's' and zt2 == 's': # add if both z_types = 'spec' or something to avoid zero issues
        middle_z = np.mean( (z[np.argmax(PDF1)], z[np.argmax(PDF2)]) )
        # then make the corresponding comb_PDF value = 1
        comb_PDF[ np.argmin(np.abs(z-middle_z)) ] = 1  
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    comb_PDF = comb_PDF / np.trapz(comb_PDF, x=z) # and normalize
    
    # so split 0-1.61 (1) and 1.62-10 (2)
    dA1 = ang_diam_dist( z[np.where(z <= 1.61)], theta )
    dA2 = ang_diam_dist( z[np.where(z > 1.61)], theta )
    z1 = z[np.where(z <= 1.61)]
    z2 = z[np.where(z > 1.61)]
    comb_PDF1 = comb_PDF[np.where(z <= 1.61)] 
    comb_PDF2 = comb_PDF[np.where(z > 1.61)] 

    PdA11 = comb_PDF1 * np.abs(dzdA(dA1, z1, theta))
    PdA11 = np.nan_to_num(PdA11) # fill the nan casued by division ny zero
    PdA12 = comb_PDF2 * np.abs(dzdA(dA2, z2, theta))
    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    # concatenate 1 and 2 just to find the max
    dA_comb = np.concatenate((dA1, dA2))
    PdA_comb = np.concatenate((PdA11, PdA12))
    max_dA = dA_comb[np.argmax(PdA_comb)]
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    dA_new = dA
    fintp1 = interp1d(dA1, PdA11, kind='linear', bounds_error=False, fill_value=0)
    fintp2 = interp1d(dA2, PdA12, kind='linear', bounds_error=False, fill_value=0)

    intr_PdA1 = fintp1(dA_new)
    intr_PdA2 = fintp2(dA_new)

    PdA = intr_PdA1+intr_PdA2
    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    if z_type != 'p' and (zt1 == 's' or zt2 == 's'): # should only happen when there is a spec z
        # find the element in dA_new that is closest to max_dA
        PdA[ np.argmin(np.abs(dA_new-max_dA)) ] = np.max(PdA_comb)  
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    
    PdA = PdA / np.trapz(PdA, x=dA_new)
    
    return np.nan_to_num(PdA), comb_PDF


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

def conv_apples(pair_df, iso_pool_df, base_dz=0.05, base_dM=0.05, dP=0.01, N_controls=2):
    # so for each pair, assemble a df from the iso pool for both it's prime and partner properties
    iso_pool_df = iso_pool_df.loc[ iso_pool_df['all_pp'] <= min_pp ].reset_index(drop=True)
    
    field = pair_df['field'].unique()[0]
    if field == 'COSMOS':
        base_dz = 0.02
        base_dM = 0.02
    print('Bobbing for apples in {}'.format(field))
    milestones = np.linspace(100, 100000, 1000, dtype=int)
    
    # split COSMOS into quadrants... hopefully this will ease the convolution calculations:
    mid_iso_RA = np.median(iso_pool_df['RA'])
    mid_iso_DEC = np.median(iso_pool_df['DEC'])
    iso_pool_df['Quadrant'] = [0]*len(iso_pool_df)
    pair_df['Quadrant'] = [0]*len(pair_df)
        
    # load in the PDFs:
    if field == 'COSMOS':
        with fits.open(cPATH+'COSMOS2020_R1/PZ/COSMOS2020_CLASSIC_R1_v2.0_LEPHARE_PZ.fits') as data:
            # fix big endian buffer error:
            COSMOS_PZ_arr = np.array(data[0].data)
        COSMOS_PZ_arrf = COSMOS_PZ_arr.byteswap().newbyteorder()
        COSMOS_PZ = pd.DataFrame(COSMOS_PZ_arrf)
        z_01 = COSMOS_PZ.loc[0,1:].to_numpy()
        PDF_array = np.array(COSMOS_PZ)
        iso_pool_df.loc[ (iso_pool_df['RA'] >= mid_iso_RA) & (iso_pool_df['DEC'] >= mid_iso_DEC), 'Quadrant' ] = 1
        iso_pool_df.loc[ (iso_pool_df['RA'] >= mid_iso_RA) & (iso_pool_df['DEC'] < mid_iso_DEC), 'Quadrant' ] = 2
        iso_pool_df.loc[ (iso_pool_df['RA'] < mid_iso_RA) & (iso_pool_df['DEC'] >= mid_iso_DEC), 'Quadrant' ] = 3
        iso_pool_df.loc[ (iso_pool_df['RA'] < mid_iso_RA) & (iso_pool_df['DEC'] < mid_iso_DEC), 'Quadrant' ] = 4
        pair_df.loc[ (pair_df['prime_RA'] >= mid_iso_RA) & (pair_df['prime_DEC'] >= mid_iso_DEC), 'Quadrant'] = 1
        pair_df.loc[ (pair_df['prime_RA'] >= mid_iso_RA) & (pair_df['prime_DEC'] < mid_iso_DEC), 'Quadrant'] = 2
        pair_df.loc[ (pair_df['prime_RA'] < mid_iso_RA) & (pair_df['prime_DEC'] >= mid_iso_DEC), 'Quadrant'] = 3
        pair_df.loc[ (pair_df['prime_RA'] < mid_iso_RA) & (pair_df['prime_DEC'] < mid_iso_DEC), 'Quadrant'] = 4
    else:
        with fits.open(PATH+'CANDELS_PDFs/'+field+'_mFDa4.fits') as data:
            CANDELS_PZ_arr = np.array(data[0].data)
        CANDELS_PZ_arrf = CANDELS_PZ_arr.byteswap().newbyteorder()
        CANDELS_PZ = pd.DataFrame(CANDELS_PZ_arrf)
        z_01 = CANDELS_PZ.loc[0,1:].to_numpy()
        PDF_array = np.array(CANDELS_PZ)
    
    controls = []
    
    for i, (ID1, z1, M1, ID2, z2, M2, Pp, Qd) in enumerate(zip(pair_df['prime_ID'], pair_df['prime_z'], pair_df['prime_M'], 
                                  pair_df['partner_ID'], pair_df['partner_z'], pair_df['partner_M'], pair_df['pair_prob'],
                                  pair_df['Quadrant'])):
        got = False
        dz = base_dz
        dM = base_dM
        tried_arr = []
        while got == False:
            iso1 = iso_pool_df.loc[ (np.abs(iso_pool_df['z']-z1) < dz) & (np.abs(iso_pool_df['MASS']-M1) < dM) &
                                  (iso_pool_df['ID'] != ID1) & (iso_pool_df['ID'] != ID2) & (iso_pool_df['Quadrant'] == Qd), 
                                  ['ID','RA','DEC','MASS','z','LX','IR_AGN_DON','IR_AGN_STR'] ] 
            iso1 = iso1.rename(columns={'RA':'RA1', 'DEC':'DEC1', 'z':'z1', 'MASS':'MASS1', 
                                        'LX':'LX1', 'IR_AGN_DON':'IR_AGN_DON1', 'IR_AGN_STR':'IR_AGN_STR1'})
            iso2 = iso_pool_df.loc[ (np.abs(iso_pool_df['z']-z2) < dz) & (np.abs(iso_pool_df['MASS']-M2) < dM) &
                                  (iso_pool_df['ID'] != ID1) & (iso_pool_df['ID'] != ID2) & (iso_pool_df['Quadrant'] == Qd),
                                  ['ID','RA','DEC','MASS','z','LX','IR_AGN_DON','IR_AGN_STR'] ]
            iso2 = iso2.rename(columns={'RA':'RA2', 'DEC':'DEC2', 'z':'z2', 'MASS':'MASS2', 
                                        'LX':'LX2', 'IR_AGN_DON':'IR_AGN_DON2', 'IR_AGN_STR':'IR_AGN_STR2'})
            # somehow combine the iso1 and iso2 df into a M x N long df that has all possible combinations of them
            # useful to calculate conv_prob columns-wise
            apple_df = iso1.merge(iso2, how='cross').rename(columns={'ID_x':'ID1','ID_y':'ID2'})
            if len(apple_df) == 0:
                dz = dz + 0.03
                dM = dM + 0.03
                continue
                
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
                continue
            
            # make sure a separation cut is adequite:
            xCOR = SkyCoord(apple_df['RA1'], apple_df['DEC1'], unit='deg')
            yCOR = SkyCoord(apple_df['RA2'], apple_df['DEC2'], unit='deg')
            apple_df['arc_sep'] = xCOR.separation(yCOR).arcsecond

            apple_df['Cp'] = Convdif(z_01, PDF_array[apple_df['ID1'],1:], PDF_array[apple_df['ID2'],1:], dv_lim=max_dv)

            if Pp > 0.01:
                apple_df2 = apple_df.loc[ (np.abs(apple_df['Cp'] - Pp) < dP) & (apple_df['arc_sep'] > max_R_kpc) &
                                         (apple_df['ID1'] != apple_df['ID2']) ].reset_index(drop=True)
                apple_df.loc[ (np.abs(apple_df['Cp'] - Pp) < dP) & (apple_df['arc_sep'] > max_R_kpc) &
                                         (apple_df['ID1'] != apple_df['ID2']), 'reuse_flag' ] = 1
            else:                                                                          # probably too strict...
                apple_df2 = apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 0.5) & (apple_df['arc_sep'] > max_R_kpc) &
                                         (apple_df['ID1'] != apple_df['ID2']) ].reset_index(drop=True)
                apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < 0.5) & (apple_df['arc_sep'] > max_R_kpc) &
                                         (apple_df['ID1'] != apple_df['ID2']), 'reuse_flag' ] = 1
                
            # add pair information:
            apple_df2['P_ID1'] = [ID1]*len(apple_df2)
            apple_df2['P_ID2'] = [ID2]*len(apple_df2)
            apple_df2['Pp'] = [Pp]*len(apple_df2)
            
            # organize to get best fit:  ### may need to think how the Cp dif is made (log or not), doubt it'd matter tho ###
            apple_df2['dif'] = (10*(np.abs(np.log10(apple_df2['Cp']) - np.log10(Pp)))**2 + 
                                    (apple_df2['z1'] - z1)**2 + (apple_df2['MASS1'] - M1)**2 +
                                (apple_df2['z2'] - z2)**2 + (apple_df2['MASS2'] - M2)**2)
            # now sort on this
            apple_df2.sort_values(by=['dif'], inplace=True, ascending=True, ignore_index=True) # this resets the index
            
            # print(field, ID1, M1, z1, ID2, M2, z2, Pp, np.max(apple_df['Cp']), len(apple_df), len(apple_df2))
        
            # take the top pair and add it to an array:
            if len(apple_df2) >= N_controls:   ### ~~~ POSSIBLE THE TWO CHOSEN PAIRS HAVE OVERLAPPING GALAXIES ~~~ ###
                controls.append(apple_df2.iloc[:N_controls])
                # give me a progress message every 500 iterations
                if np.any(milestones==i) == True:
                    print('{0}/{1} controls selected for {2}'.format(i, len(pair_df), field))
                got = True
            else:
                # get an array of str IDs that have been tried already
                tried_arr = np.concatenate( (tried_arr, apple_df.loc[ apple_df['reuse_flag'] == 0, 'pair_str' ]) ) 
                dz = dz + 0.03
                dM = dM + 0.03
                if dz > 0.4:
                    print('dz exceeds 0.4', field, ID1, M1, z1, ID2, M2, z2, Pp, np.max(apple_df['Cp']), len(apple_df), len(apple_df2))
                # print('expanding search ===>', dz, dM)
                # COSMOS will still get congested... should print out when we get too far away mass/redshift wise...
                # could also tell the code to not recalculate conv prob for combinations already calculated and failed before #
                # may also work better without any kind of bottom pair prob cut
                
    # combine all control_dfs into one and save for this field:
    # controls is a list of a 2 index dfs, so need to combine these all\
    control_df = pd.DataFrame( np.concatenate(controls), columns=pd.DataFrame(controls[0]).columns )
    # add field information
    control_df['field'] = [field]*len(control_df)
    
    if save == True:
        # save the field df as a csv
        control_df.to_csv(conv_PATH+'control_output/control_N-'+str(N_controls)+'_ztype-'+z_type+'_'+field+'_'+date+'.csv', index=False)
        print('Control df saved in {}'.format(field))
        
        
        # print(len(iso1), len(iso2))
        # print(round(np.abs(z1-z2),2), len(apple_df)-len(apple_df2))
        
        ### ~~~ does it make sense that sometimes all matches are within the threshold... ~~~ ###
        ### ~~~ seems to be tied with redhsift difference <== what we'd expect 
        
        ### ~~~ need to make sure that when we pick these, we don't pick a source that matches to itself
        ### ~~~ AND we could also calulate an arcsecond sep column and make sure they are beyond ~ 25" or something ~~~ ###
        ### ~~~ ===> this would take care of duplicates, albiet adding some time to the code
        ### ~~~ through in a min pair prob for the whole field and that's gotta be the best you can do for this ~~~ ###
        
        
    # # just save this array
    # test_count = np.array(test_count)
    # np.savetxt(field+'_testcount.txt', test_count, delimiter=',')
               
    return
        

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