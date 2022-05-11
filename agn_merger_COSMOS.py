# Sean Dougherty
# 04/12/2022
# agn_merger code for COSMOS, considering the way the COSMOS data is stored
# it is simpler to just copy into a new file and adjust as needed.

# import libraries
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' -> could change to None

import numpy as np
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
import math as m
from multiprocessing import Pool, freeze_support, RLock
import functools

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from numpy import random

import sys

PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/'

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
# determine conservative ang separation correspondong to 150 kpc at z = 0.5
R_kpc = cosmo.arcsec_per_kpc_proper(0.5) # arcsec/kpc at z=0.5
max_R_kpc = (150*u.kpc * R_kpc) # in arcseconds ### this is the bug right here

mass_lo = 8.5 # lower mass limit of the more massive galaxy in a pair that we want to consider
n = 500 # number of draws
gamma = 1.4 # for k correction calculation
max_sep = 150 # kpc
max_iso = 5000

z_type = 'p' # I know universal variables are bad but...

# load my matched COSMOS catalog:
df = pd.read_csv(PATH+'matched_COSMOS2020.csv')
df = df[ (df['lp_type'] != 1) & (df['lp_type'] != -99) & (df['lp_mass_med'] > (mass_lo-1)) & 
           (df['FLAG_COMBINED'] == 0)]

# print('CHECK')
# print(df.columns[8]) # could be reason matches fail...

# good indices: get the rows from the drawn data csv based on df row selection
gidx = df.index
df = df.reset_index(drop=True)

### NOTE ###
# the drawn data has no cuts, so I will need to cut what gets loaded in below -> use same cuts as above
### NOTE ###


def determine_pairs(itr):
    ### 1 #######################################################################################################
    print('CURRENT ITERATION: {}'.format(itr))
    all_df = df
    if z_type == 'p':
        # if we are choosing just photo-z's, stick with the draws
        # bc we are trying to parallelize this, read in one specific column at a time to ave space
        z_df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/draw_df_z.csv', usecols=[itr], dtype=float)
        z_drawn = np.array(z_df[str(itr)])
        z_M = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/draw_df_M.csv', dtype=float)
        M_drawn = np.array(z_M['0'])
        LX_df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/draw_df_LX.csv', usecols=[itr], dtype=float)
        LX_drawn = np.nan_to_num(np.array(LX_df[str(itr)]))   
        #print('checking length of drawn z list')
        all_df['drawn_z'] = z_drawn[gidx]
        all_df['drawn_M'] = M_drawn[gidx]
        all_df['drawn_LX'] = LX_drawn[gidx]
        
    ### BUILD STRUCTURE FOR z_type = 'phot+spec_z' ###
    elif z_type == 'ps':
        # bc we are trying to parallelize this, read in one specific column at a time to ave space
        z_df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/draw_df_z.csv', usecols=[itr], dtype=float)
        z_drawn = np.array(z_df[str(itr)])
        z_M = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/draw_df_M.csv', dtype=float)
        M_drawn = np.array(z_M['0'])
        LX_df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/draw_df_LX.csv', usecols=[itr], dtype=float)
        LX_drawn = np.nan_to_num(np.array(LX_df[str(itr)]))        
        all_df['drawn_z'] = z_drawn[gidx]
        all_df['drawn_M'] = M_drawn[gidx]
        all_df['drawn_LX'] = LX_drawn[gidx]
        # if there is a spec q on quality > 1, change drawn_z to spec_z
        ### WILL NEED TO THINK CAREFULLY ON HOW THIS EFFECTS DRAWN M AND LX ###
        all_df.loc[ (all_df['q_zspec'] > 1) , 'drawn_z'] = all_df['zspec']   
        all_df.loc[ (all_df['q_zspec'] > 1) , 'drawn_LX'] = ( all_df['F0.5-10_2015'] * 4 * np.pi *
                                                            ((cosmo.luminosity_distance(all_df['drawn_z']).to(u.cm))**2) * 
                                                            ((1+all_df['drawn_z'])**(gamma-2)) )
        
    elif z_type == 's':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe
        z_df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/draw_df_z.csv', usecols=[itr], dtype=float)
        z_drawn = np.array(z_df[str(itr)])
        z_M = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/draw_df_M.csv', dtype=float)
        M_drawn = np.array(z_M['0'])
        LX_df = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/draw_df_LX.csv', usecols=[itr], dtype=float)
        LX_drawn = np.nan_to_num(np.array(LX_df[str(itr)]))  
        all_df['drawn_z'] = z_drawn[gidx]
        all_df['drawn_M'] = M_drawn[gidx]
        all_df['drawn_LX'] = LX_drawn[gidx]
        # make drawn_z the spec z and throw out the rest
        all_df.loc[ (all_df['q_zspec'] > 1) , 'drawn_z'] = all_df['zspec']   
        all_df.loc[ (all_df['q_zspec'] > 1) , 'drawn_LX'] = ( all_df['F0.5-10_2015'] * 4 * np.pi *
                                                            ((cosmo.luminosity_distance(all_df['drawn_z']).to(u.cm))**2) * 
                                                            ((1+all_df['drawn_z'])**(gamma-2)) )
        all_df = all_df[ (all_df['q_zspec'] > 1) ]
        
        
    # print('running iteration {}'.format(itr))
        
    ### CHECK THAT THE SPEC Z CUT WORKED ###
    # print(all_df['zspec'], all_df['q_zspec'], all_df['drawn_z'])
    
    # find out how many zspecs we actually end up using
    # print('number of spec z: ', all_df.loc[ (all_df['zspec'] > 0.5) & (all_df['q_zspec'] > 1) , 'drawn_z'].count())
    
    # make definite redshift cut:
    all_df = all_df[ (all_df['drawn_z'] >= 0.5) & (all_df['drawn_z'] <= 3.0) ]
    # reset this index:
    all_df = all_df.reset_index(drop=True) # this probably means that previous results are hooplaa
    # keep track of how many spec-z's were used each time
    zspec_count = len(all_df.loc[ all_df['q_zspec'] > 1 ])

        
    # match catalogs:
    df_pos = SkyCoord(all_df['ALPHA_J2000'],all_df['DELTA_J2000'],unit='deg')
    idxc, idxcatalog, d2d, d3d = df_pos.search_around_sky(df_pos, max_R_kpc)
    # idxc is INDEX of the item being searched around
    # idxcatalog is INDEX of all galaxies within arcsec
    # d2d is the arcsec differece
    
    # place galaxy pairs into a df and get rid of duplicate pairs:
    matches = {'prime_index':idxc, 'partner_index':idxcatalog, 'arc_sep': d2d.arcsecond}
    match_df = pd.DataFrame(matches)
    
    pair_df = match_df[ (match_df['arc_sep'] != 0.00) ]
    # get rid of inverse row pairs with mass ratio -------------------------------------> CHANGED THIS, CHECK THAT OK
    pair_df['mass_ratio'] = (np.array(all_df.loc[pair_df['prime_index'], 'drawn_M']) - 
                             np.array(all_df.loc[pair_df['partner_index'],'drawn_M']) )
    
    ### 2 #######################################################################################################
    # confidently isolated galaxies only match to themselves, so get rid of ID's with other matches
    # print('CHECK IF THERE ARE ANY CONFIDENT MATCHES')
    # print(len(pd.unique(pair_df['prime_index'])))
    
    iso_df = match_df[ (match_df['arc_sep'] == 0.00) ]
    
    # let's change this to galaxy ID's
    iso_conf_id = np.array(iso_df['prime_index'])
    pair_ear_id = np.array(pair_df['prime_index'])
    mask_conf = np.isin(iso_conf_id, pair_ear_id, invert=True)
    iso_conf = iso_conf_id[mask_conf]
    
    ### 3 #######################################################################################################
    pair_df = pair_df[ (pair_df['mass_ratio'] >= 0) ] ### BE CAREFUL WITH ISO SELECTION
    
    ######## ways to fix this duplicate issue -> separate out cases with mass_ratio = exactly 0 then...
    # potential solution
    sorted_idx_df = pd.DataFrame(np.sort((pair_df.loc[:,['prime_index','partner_index']]).values, axis=1), 
                                    columns=(pair_df.loc[:,['prime_index','partner_index']]).columns).drop_duplicates()
    pair_df = pair_df.reset_index(drop=True)
    pair_df = pair_df.iloc[sorted_idx_df.index]
    ######## - works, but may consider moving this later in the function to decrease data size in this computation
        
    # Do the second bit of mass_ratio cut after iso gal selection so isolated sample doesn't include high mass ratio pairs
    pair_df = pair_df[ (pair_df['mass_ratio'] <= 1) ] ### BE CAREFUL WITH ISO SELECTION
    
    if len(pair_df) == 0:
        print(field)
        return
    
    # calculate relative line of sight velocity
    pair_df['dv'] = ( (((np.array(all_df.loc[pair_df['prime_index'], 'drawn_z'])+1)**2 -1)/ 
                       ((np.array(all_df.loc[pair_df['prime_index'], 'drawn_z'])+1)**2 +1)) - 
                     (((np.array(all_df.loc[pair_df['partner_index'], 'drawn_z'])+1)**2 -1)/ 
                      ((np.array(all_df.loc[pair_df['partner_index'], 'drawn_z'])+1)**2 +1)) ) * 2.998e5
    
    ### 4 #######################################################################################################
    # calculate projected separation at z
    #R_kpc = cosmo.arcsec_per_kpc_proper(np.array((all_df.iloc[pair_df['prime_index']])['drawn_z'])
    pair_df['kpc_sep'] = (pair_df['arc_sep']) / cosmo.arcsec_per_kpc_proper(all_df.loc[pair_df['prime_index'], 'drawn_z'])
        
    #print('before true pairs:', len(pair_df))
    true_pairs = pair_df[ (pair_df['kpc_sep'] <= max_sep*u.kpc) & (abs(pair_df['dv']) <= 1000) ]
    
    ### ADD MASS LIMIT CUT NOW TO TRUE PAIRS, BC WE WILL NEED TO MATCH THE SMALLER GALS IN OUR ISO GROUP
    # n, bins, patches = plt.hist(all_df.loc[ true_pairs['prime_index'], 'drawn_M' ], bins=50, histtype='step')
    # used np array function to resolve duplicate index issues in pandas
    
    ### 5 #######################################################################################################
    true_pairs = true_pairs.iloc[ np.where( np.array(all_df.loc[ true_pairs['prime_index'], 'drawn_M' ] > mass_lo) == True ) ]
    
    
    # plt.hist(all_df.loc[ true_pairs['prime_index'], 'drawn_M' ], bins=bins, histtype='step')
    # plt.hist(all_df.loc[ true_pairs['partner_index'], 'drawn_M' ], bins=bins, histtype='step')
    # plt.show()
    # histograms confirm the mass cut on prime galaxies was successful
    ### -----> revisit this syntax later...
        
    ### 6 #######################################################################################################
    # add galaxies that aren't pairs into the isolated sample:
    #iso_add = (pair_df[ (pair_df['kpc_sep'] > 100*u.kpc) | (abs(pair_df['dv']) > 10000) ])
    iso_add = (pair_df[ (abs(pair_df['dv']) > max_iso) ])

    # just stack prime and partner indices into massive array:
    iso_add_idx = np.concatenate( (np.array(iso_add['prime_index']), np.array(iso_add['partner_index'])), axis=0)
    # return unique indices
    iso_add_uniq = np.unique(iso_add_idx)
    # get rid of cases where those indices appear elsewhere, so create array for true pair indices
    true_pair_idx = np.concatenate( (np.array(true_pairs['prime_index']), np.array(true_pairs['partner_index'])), axis=0)
    # only keep the elements that aren't in true pair:
    mask = np.isin(iso_add_uniq, true_pair_idx, invert=True)
    iso_unq = iso_add_uniq[mask]
    
    
    ### 7 #######################################################################################################
    # Collect all iso ID's:
    all_iso = np.concatenate( (iso_conf, iso_unq), axis=0)
    
    ### perhaps I should just create an array for pair and iso values rather than tracing them back each time?
    
    # select control galaxies from iso_df ---> needs to be fixed ----> need to make sure indices are right here...
    pair_mass = np.concatenate( (np.array(all_df.loc[true_pairs['prime_index'], 'drawn_M']), 
                                          np.array(all_df.loc[true_pairs['partner_index'], 'drawn_M'])), axis=0 )
    pair_z = np.concatenate( (np.array(all_df.loc[true_pairs['prime_index'], 'drawn_z']), 
                                          np.array(all_df.loc[true_pairs['partner_index'], 'drawn_z'])), axis=0 )
    iso_mass = all_df.loc[all_iso, 'drawn_M']
    iso_z = all_df.loc[all_iso, 'drawn_z']
    
    # print('HEREEEE')
    # print(iso_conf)
    # print(iso_unq)
    # print(all_iso)
    # pair_idx = np.concatenate( (np.array(true_pairs['prime_index']), np.array(true_pairs['partner_index'])), axis=0)
    # iso_idx = all_iso
    # for idx in pair_idx:
    #     if idx in iso_idx:
    #         print('TRUEE DAT')
    # print('END')
        
    controls = get_control(iso_mass, iso_z, pair_mass, pair_z)
        
    ### 8 #######################################################################################################
    iso_idx = all_iso
    
    middle_idx = len(controls)//2
    prime_controls = controls[:middle_idx]
    partner_controls = controls[middle_idx:]
            
    # add prime control galaxies to true_pair df ----> reminder these are indices of all_df (CHECK -> pretty sure)
    # problem that nans won't index
    # true_pairs['prime_control_idx1'] = iso_idx[(prime_controls[:,0]).tolist()]
    # true_pairs['prime_control_idx2'] = iso_idx[(prime_controls[:,1]).tolist()]
    # true_pairs['partner_control_idx1'] = iso_idx[(partner_controls[:,0]).tolist()]
    # true_pairs['partner_control_idx2'] = iso_idx[(partner_controls[:,1]).tolist()]
    
    # add other important data to the dataframe ====> all the drawn values 
    # worry about logical position in the df later
    true_pairs['prime_drawn_z'] = np.array(all_df.loc[true_pairs['prime_index'], 'drawn_z'])
    true_pairs['prime_drawn_M'] = np.array(all_df.loc[true_pairs['prime_index'], 'drawn_M'])
    true_pairs['prime_drawn_LX'] = np.array(all_df.loc[true_pairs['prime_index'], 'drawn_LX'])
    
    true_pairs['partner_drawn_z'] = np.array(all_df.loc[true_pairs['partner_index'], 'drawn_z'])
    true_pairs['partner_drawn_M'] = np.array(all_df.loc[true_pairs['partner_index'], 'drawn_M'])
    true_pairs['partner_drawn_LX'] = np.array(all_df.loc[true_pairs['partner_index'], 'drawn_LX'])
    
    
    # Dealing with missing data (ie 'nan')
    # must be a better way to do this without a for loop...
    idx11_id=[]
    idx11_cid=[]
    idx11_z=[]
    idx11_M=[]
    idx11_LX=[]
    idx12_id=[]
    idx12_cid=[]
    idx12_z=[]
    idx12_M=[]
    idx12_LX=[]
    idx21_id=[]
    idx21_cid=[]
    idx21_z=[]
    idx21_M=[]
    idx21_LX=[]
    idx22_id=[]
    idx22_cid=[]
    idx22_z=[]
    idx22_M=[]
    idx22_LX=[]
    
    for idx11, idx12, idx21, idx22 in zip((prime_controls[:,0]), (prime_controls[:,1]), 
                                         (partner_controls[:,0]), (partner_controls[:,1])):
        if np.isnan(idx11) == True:
            idx11_id.append( np.nan )
            idx11_cid.append( np.nan )
            idx11_z.append( np.nan )
            idx11_M.append( np.nan )
            idx11_LX.append( np.nan )
        else:
            idx11_id.append( iso_idx[idx11] )
            idx11_cid.append( all_df.loc[iso_idx[idx11], 'ID'] )
            idx11_z.append( all_df.loc[iso_idx[idx11], 'drawn_z'] )
            idx11_M.append( all_df.loc[iso_idx[idx11], 'drawn_M'] )
            idx11_LX.append( all_df.loc[iso_idx[idx11], 'drawn_LX'] )
            
        if np.isnan(idx12) == True:
            idx12_id.append( np.nan )
            idx12_cid.append( np.nan )
            idx12_z.append( np.nan )
            idx12_M.append( np.nan )
            idx12_LX.append( np.nan )
        else:
            idx12_id.append( iso_idx[idx12] )
            idx12_cid.append( all_df.loc[iso_idx[idx12], 'ID'] )
            idx12_z.append( all_df.loc[iso_idx[idx12], 'drawn_z'] )
            idx12_M.append( all_df.loc[iso_idx[idx12], 'drawn_M'] )
            idx12_LX.append( all_df.loc[iso_idx[idx12], 'drawn_LX'] )
            
        if np.isnan(idx21) == True:
            idx21_id.append( np.nan )
            idx21_cid.append( np.nan )
            idx21_z.append( np.nan )
            idx21_M.append( np.nan )
            idx21_LX.append( np.nan )
        else:
            idx21_id.append( iso_idx[idx21] )
            idx21_cid.append( all_df.loc[iso_idx[idx21], 'ID'] )
            idx21_z.append( all_df.loc[iso_idx[idx21], 'drawn_z'] )
            idx21_M.append( all_df.loc[iso_idx[idx21], 'drawn_M'] )
            idx21_LX.append( all_df.loc[iso_idx[idx21], 'drawn_LX'] )
            
        if np.isnan(idx22) == True:
            idx22_id.append( np.nan )
            idx22_cid.append( np.nan )
            idx22_z.append( np.nan )
            idx22_M.append( np.nan )
            idx22_LX.append( np.nan )
        else:
            idx22_id.append( iso_idx[idx22] )
            idx22_cid.append( all_df.loc[iso_idx[idx22], 'ID'] )
            idx22_z.append( all_df.loc[iso_idx[idx22], 'drawn_z'] )
            idx22_M.append( all_df.loc[iso_idx[idx22], 'drawn_M'] )
            idx22_LX.append( all_df.loc[iso_idx[idx22], 'drawn_LX'] )
                        
    true_pairs['prime_control1_ID'] = idx11_id      
    true_pairs['prime_control1_drawn_z'] = idx11_z
    true_pairs['prime_control1_drawn_M'] = idx11_M
    true_pairs['prime_control1_drawn_LX'] = idx11_LX
    
    true_pairs['prime_control2_ID'] = idx12_id 
    true_pairs['prime_control2_drawn_z'] = idx12_z
    true_pairs['prime_control2_drawn_M'] = idx12_M
    true_pairs['prime_control2_drawn_LX'] = idx12_LX
    
    true_pairs['partner_control1_ID'] = idx21_id 
    true_pairs['partner_control1_drawn_z'] = idx21_z
    true_pairs['partner_control1_drawn_M'] = idx21_M
    true_pairs['partner_control1_drawn_LX'] = idx21_LX
    
    true_pairs['partner_control2_ID'] = idx22_id 
    true_pairs['partner_control2_drawn_z'] = idx22_z
    true_pairs['partner_control2_drawn_M'] = idx22_M
    true_pairs['partner_control2_drawn_LX'] = idx22_LX    
    
    # plot histograms to see that distribution of mass and z is the same for pairs and samples:
#     histp_z = np.concatenate( (np.array(true_pairs['prime_drawn_z']), np.array(true_pairs['partner_drawn_z'])), axis=0 )
#     histp_M = np.concatenate( (np.array(true_pairs['prime_drawn_M']), np.array(true_pairs['partner_drawn_M'])), axis=0 )
    
#     histc_z = np.concatenate( (np.array(idx11_z), np.array(idx12_z), np.array(idx21_z), np.array(idx22_z)), axis=0 )
#     histc_M = np.concatenate( (np.array(idx11_M), np.array(idx12_M), np.array(idx21_M), np.array(idx22_M)), axis=0 )
    
    # histp_M = np.array(true_pairs['prime_drawn_M'])
    # histc_M = np.concatenate( (np.array(idx11_M), np.array(idx12_M) ), axis=0)#, np.array(idx21_M), np.array(idx22_M)), axis=0 )
    
    # histp_M = np.array(true_pairs['partner_drawn_M'])
    # histc_M = np.concatenate( (np.array(idx21_M), np.array(idx22_M) ), axis=0)#, np.array(idx21_M), np.array(idx22_M)), axis=0 )

    # # may not be accurate until we have a better mass cut -> better pair to isolated sample
    # plt.hist(histp_z, bins=50, density=True, histtype='step')
    # plt.hist(histc_z, bins=50, density=True, histtype='step')
    # plt.title(field)
    # seems that there just aren't as many galaxies to match at high z, so control sample is currently biased towarads low z
    # mass seems to be pretty tight tho
    # potential solution is to bin then pick controls per bin, to all some duplicates in independent bins
    # plt.show()
    
    # # test if there are du[licates here
    # post_all_pair = np.concatenate( (np.array(true_pairs.loc[:, 'prime_index']), 
    #                                                            np.array(true_pairs.loc[:, 'partner_index'])), axis=0 )
    # post_all_con = np.concatenate( (np.array(true_pairs.loc[:, 'prime_control1_ID']), 
    #                                                            np.array(true_pairs.loc[:, 'prime_control2_ID']),
    #                                                        np.array(true_pairs.loc[:, 'partner_control1_ID']),
    #                                                        np.array(true_pairs.loc[:, 'partner_control2_ID'])), axis=0 )
    # for idx in post_all_pair:
    #     if idx in post_all_con:
    #         print('TRUEE DAT')
    # print('THEY ARE SEPARATE')
    # # print(post_all_pair)
    # print(post_all_con)
    true_pairs['field'] = ['COSMOS']*len(true_pairs)
    true_pairs['prime_cat_ID'] = np.array(all_df.loc[ true_pairs['prime_index'], 'ID' ])
    true_pairs['partner_cat_ID'] = np.array(all_df.loc[ true_pairs['partner_index'], 'ID' ])
    
    true_pairs['prime_control1_cat_ID'] = idx11_cid
    true_pairs['prime_control2_cat_ID'] = idx12_cid
    true_pairs['partner_control1_cat_ID'] = idx21_cid
    true_pairs['partner_control2_cat_ID'] = idx22_cid


    if z_type == 'p':
        true_pairs.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/COSMOS/photoz/kpc'+str(max_sep)+'/'+str(itr)+'.csv',
                            index=False)
        print('saved no. {}'.format(itr))
    
    if z_type == 'ps':
        true_pairs.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/COSMOS/photo-specz/kpc'+str(max_sep)+'/'+str(itr)+'.csv',
                           index=False)
        print('saved no. {}'.format(itr))

    if z_type == 's':
        print('SAVED')
        true_pairs.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/COSMOS/specz/kpc'+str(max_sep)+'/'+str(itr)+'.csv',
                           index=False)
    
    
    
# -------------------------------------------------------------------------------------------------------------------------- #

def get_control(control_mass, control_z, mass, redshift, N_control=2, zfactor=0.2, mfactor=2): 

    dz = zfactor
    
    ### WILL NEED TO INCLUDE OVERDENSITY CODE ###
    ### WILL BE VERY TIME EXPENSIVE MOST LIKELY ###
    
    ### TRY SHUFFLING PAIR ARRAYS ### ADD DISTANCE METRIC ###
    # added combinatory distance metric, will need to play around with whether I want hard z and mass control limits...

    # create list to store lists of control indices per prime galaxy
    control_all = []
        
    # create a list for all ID's to make sure there are no duplicates
    control_dup = []
    
    # create a dataframe from the isolated galaxy data
    iso = {'z':control_z, 'mass':control_mass}
    all_iso_df = pd.DataFrame( iso )
    # somehow indices were carried with these values, so if we want to index a list of indices we want:
    all_iso_df = all_iso_df.reset_index(drop=True)
    
    for i, (m, z) in enumerate(zip(mass, redshift)):
        
        control = []

        zmin = z - dz
        zmax = z + dz
        mmin = m-np.log10(mfactor)
        mmax = m+np.log10(mfactor)

        # create a dataframe for possible matches
        cmatch_df = all_iso_df[ (all_iso_df['z'] >= zmin) & (all_iso_df['z'] <= zmax) & (all_iso_df['mass'] >= mmin) &
                               (all_iso_df['mass'] <= mmax) ]
        
        # create columns for difference between z/mass control and pair z/m
        cmatch_df['dif'] = (cmatch_df['z'] - z)**2 + (cmatch_df['mass'] - m) **2
                
        # need to sort dataframe based on two columns THEN continue
        cmatch_df.sort_values(by=['dif'], inplace=True, ascending = True)

        # immediately get rid of control galaxies that have already been selected
        cmatch_df = cmatch_df[ ((cmatch_df.index).isin(control_dup) == False) ]
        
        mcount = 0

        for idx in cmatch_df.index:

            control.append(idx)
            control_dup.append(idx)
            mcount+=1

            if mcount == N_control: 
                break

        if mcount < N_control:
            #print('Not enough control galaxies for object {}!'.format(i))
            while len(control) < N_control:
                control.append(np.nan)

        control_all.append(control)

    # return as an array
    return np.asarray(control_all, dtype=object)


# -------------------------------------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    # create a dictionary to store dfs per each iteration
    field_dict = {}
    
    # create a list to store spec_z counts and a dictionary to store averages
    N_zspec_all = []
    N_zspec = {}
    
    ### Create a multiprocessing Pool ###
    if z_type == 's':
        determine_pairs(itr=0)
    else:
        print('creating pool')
        pool = Pool(8)  

        print('running determine_pairs()')
        pool.map(determine_pairs, range(0, n))

        print('files written!')

        # close pool
        pool.close()
        pool.join()

    #field_dict[str(it)] = results # <---- gotta keep this consistent as well
    
