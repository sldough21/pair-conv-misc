# Sean Dougherty
# 03/11/2022
# a reorganized version of z_cuts.py to maximize efficiency

# import libraries
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' -> could change to None

import numpy as np
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

from numpy import random

import matplotlib.pyplot as plt

import sys

PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/Pair Project - Updated Data/'
cPATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/'

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
# determine conservative ang separation correspondong to 150 kpc at z = 0.5
R_kpc = cosmo.arcsec_per_kpc_proper(0.5) # arcsec/kpc at z=0.5
max_R_kpc = (150*u.kpc * R_kpc) # in arcseconds ### this is the bug right here

mass_lo = 9.0 # lower mass limit of the more massive galaxy in a pair that we want to consider
n = 10000 # number of draws
gamma = 1.4 # for k correction calculation

max_sep = 150 # kpc # 120
max_iso = 5000 # dv

z_type = 's'
preload = False
select_controls = True


# -------------------------------------------------------------------------------------------------------------------------- #


def main():
    
    print('beginning main()')
    
    # we want to parallelize the data by fields, so:
    all_fields = ['GDS','EGS','COS','GDN','UDS','COSMOS']
    
    
    # Create a multiprocessing Pool
    pool = Pool() 
    pool.map(process_samples, all_fields)

    # close pool
    pool.close()
    pool.join()
    
    print('done')
        
    
    # for now, run things without pooling -> easier to read errors
    # process_samples('GDS')
    
# -------------------------------------------------------------------------------------------------------------------------- #

    
def process_samples(field):
    # this is essentially the main function but for each field, to be combined and saved as csv's upon completion
    print('beginning process_samples() for {}'.format(field))
   
    if field == 'COSMOS':
        df = pd.read_csv(cPATH+'select_COSMOS2020.csv')
        df['SIG_DIFF'] = df['lp_zPDF_u68'] - df['lp_zPDF_l68']
        df = df[ (df['lp_type'] != 1) & (df['lp_type'] != -99) & (df['lp_mass_med'] > (mass_lo-1)) & 
           (df['FLAG_COMBINED'] == 0) ]
        df = df.rename(columns={'ALPHA_J2000':'RA', 'DELTA_J2000':'DEC', 'lp_mass_med':'MASS', 'lp_zPDF':'ZPHOT_PEAK',
                               'F0.5-10_2015':'FX', 'UVISTA_H_MAG_APER2':'HMAG'})
        if z_type != 'p':
            zspec = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/zspec_cats/COS/COSMOS_ZSPEC_CAT.csv')
            df['ZSPEC'] = zspec['G_ZSPEC']
            df.loc[ df['ZSPEC'] > 0, 'SIG_DIFF'] = 0
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

        # JUST FOCUS ON PHOTZs for now
        if z_type != 'p':
            zspec = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/zspec_cats/'+field+'/ALL_CANDELS_'+field+'_ZSPEC_CAT_wAIRD.csv')
            zspec.loc[zspec['ZSPEC_AIRD'] == zspec['ZBEST_AIRD'], 'G_ZSPEC'] = zspec['ZSPEC_AIRD'] # assuming this worked..
            df['ZSPEC'] = zspec['G_ZSPEC']
            df.loc[ df['ZSPEC'] > 0, 'SIG_DIFF'] = 0
            

    ##### SMALLER SAMPLE SIZE FOR TEST #####
    # df = df.iloc[0:200]
    
    # draw data for each galaxy and calculate Lx(z) and M(z)
    if z_type == 's':
        draw_df_z = df['ZSPEC'] # these won't at all matter...
        draw_df_LX = df['FX']
    else:
        draw_df_z, draw_df_LX = draw_z(df, field)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
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
    
    # loop through number of iterations:
    for it in range(n):
        print( 'CURRENT ITERATION - '+field, it )
        # don't return anything, just do it
        determine_pairs(it, df, draw_df_z.iloc[it], draw_df_LX.iloc[it], z_type, field)
        
        if z_type == 's':
            break
        
    return
    
# -------------------------------------------------------------------------------------------------------------------------- #

    
def draw_z(df, field):
    print('Running draw_z for {}'.format(field))
    
    # the COSMOS PDF are all loaded together, so do this outside for loop:
    if field == 'COSMOS':
        with fits.open(cPATH+'COSMOS2020_R1/PZ/COSMOS2020_CLASSIC_R1_v2.0_LEPHARE_PZ.fits') as data:
            # fix big endian buffer error:
            COSMOS_PZ_arr = np.array(data[0].data)
        COSMOS_PZ_arrf = COSMOS_PZ_arr.byteswap().newbyteorder()
        COSMOS_PZ = pd.DataFrame(COSMOS_PZ_arrf)
        z = COSMOS_PZ.loc[0,1:].to_numpy()
        COSMOS_PZ = COSMOS_PZ.T
    
    # initialize dictionary
    draw_z = {}
    draw_LX = {}
    
    for i in tqdm(range(0, len(df['ID'])), miniters=50): ### THIS BREAKS FOR GDN ### <-- no should be fine actually
        # load PDFs based on string ID
        ID_str = df.loc[i,'ID']
        if len(str(ID_str)) == 1: id_string = '0000'+str(ID_str)
        if len(str(ID_str)) == 2: id_string = '000'+str(ID_str)
        if len(str(ID_str)) == 3: id_string = '00'+str(ID_str)
        if len(str(ID_str)) == 4: id_string = '0'+str(ID_str)
        if len(str(ID_str)) == 5: id_string = str(ID_str)

        if field == "GDS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSS_ID'+id_string+'.pzd'
            pdf1_df = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                  'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            pdf1 = np.array(pdf1_df['mFDa4'])
            z = np.array(pdf1_df['z'])
        elif field == "EGS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/EGS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_EGS_ID'+id_string+'.pzd'
            pdf1_df = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                  'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            pdf1 = np.array(pdf1_df['mFDa4'])
            z = np.array(pdf1_df['z'])
        elif field == "GDN":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSN_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSN_ID'+id_string+'.pzd'
            pdf1_df = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                  'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            pdf1 = np.array(pdf1_df['mFDa4'])
            z = np.array(pdf1_df['z'])
        elif field == "COS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/COSMOS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_COSMOS_ID'+id_string+'.pzd'
            pdf1_df = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                  'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            pdf1 = np.array(pdf1_df['mFDa4'])
            z = np.array(pdf1_df['z'])
        elif field == "UDS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/UDS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_UDS_ID'+id_string+'.pzd' 
            pdf1_df = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                  'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
            pdf1 = np.array(pdf1_df['mFDa4'])
            z = np.array(pdf1_df['z'])
        elif field == 'COSMOS':
            pdf1 = np.array(COSMOS_PZ.loc[ 1:, ID_str ])
     
        draw1 = random.choice(z, size=n, p=(pdf1/np.sum(pdf1)))
        
        # calculate luminosity
        DL_mpc = cosmo.luminosity_distance(draw1) # in Mpc -> convert to cm
        DL = DL_mpc.to(u.cm) # distance in cm
        # calculate the k correction
        kz = (1+draw1)**(gamma-2)
        LXz = df.loc[i, 'FX'] * 4 * np.pi * (DL**2) * kz
        
        # add entry into dictionary
        draw_z['gal_'+str(ID_str)+'_z'] = draw1
        draw_LX['gal_'+str(ID_str)+'_LX'] = LXz
    
    # convert dictionary to dataframe with gal ID as columns and redshift selections are rows
    draw_df_z = pd.DataFrame.from_dict(draw_z)
    draw_df_LX = pd.DataFrame.from_dict(draw_LX)
    
    
    ### 0 #######################################################################################################
    # I want to save these draws for GDS and work through the rest of the code with them
    # draw_df_z.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/test_output/drawn_z.csv', index=False)
    # draw_df_M.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/test_output/drawn_M.csv', index=False)
    # draw_df_LX.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/test_output/drawn_LX.csv', index=False)
    # draw_df_IR_AGN_DON.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/test_output/drawn_DON.csv', index=False)
    # draw_df_IR_AGN_STR.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/test_output/drawn_STR.csv', index=False)
    # print('WRITTEN')
    # sys.exit()
    
    
    return draw_df_z, draw_df_LX


# -------------------------------------------------------------------------------------------------------------------------- #


def determine_pairs(it, all_df, current_zdraw_df, current_LXdraw_df, z_type, field):
    ### 1 #######################################################################################################
    if z_type == 'p':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe, but first check for consistent lengths:
        z_drawn = current_zdraw_df.to_numpy()
        LX_drawn = current_LXdraw_df.to_numpy()
        all_df['drawn_z'] = z_drawn
        all_df['drawn_LX'] = LX_drawn
        
    ### BUILD STRUCTURE FOR z_type = 'phot+spec_z' ###
    elif z_type == 'ps':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe
        z_drawn = current_zdraw_df.to_numpy()
        LX_drawn = current_LXdraw_df.to_numpy()
        all_df['drawn_z'] = z_drawn
        all_df['drawn_LX'] = LX_drawn
        # if there is a spec q on quality > 1, change drawn_z to spec_z
        ### WILL NEED TO THINK CAREFULLY ON HOW THIS EFFECTS DRAWN M AND LX ###
        all_df.loc[ (all_df['ZSPEC'] > 0), 'drawn_z'] = all_df['ZSPEC']   
        all_df.loc[ (all_df['ZSPEC'] > 0), 'drawn_LX'] = ( all_df['FX'] * 4 * np.pi *
                                                            ((cosmo.luminosity_distance(all_df['drawn_z']).to(u.cm))**2) * 
                                                            ((1+all_df['drawn_z'])**(gamma-2)) )
        
    elif z_type == 's':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe
        all_df['drawn_z'] = [-99]*len(all_df)
        all_df['drawn_LX'] = [-99]*len(all_df)
        # make drawn_z the spec z and throw out the rest
        all_df.loc[ (all_df['ZSPEC'] > 0), 'drawn_z'] = all_df['ZSPEC']
        all_df.loc[ (all_df['ZSPEC'] > 0), 'drawn_LX'] = ( all_df['FX'] * 4 * np.pi *       # integration error bc z=-99 #
                                                            ((cosmo.luminosity_distance(all_df['drawn_z']).to(u.cm))**2) * 
                                                            ((1+all_df['drawn_z'])**(gamma-2)) )
        all_df = all_df[ (all_df['ZSPEC'] > 0) ]
    
 
    # make definite redshift cut:
    all_df = all_df[ (all_df['drawn_z'] >= 0.5) & (all_df['drawn_z'] <= 3.0) ]
    
    # reset this index:    
    all_df = all_df.reset_index(drop=True) # this probably means that previous results are hooplaa
        
    # match catalogs:
    df_pos = SkyCoord(all_df['RA'],all_df['DEC'],unit='deg')
    idxc, idxcatalog, d2d, d3d = df_pos.search_around_sky(df_pos, max_R_kpc)
    # idxc is INDEX of the item being searched around
    # idxcatalog is INDEX of all galaxies within arcsec
    # d2d is the arcsec differece
    
    # place galaxy pairs into a df and get rid of duplicate pairs:
    matches = {'prime_index':idxc, 'partner_index':idxcatalog, 'arc_sep': d2d.arcsecond}
    match_df = pd.DataFrame(matches)
    
    pair_df = match_df[ (match_df['arc_sep'] != 0.00) ]
    # get rid of inverse row pairs with mass ratio -------------------------------------> CHANGED THIS, CHECK THAT OK
    pair_df['mass_ratio'] = (np.array(all_df.loc[pair_df['prime_index'], 'MASS']) - 
                             np.array(all_df.loc[pair_df['partner_index'],'MASS']) )
    
    ### 2 #######################################################################################################
    # confidently isolated galaxies only match to themselves, so get rid of ID's with other matches
    iso_df = match_df[ (match_df['arc_sep'] == 0.00) ]
    
    # let's change this to galaxy ID's
    iso_conf_id = np.array(iso_df['prime_index'])
    pair_ear_id = np.array(pair_df['prime_index'])
    mask_conf = np.isin(iso_conf_id, pair_ear_id, invert=True)
    iso_conf = iso_conf_id[mask_conf]
    
    ### 3 #######################################################################################################
    pair_df = pair_df[ (pair_df['mass_ratio'] >= 0) ] 
    
    sorted_idx_df = pd.DataFrame(np.sort((pair_df.loc[:,['prime_index','partner_index']]).values, axis=1), 
                                    columns=(pair_df.loc[:,['prime_index','partner_index']]).columns).drop_duplicates()
    pair_df = pair_df.reset_index(drop=True)
    pair_df = pair_df.iloc[sorted_idx_df.index]
            
    # Do the second bit of mass_ratio cut after iso gal selection so isolated sample doesn't include high mass ratio pairs
    pair_df = pair_df[ (pair_df['mass_ratio'] <= 1) ]
    
    if len(pair_df) == 0: # for spec z stuff
        print(field)
        true_pairs =  pair_df
        zspec_count = 0
        return true_pairs, zspec_count
    
    # calculate relative line of sight velocity
    pair_df['dv'] = ( (((np.array(all_df.loc[pair_df['prime_index'], 'drawn_z'])+1)**2 -1)/ 
                       ((np.array(all_df.loc[pair_df['prime_index'], 'drawn_z'])+1)**2 +1)) - 
                     (((np.array(all_df.loc[pair_df['partner_index'], 'drawn_z'])+1)**2 -1)/ 
                      ((np.array(all_df.loc[pair_df['partner_index'], 'drawn_z'])+1)**2 +1)) ) * 2.998e5
    
    ### 4 #######################################################################################################
    # calculate projected separation at z
    pair_df['kpc_sep'] = (pair_df['arc_sep']) / cosmo.arcsec_per_kpc_proper(all_df.loc[pair_df['prime_index'], 'drawn_z'])
        
    true_pairs = pair_df[ (pair_df['kpc_sep'] <= max_sep*u.kpc) & (abs(pair_df['dv']) <= 1000) ]
    if z_type == 's':
        print('{0} pairs in {1}, total spec-zs = {2}'.format(len(true_pairs), field, len(all_df.loc[all_df['ZSPEC'] > 0])))
    
    ### 5 #######################################################################################################
    true_pairs = true_pairs.iloc[ np.where( np.array(all_df.loc[ true_pairs['prime_index'], 'MASS' ] > mass_lo) == True ) ]
                          # watch iloc...
    
    if len(true_pairs) == 0: ###### HERE IS THE PROBLEM ######
        print(len(all_df.loc[all_df['ZSPEC'] > 0]))
        true_pairs =  true_pairs
        zspec_count = 0
        return true_pairs
        
    ### 6 #######################################################################################################

    # add galaxies that aren't pairs into the isolated sample:
    iso_add = (pair_df[ (abs(pair_df['dv']) > max_iso) | (pair_df['kpc_sep'] > max_sep) ])

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
        
    # select control galaxies from iso_df ---> needs to be fixed ----> need to make sure indices are right here...
    pair_mass = np.concatenate( (np.array(all_df.loc[true_pairs['prime_index'], 'MASS']), 
                                          np.array(all_df.loc[true_pairs['partner_index'], 'MASS'])), axis=0 )
    pair_z = np.concatenate( (np.array(all_df.loc[true_pairs['prime_index'], 'drawn_z']), 
                                          np.array(all_df.loc[true_pairs['partner_index'], 'drawn_z'])), axis=0 )
    pair_idx = np.concatenate( (np.array(true_pairs['prime_index']), np.array(true_pairs['partner_index'])), axis=0 )
    iso_mass = all_df.loc[all_iso, 'MASS']
    iso_z = all_df.loc[all_iso, 'drawn_z']
    iso_idx = all_iso
    
    if select_controls == True:
        # shuffle pair info to get rid of prime mass bias
        data_length = pair_idx.shape[0]
        # Here we create an array of shuffled indices
        shuf_order = np.arange(data_length)
        np.random.shuffle(shuf_order)

        shuf_idx = pair_idx[shuf_order] # Shuffle the original data
        shuf_mass = pair_mass[shuf_order]
        shuf_z = pair_z[shuf_order]

        # run controls function
        shuf_controls = get_control(iso_idx, iso_mass, iso_z, shuf_idx, shuf_mass, shuf_z)

        # Create an inverse of the shuffled index array (to reverse the shuffling operation, or to "unshuffle")
        unshuf_order = np.zeros_like(shuf_order)
        unshuf_order[shuf_order] = np.arange(data_length)

        unshuf_controls = shuf_controls[unshuf_order] # Unshuffle the shuffled data

        controls = unshuf_controls # just so the names remain the same
    else:
        controls = np.full((len(pair_idx), 2), -99)
        
    
    # # let's output the matched fraction based on c_flag
    # c_flag_all = np.concatenate(c_flag)
    # tp = len(c_flag_all)
    # tm = len(c_flag_all[ np.where(c_flag_all == 0) ])
    # tr = len(c_flag_all[ np.where(c_flag_all == 1) ])
    # tf = len(c_flag_all[ np.where(c_flag_all == 2) ])
    # print('UNIQUE MATCH FRACTION IN {} = {}'.format(field, tm/tp))
    # print('RECYCLED FRACTION IN {} = {}'.format(field, tr/tp))
    # print('MATCH FRACTION WITH RECYCLING IN {} = {}'.format(field, (tm+tr)/tp)) 
    # print('FAIL FRACTION IN {} = {}'.format(field, tf/tp)) 
        
    ### 8 #######################################################################################################
    
    middle_idx = len(controls)//2
    prime_controls = controls[:middle_idx]
    partner_controls = controls[middle_idx:]
    
    # add other important data to the dataframe ====> all the drawn values 
    # worry about logical position in the df later
    true_pairs['prime_ID'] = np.array(all_df.loc[ true_pairs['prime_index'], 'ID' ])
    true_pairs['prime_drawn_z'] = np.array(all_df.loc[true_pairs['prime_index'], 'drawn_z'])
    true_pairs['prime_M'] = np.array(all_df.loc[true_pairs['prime_index'], 'MASS'])
    true_pairs['prime_2sigma'] = np.array(all_df.loc[true_pairs['prime_index'], 'SIG_DIFF'])
    true_pairs['prime_drawn_LX'] = np.array(all_df.loc[true_pairs['prime_index'], 'drawn_LX'])
    true_pairs['prime_IR_AGN_DON'] = np.array(all_df.loc[true_pairs['prime_index'], 'IR_AGN_DON'])
    true_pairs['prime_IR_AGN_STR'] = np.array(all_df.loc[true_pairs['prime_index'], 'IR_AGN_STR'])
    
    true_pairs['partner_ID'] = np.array(all_df.loc[ true_pairs['partner_index'], 'ID' ])
    true_pairs['partner_drawn_z'] = np.array(all_df.loc[true_pairs['partner_index'], 'drawn_z'])
    true_pairs['partner_M'] = np.array(all_df.loc[true_pairs['partner_index'], 'MASS'])
    true_pairs['partner_2sigma'] = np.array(all_df.loc[true_pairs['partner_index'], 'SIG_DIFF'])
    true_pairs['partner_drawn_LX'] = np.array(all_df.loc[true_pairs['partner_index'], 'drawn_LX'])
    true_pairs['partner_IR_AGN_DON'] = np.array(all_df.loc[true_pairs['partner_index'], 'IR_AGN_DON'])
    true_pairs['partner_IR_AGN_STR'] = np.array(all_df.loc[true_pairs['partner_index'], 'IR_AGN_STR'])
    
    # prime galaxy control 1
    true_pairs['c1_prime_ID'] = np.array(all_df.loc[ prime_controls, 'ID' ])
    true_pairs['c1_prime_drawn_z'] = np.array(all_df.loc[ prime_controls, 'drawn_z' ])
    true_pairs['c1_prime_M'] = np.array(all_df.loc[ prime_controls, 'MASS' ])
    true_pairs['c1_prime_2sigma'] = np.array(all_df.loc[ prime_controls, 'SIG_DIFF' ])
    true_pairs['c1_prime_drawn_LX'] = np.array(all_df.loc[ prime_controls, 'drawn_LX' ]) 
    true_pairs['c1_prime_IR_AGN_DON'] = np.array(all_df.loc[ prime_controls, 'IR_AGN_DON' ]) 
    true_pairs['c1_prime_IR_AGN_STR'] = np.array(all_df.loc[ prime_controls, 'IR_AGN_STR' ]) 
    
    # partner galaxy control 1
    # prime galaxy control 1
    true_pairs['c1_partner_ID'] = np.array(all_df.loc[ partner_controls, 'ID' ])
    true_pairs['c1_partner_drawn_z'] = np.array(all_df.loc[ partner_controls, 'drawn_z' ])
    true_pairs['c1_partner_M'] = np.array(all_df.loc[ partner_controls, 'MASS' ])
    true_pairs['c1_partner_2sigma'] = np.array(all_df.loc[ partner_controls, 'SIG_DIFF' ])
    true_pairs['c1_partner_drawn_LX'] = np.array(all_df.loc[ partner_controls, 'drawn_LX' ]) 
    true_pairs['c1_partner_IR_AGN_DON'] = np.array(all_df.loc[ partner_controls, 'IR_AGN_DON' ]) 
    true_pairs['c1_partner_IR_AGN_STR'] = np.array(all_df.loc[ partner_controls, 'IR_AGN_STR' ]) 
    
    true_pairs['field'] = [field]*len(true_pairs)

    # just save them here...
    if z_type == 'p':
        true_pairs.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/MC/photz/'+field+'_'+str(it)+'.csv', index=False)
        print('SAVED {0} ITERATION {1}/{2}'.format(field, it, n))
    if z_type == 'ps':
        true_pairs.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/MC/photspecz/'+field+'_'+str(it)+'.csv', index=False)
    if z_type == 's':
        true_pairs.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/MC/specz/'+field+'_'+str(it)+'.csv', index=False)

    return
    
    
    
# -------------------------------------------------------------------------------------------------------------------------- #

def get_control(control_ID, control_mass, control_z, gal_ID, mass, redshift, N_control=1, zfactor=0.2, mfactor=2): 
    
    matches = [] # IDs to drop
    # assemble the isolated df
    iso_df = pd.DataFrame({'IDX': control_ID, 'z': control_z, 'mass': control_mass})
    
    for i, (z, m) in enumerate(zip(redshift, mass)):
        
        cmatch_df = iso_df

        # create columns for difference between z/mass control and pair z/m   # may want to weigh mass higher than z...
        cmatch_df['dif'] = (cmatch_df['z'] - z)**2 + (cmatch_df['mass'] - m)**2

        # need to sort dataframe based on two columns THEN continue
        cmatch_df.sort_values(by=['dif'], inplace = True, ascending = True)
        
        # top one is the match
        match = cmatch_df.iloc[0]
        matches.append(match['IDX'])
        
        iso_df = cmatch_df.drop( cmatch_df[ (cmatch_df['IDX'] == match['IDX']) ].index ).reset_index(drop=True)
        # SHOULD TRIPLE CHECK THIS WORKS
                
    return np.array(matches)

# -------------------------------------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    main()