# Sean Dougherty
# 03/11/2022
# a reorganized version of z_cuts.py to maximize efficiency

# import libraries
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' -> could change to None

import numpy as np
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

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
# determine conservative ang separation correspondong to 150 kpc at z = 0.5
R_kpc = cosmo.arcsec_per_kpc_proper(0.5) # arcsec/kpc at z=0.5
max_R_kpc = (150*u.kpc * R_kpc) # in arcseconds ### this is the bug right here

mass_lo = 8.5 # lower mass limit of the more massive galaxy in a pair that we want to consider
n = 500 # number of draws
gamma = 1.4 # for k correction calculation

max_sep = 150 # kpc
max_iso = 5000 # dv

z_type = 's'


# -------------------------------------------------------------------------------------------------------------------------- #


def main():
    
    print('beginning main()')
    
    # we want to parallelize the data by fields, so:
    all_fields = ['GDS','EGS','COS','GDN','UDS']
    
    
    # Create a multiprocessing Pool
    pool = Pool() 
    all_data = pool.map(process_samples, all_fields)

    ### ERROR FOR SPEC Z IS IN RETURNING DATA TO ALL DATA ###
    

    # close pool
    pool.close()
    pool.join()

    print('Q_ZSPEC > 1 before')
    print('average number of spec-z used in each iteration by field:')
    print('GDS: {}'.format(all_data[0][1]))
    print('EGS: {}'.format(all_data[1][1]))
    print('COS: {}'.format(all_data[2][1]))
    print('GDN: {}'.format(all_data[3][1]))
    print('UDS: {}'.format(all_data[4][1]))


    # endf = pd.DataFrame(all_data[0]['0'])
    # endf.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/photoz_TEST.csv')

    # combine dfs for each iteration and save them as a csv file:
    # all_data will be 5 dictionaries... test:
    GDS_dict = all_data[0][0]
    EGS_dict = all_data[1][0]
    COS_dict = all_data[2][0]
    GDN_dict = all_data[3][0]
    UDS_dict = all_data[4][0]
    
    for it in GDS_dict:
        combined_df = pd.concat([GDS_dict[it], EGS_dict[it], COS_dict[it], GDN_dict[it], UDS_dict[it]])

        if z_type == 'p':
            combined_df.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/photoz_results/kpc'+str(max_sep)+'/'+str(it)+'.csv', index=False)
        if z_type == 'ps':
            combined_df.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/photo-specz_results/kpc'+str(max_sep)+'/'+str(it)+'.csv', index=False)
        if z_type == 's':
            combined_df.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/specz_results/kpc'+str(max_sep)+'/'+str(it)+'.csv', index=False)

    print('files written!')
        
    
    # for now, run things without pooling -> easier to read errors
    # process_samples('GDS')
    
# -------------------------------------------------------------------------------------------------------------------------- #

    
def process_samples(field):
    # this is essentially the main function but for each field, to be combined and saved as csv's upon completion
    print('beginning process_samples() for {}'.format(field))

#     # load data 
#     df_1 = pd.read_csv(PATH+'CANDELS_'+field+'_1018_LX.csv')
#     df_2 = pd.read_csv(PATH+'CANDELS_'+field+'_1018_LX_z.csv')
#     df_3 = pd.read_csv(PATH+'zcat_'+field+'_v2.0.csv') # need to ask Dale where this data is from !!! # are the zhi/lo accurate tho...
    
#     df = df_1.join(df_2).join(df_3)
#     #print(df['zbest'], df['zbest2']) # these are differet, though the ID's are correct
    
#     # check that IDs are consistent then drop IDs
#     df = df.drop(['id2'], axis=1)
    
    
    # load in new catalogs
    # with fits.open(PATH+'CANDELS_Catalogs/CANDELS.'+field+'.1018.Lx_best.wFx.fits') as data:
    #     df_data = np.array(data[1].data)
    # # to fix endian error reading fits file
    # df_fix = df_data.byteswap().newbyteorder()
    # df = pd.DataFrame(df_fix)
    
    # load in new catalogs w/Aird
    df = pd.read_csv(PATH+'CANDELS_Catalogs/CANDELS.'+field+'.1018.Lx_best.wFx_AIRD.csv')

    # throw away galaxies with PDF confidence intervals beyond redshift range
    #df = df.drop(df[ (df['zlo'] > 3.5) | (df['zhi'] < 0.25) ].index)
    # make additional quality cuts -> make cut on mass log(M) = 1 below limit to get pairs below mass limit
    if z_type != 'p':
        zspec = pd.read_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/zspec_cats/'+field+'/ALL_CANDELS_zcat_'+field+'_zspec_wAIRD.csv')
        zspec.loc[zspec['ZSPEC_AIRD'] == zspec['ZBEST_AIRD'], 'G_ZSPEC'] = zspec['ZSPEC_AIRD'] # assuming this worked..
        df['ZSPEC'] = zspec['G_ZSPEC']
        df = df[ (df['CLASS_STAR'] < 0.9) & (df['PHOTFLAG'] == 0) & (df['MASS'] > (mass_lo-1)) ] ### & (df['ZSPEC'] > 0) ###
    else:                                                                                        ### think about this later
        df = df[ (df['CLASS_STAR'] < 0.9) & (df['PHOTFLAG'] == 0) & (df['MASS'] > (mass_lo-1)) ]
    
    ### THIS SHOULD MAKE IT SO IN 'PS' THE ONES WITH SPEC Z'S SHOULDN'T BE EXCLUDED ### ^^^ ###
    ### WILL NEED TO MAKE MY SPEC-ZS CONSISTENT ACROSS FIELDS THO
    
    
    # check for the spec-z exception and count:
    # print('number of gals with zspec outside redshift range:', len( df[ ((df['ZSPEC'] > 3)) |
    #                                                                    ((df['ZSPEC'] < 0.5) & (df['ZSPEC'] > 0)) ]) )
    
    # reset index
    df = df.reset_index(drop=True)
    
    ##### SMALLER SAMPLE SIZE FOR TEST #####
    # df = df.iloc[0:200]
    
    # draw data for each galaxy and calculate Lx(z) and M(z)
    draw_df_z, draw_df_M, draw_df_LX, draw_df_IR_AGN_DON, draw_df_IR_AGN_STR  = draw_z(df, field)
    
    # create a dictionary to store dfs per each iteration
    field_dict = {}
    
    # create a list to store spec_z counts and a dictionary to store averages
    N_zspec_all = []
    N_zspec = {}
    # loop through number of iterations:
    for it in range(0, len(draw_df_z)):
        print( 'CURRENT ITERATION - '+field, it )
        # calculate separation and delta V ----> might not need LX drop for this step... we'll see
        results, zspec_count = determine_pairs(df, draw_df_z.iloc[it], draw_df_M.iloc[it], draw_df_LX.iloc[it],
                                               draw_df_IR_AGN_DON.iloc[it], draw_df_IR_AGN_STR.iloc[it], z_type, field)
        
        N_zspec_all.append(zspec_count)
        # add dataframe to the dictionary
        field_dict[str(it)] = results
        
        if z_type == 's':
            break
                
    N_zspec[field] = np.mean(N_zspec_all)
        
    return field_dict, N_zspec
    
# -------------------------------------------------------------------------------------------------------------------------- #

    
def draw_z(df, field): # <20 min for one field
    print('Running draw_z for {}'.format(field))
    
    # initialize dictionary
    draw_z = {}
    draw_M = {}
    draw_LX = {}
    draw_IR_AGN_DON = {}
    draw_IR_AGN_STR = {}
    
    for i in range(0, len(df['ID'])): ### THIS BREAKS FOR GDN ### <-- no should be fine actually
        # load PDFs based on string ID
        ID_str = df.loc[i,'ID']
        if len(str(ID_str)) == 1: id_string = '0000'+str(ID_str)
        if len(str(ID_str)) == 2: id_string = '000'+str(ID_str)
        if len(str(ID_str)) == 3: id_string = '00'+str(ID_str)
        if len(str(ID_str)) == 4: id_string = '0'+str(ID_str)
        if len(str(ID_str)) == 5: id_string = str(ID_str)

        if field == "GDS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSS_ID'+id_string+'.pzd'
        elif field == "EGS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/EGS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_EGS_ID'+id_string+'.pzd'
        elif field == "GDN":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSN_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSN_ID'+id_string+'.pzd'
        elif field == "COS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/COSMOS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_COSMOS_ID'+id_string+'.pzd'
        elif field == "UDS":
            pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/UDS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_UDS_ID'+id_string+'.pzd' 

        # read the PDFs
        pdf1 = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                  'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')

        # draw the samples
        sum1 = np.sum(pdf1['mFDa4'])
     
        draw1 = random.choice(pdf1['z'], size=n, p=(pdf1['mFDa4']/sum1))
        
        # this is also where you could calculate Lx(z) and M(z)...
        Mz = [df.loc[i, 'MASS']] * n
        # LXz = [df.loc[i, 'LX']] * n
        
        # calculate luminosity
        DL_mpc = cosmo.luminosity_distance(draw1) # in Mpc -> convert to cm
        DL = DL_mpc.to(u.cm) # distance in cm
        # calculate the k correction
        kz = (1+draw1)**(gamma-2)
        LXz = df.loc[i, 'FX'] * 4 * np.pi * (DL**2) * kz
        # classify as IR AGN, gotta be a 1 or 0 system
        
        # look at IR luminosities
        if field == 'GDN':
            f3p6 = df.loc[i, 'IRAC_CH1_SCANDELS_FLUX']
            f4p5 = df.loc[i, 'IRAC_CH2_SCANDELS_FLUX']
            f5p8 = df.loc[i, 'IRAC_CH3_FLUX']
            f8p0 = df.loc[i, 'IRAC_CH4_FLUX']
        else:
            f3p6 = df.loc[i, 'IRAC_CH1_FLUX']
            f4p5 = df.loc[i, 'IRAC_CH2_FLUX']
            f5p8 = df.loc[i, 'IRAC_CH3_FLUX']
            f8p0 = df.loc[i, 'IRAC_CH4_FLUX']
            
        if f3p6 <= 0 or f4p5 <= 0 or f5p8 <= 0 or f8p0 <= 0:
            IR_AGN_DON = [0]*n
            IR_AGN_STR = [0]*n
        else:
            x = np.log10(f5p8/f3p6)
            y = np.log10(f8p0/f4p5)
            F03p6 = 280.9 #±4.1 Jy
            F04p5 = 179.7 #±2.6 Jy
            F05p8 = 115.0 #±1.7 Jy
            F08p0 = 64.9 #±0.9 Jy         # convert from uJy to Jy
            mv3p6 = 2.5*np.log10(F03p6 / (f3p6/1e6))
            mv4p5 = 2.5*np.log10(F04p5 / (f4p5/1e6))
            mv5p8 = 2.5*np.log10(F05p8 / (f5p8/1e6))
            mv8p0 = 2.5*np.log10(F08p0 / (f8p0/1e6))
            
            if ((x >= 0.08) and (y >= 0.15) and (y >= (1.21*x)-0.27) and (y <= (1.21*x)+0.27) and 
                (f4p5 > f3p6) and (f5p8 > f4p5) and (f8p0 > f5p8)):
                IR_AGN_DON = [1]*n
            else:
                IR_AGN_DON = [0]*n
            if ((mv5p8 - mv8p0 > 0.6) and (mv3p6 - mv4p5 > 0.2 * (mv5p8 - mv8p0) + 0.18) and 
                (mv3p6 - mv4p5 > 2.5 * (mv5p8 - mv8p0) - 3.5)):
                # print(mv3p6, mv4p5, mv5p8, mv8p0)
                # print(mv3p6-mv4p5, mv5p8-mv8p0)     ISSUE HERE
                # sys.exit()
                IR_AGN_STR = [1]*n
            else:
                IR_AGN_STR = [0]*n           
        
        # add entry into dictionary
        draw_z['gal_'+str(ID_str)+'_z'] = draw1
        draw_M['gal_'+str(ID_str)+'_M'] = Mz
        draw_LX['gal_'+str(ID_str)+'_LX'] = LXz
        draw_IR_AGN_DON['gal_'+str(ID_str)+'_IR_AGN_DON'] = IR_AGN_DON
        draw_IR_AGN_STR['gal_'+str(ID_str)+'_IR_AGN_STR'] = IR_AGN_STR
    
    # convert dictionary to dataframe with gal ID as columns and redshift selections are rows
    draw_df_z = pd.DataFrame.from_dict(draw_z)
    draw_df_M = pd.DataFrame.from_dict(draw_M)
    draw_df_LX = pd.DataFrame.from_dict(draw_LX)
    draw_df_IR_AGN_DON = pd.DataFrame.from_dict(draw_IR_AGN_DON)
    draw_df_IR_AGN_STR = pd.DataFrame.from_dict(draw_IR_AGN_STR)
    
    
    ### 0 #######################################################################################################
    # I want to save these draws for GDS and work through the rest of the code with them
    # draw_df_z.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/test_output/drawn_z.csv', index=False)
    # draw_df_M.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/test_output/drawn_M.csv', index=False)
    # draw_df_LX.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/test_output/drawn_LX.csv', index=False)
    # print('WRITTEN')
    
    
    return draw_df_z, draw_df_M, draw_df_LX, draw_df_IR_AGN_DON, draw_df_IR_AGN_STR


# -------------------------------------------------------------------------------------------------------------------------- #


def determine_pairs(all_df, current_zdraw_df, current_Mdraw_df, current_LXdraw_df,
                    current_IR_AGN_DON_draw_df, current_IR_AGN_STR_draw_df, z_type, field):
    ### 1 #######################################################################################################
    if z_type == 'p':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe, but first check for consistent lengths:
        z_drawn = current_zdraw_df.to_numpy()
        M_drawn = current_Mdraw_df.to_numpy()
        LX_drawn = current_LXdraw_df.to_numpy()
        IR_AGN_DON_drawn = current_IR_AGN_DON_draw_df.to_numpy()
        IR_AGN_STR_drawn = current_IR_AGN_STR_draw_df.to_numpy()
        all_df['drawn_z'] = z_drawn
        all_df['drawn_M'] = M_drawn
        all_df['drawn_LX'] = LX_drawn
        all_df['IR_AGN_DON'] = IR_AGN_DON_drawn
        all_df['IR_AGN_STR'] = IR_AGN_STR_drawn
        
    ### BUILD STRUCTURE FOR z_type = 'phot+spec_z' ###
    elif z_type == 'ps':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe
        z_drawn = current_zdraw_df.to_numpy()
        M_drawn = current_Mdraw_df.to_numpy()
        LX_drawn = current_LXdraw_df.to_numpy()
        IR_AGN_DON_drawn = current_IR_AGN_DON_draw_df.to_numpy()
        IR_AGN_STR_drawn = current_IR_AGN_STR_draw_df.to_numpy()
        all_df['drawn_z'] = z_drawn
        all_df['drawn_M'] = M_drawn
        all_df['drawn_LX'] = LX_drawn
        all_df['IR_AGN_DON'] = IR_AGN_DON_drawn
        all_df['IR_AGN_STR'] = IR_AGN_STR_drawn
        # if there is a spec q on quality > 1, change drawn_z to spec_z
        ### WILL NEED TO THINK CAREFULLY ON HOW THIS EFFECTS DRAWN M AND LX ###
        all_df.loc[ (all_df['ZSPEC'] > 0), 'drawn_z'] = all_df['ZSPEC']   
        all_df.loc[ (all_df['ZSPEC'] > 0), 'drawn_LX'] = ( all_df['FX'] * 4 * np.pi *
                                                            ((cosmo.luminosity_distance(all_df['drawn_z']).to(u.cm))**2) * 
                                                            ((1+all_df['drawn_z'])**(gamma-2)) )
        
    elif z_type == 's':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe
        z_drawn = current_zdraw_df.to_numpy()
        M_drawn = current_Mdraw_df.to_numpy()
        LX_drawn = current_LXdraw_df.to_numpy()
        IR_AGN_DON_drawn = current_IR_AGN_DON_draw_df.to_numpy()
        IR_AGN_STR_drawn = current_IR_AGN_STR_draw_df.to_numpy()
        all_df['drawn_z'] = z_drawn
        all_df['drawn_M'] = M_drawn
        all_df['drawn_LX'] = LX_drawn
        all_df['IR_AGN_DON'] = IR_AGN_DON_drawn
        all_df['IR_AGN_STR'] = IR_AGN_STR_drawn
        # make drawn_z the spec z and throw out the rest
        all_df.loc[ (all_df['ZSPEC'] > 0), 'drawn_z'] = all_df['ZSPEC']
        all_df.loc[ (all_df['ZSPEC'] > 0), 'drawn_LX'] = ( all_df['FX'] * 4 * np.pi *
                                                            ((cosmo.luminosity_distance(all_df['drawn_z']).to(u.cm))**2) * 
                                                            ((1+all_df['drawn_z'])**(gamma-2)) )

        all_df = all_df[ (all_df['ZSPEC'] >= 0) ] # will have to exculde COSMOS results
    
        
    ### CHECK THAT THE SPEC Z CUT WORKED ###
    # print(all_df['zspec'], all_df['q_zspec'], all_df['drawn_z'])
    
    # find out how many zspecs we actually end up using
    # print('number of spec z: ', all_df.loc[ (all_df['zspec'] > 0.5) & (all_df['q_zspec'] > 1) , 'drawn_z'].count())
    
    # make definite redshift cut:
    all_df = all_df[ (all_df['drawn_z'] >= 0.5) & (all_df['drawn_z'] <= 3.0) ]
    
    # # look at sample distribution at higher and lower IDs
    # lower = all_df.iloc[ :round(len(all_df)/2) ]
    # upper = all_df.iloc[ round(len(all_df)/2): ]
    # bins,a,b = plt.hist(lower['drawn_z'], 50, label='lower')
    # bins,a,b = plt.hist(upper['drawn_z'], 50, label='upper')
    # plt.show()
    
    # reset this index:    
    all_df = all_df.reset_index(drop=True) # this probably means that previous results are hooplaa
    # keep track of how many spec-z's were used each time
    zspec_count = len(all_df.loc[ all_df['ZSPEC'] > 0 ])
        
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
    
    if len(pair_df) == 0: ###### HERE IS THE PROBLEM ######
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
    #R_kpc = cosmo.arcsec_per_kpc_proper(np.array((all_df.iloc[pair_df['prime_index']])['drawn_z'])
    pair_df['kpc_sep'] = (pair_df['arc_sep']) / cosmo.arcsec_per_kpc_proper(all_df.loc[pair_df['prime_index'], 'drawn_z'])
        
    #print('before true pairs:', len(pair_df))
    true_pairs = pair_df[ (pair_df['kpc_sep'] <= max_sep*u.kpc) & (abs(pair_df['dv']) <= 1000) ]
    #print(true_pairs)
    
    ### ADD MASS LIMIT CUT NOW TO TRUE PAIRS, BC WE WILL NEED TO MATCH THE SMALLER GALS IN OUR ISO GROUP
    # n, bins, patches = plt.hist(all_df.loc[ true_pairs['prime_index'], 'drawn_M' ], bins=50, histtype='step')
    # used np array function to resolve duplicate index issues in pandas
    
    ### 5 #######################################################################################################
    true_pairs = true_pairs.iloc[ np.where( np.array(all_df.loc[ true_pairs['prime_index'], 'drawn_M' ] > mass_lo) == True ) ]
    
    if len(true_pairs) == 0: ###### HERE IS THE PROBLEM ######
        print(field)
        true_pairs =  true_pairs
        zspec_count = 0
        return true_pairs, zspec_count
    
    
    # plt.hist(all_df.loc[ true_pairs['prime_index'], 'drawn_M' ], bins=bins, histtype='step')
    # plt.hist(all_df.loc[ true_pairs['partner_index'], 'drawn_M' ], bins=bins, histtype='step')
    # plt.show()
    # histograms confirm the mass cut on prime galaxies was successful
    ### -----> revisit this syntax later...
        
    ### 6 #######################################################################################################

    # add galaxies that aren't pairs into the isolated sample:
    #iso_add = (pair_df[ (pair_df['kpc_sep'] > 100*u.kpc) | (abs(pair_df['dv']) > 10000) ])
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
    
    ### perhaps I should just create an array for pair and iso values rather than tracing them back each time?
    
    # select control galaxies from iso_df ---> needs to be fixed ----> need to make sure indices are right here...
    pair_mass = np.concatenate( (np.array(all_df.loc[true_pairs['prime_index'], 'drawn_M']), 
                                          np.array(all_df.loc[true_pairs['partner_index'], 'drawn_M'])), axis=0 )
    pair_z = np.concatenate( (np.array(all_df.loc[true_pairs['prime_index'], 'drawn_z']), 
                                          np.array(all_df.loc[true_pairs['partner_index'], 'drawn_z'])), axis=0 )
    pair_idx = np.concatenate( (np.array(true_pairs['prime_index']), np.array(true_pairs['partner_index'])), axis=0 )
    iso_mass = all_df.loc[all_iso, 'drawn_M']
    iso_z = all_df.loc[all_iso, 'drawn_z']
    iso_idx = all_iso


    # shuffle pair info to get rid of prime mass bias
    data_length = pair_idx.shape[0]
    # Here we create an array of shuffled indices
    shuf_order = np.arange(data_length)
    np.random.shuffle(shuf_order)

    shuf_idx = pair_idx[shuf_order] # Shuffle the original data
    shuf_mass = pair_mass[shuf_order]
    shuf_z = pair_z[shuf_order]
    
    # run controls function
    shuf_controls, shuf_c_flag = get_control(iso_idx, iso_mass, iso_z, shuf_idx, shuf_mass, shuf_z)

    # Create an inverse of the shuffled index array (to reverse the shuffling operation, or to "unshuffle")
    unshuf_order = np.zeros_like(shuf_order)
    unshuf_order[shuf_order] = np.arange(data_length)

    unshuf_controls = shuf_controls[unshuf_order] # Unshuffle the shuffled data
    unshuf_cflag = shuf_c_flag[unshuf_order]
    
    controls = unshuf_controls # just so the names remain the same
    c_flag = unshuf_cflag
    
    # let's modify control selection to see if selection improves
    
    
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
    c1prime_no99 = prime_controls[:,0][np.where(prime_controls[:,0] != -99)]
    c2prime_no99 = prime_controls[:,1][np.where(prime_controls[:,1] != -99)]
    prime_flags = c_flag[:middle_idx]
    
    partner_controls = controls[middle_idx:]
    c1partner_no99 = partner_controls[:,0][np.where(partner_controls[:,0] != -99)]
    c2partner_no99 = partner_controls[:,1][np.where(partner_controls[:,1] != -99)]
    partner_flags = c_flag[middle_idx:]
            
    # add prime control galaxies to true_pair df ----> reminder these are indices of all_df (CHECK -> pretty sure)
    # problem that nans won't index
    # true_pairs['prime_control_idx1'] = iso_idx[(prime_controls[:,0]).tolist()]
    # true_pairs['prime_control_idx2'] = iso_idx[(prime_controls[:,1]).tolist()]
    # true_pairs['partner_control_idx1'] = iso_idx[(partner_controls[:,0]).tolist()]
    # true_pairs['partner_control_idx2'] = iso_idx[(partner_controls[:,1]).tolist()]
    
    # add other important data to the dataframe ====> all the drawn values 
    # worry about logical position in the df later
    true_pairs['prime_cat_ID'] = np.array(all_df.loc[ true_pairs['prime_index'], 'ID' ])
    true_pairs['prime_drawn_z'] = np.array(all_df.loc[true_pairs['prime_index'], 'drawn_z'])
    true_pairs['prime_drawn_M'] = np.array(all_df.loc[true_pairs['prime_index'], 'drawn_M'])
    true_pairs['prime_drawn_LX'] = np.array(all_df.loc[true_pairs['prime_index'], 'drawn_LX'])
    true_pairs['prime_IR_AGN_DON'] = np.array(all_df.loc[true_pairs['prime_index'], 'IR_AGN_DON'])
    true_pairs['prime_IR_AGN_STR'] = np.array(all_df.loc[true_pairs['prime_index'], 'IR_AGN_STR'])
    
    true_pairs['partner_cat_ID'] = np.array(all_df.loc[ true_pairs['partner_index'], 'ID' ])
    true_pairs['partner_drawn_z'] = np.array(all_df.loc[true_pairs['partner_index'], 'drawn_z'])
    true_pairs['partner_drawn_M'] = np.array(all_df.loc[true_pairs['partner_index'], 'drawn_M'])
    true_pairs['partner_drawn_LX'] = np.array(all_df.loc[true_pairs['partner_index'], 'drawn_LX'])
    true_pairs['partner_IR_AGN_DON'] = np.array(all_df.loc[true_pairs['partner_index'], 'IR_AGN_DON'])
    true_pairs['partner_IR_AGN_STR'] = np.array(all_df.loc[true_pairs['partner_index'], 'IR_AGN_STR'])
    
    true_pairs['prime_control1_idx'] = prime_controls[:,0]
    true_pairs['prime_control2_idx'] = prime_controls[:,1]
    true_pairs['partner_control1_idx'] = partner_controls[:,0]
    true_pairs['partner_control2_idx'] = partner_controls[:,1]
    
    # prime galaxy control 1
    true_pairs['prime_control1_cat_ID'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control1_idx'] != -99, 'prime_control1_cat_ID' ] = np.array(all_df.loc[ c1prime_no99, 'ID' ]) 
    true_pairs['prime_control1_drawn_z'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control1_idx'] != -99, 'prime_control1_drawn_z' ] = np.array(all_df.loc[ c1prime_no99, 'drawn_z' ]) 
    true_pairs['prime_control1_drawn_M'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control1_idx'] != -99, 'prime_control1_drawn_M' ] = np.array(all_df.loc[ c1prime_no99, 'drawn_M' ])  
    true_pairs['prime_control1_drawn_LX'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control1_idx'] != -99, 'prime_control1_drawn_LX' ] = np.array(all_df.loc[ c1prime_no99, 'drawn_LX' ])  
    true_pairs['prime_control1_IR_AGN_DON'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control1_idx'] != -99, 'prime_control1_IR_AGN_DON' ] = np.array(all_df.loc[ c1prime_no99, 'IR_AGN_DON' ]) 
    true_pairs['prime_control1_IR_AGN_STR'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control1_idx'] != -99, 'prime_control1_IR_AGN_STR' ] = np.array(all_df.loc[ c1prime_no99, 'IR_AGN_STR' ]) 
    
    # prime galaxy control 2
    true_pairs['prime_control2_cat_ID'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control2_idx'] != -99, 'prime_control2_cat_ID' ] = np.array(all_df.loc[ c2prime_no99, 'ID' ]) 
    true_pairs['prime_control2_drawn_z'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control2_idx'] != -99, 'prime_control2_drawn_z' ] = np.array(all_df.loc[ c2prime_no99, 'drawn_z' ]) 
    true_pairs['prime_control2_drawn_M'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control2_idx'] != -99, 'prime_control2_drawn_M' ] = np.array(all_df.loc[ c2prime_no99, 'drawn_M' ])  
    true_pairs['prime_control2_drawn_LX'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control2_idx'] != -99, 'prime_control2_drawn_LX' ] = np.array(all_df.loc[ c2prime_no99, 'drawn_LX' ])  
    true_pairs['prime_control2_IR_AGN_DON'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control2_idx'] != -99, 'prime_control2_IR_AGN_DON' ] = np.array(all_df.loc[ c2prime_no99, 'IR_AGN_DON' ]) 
    true_pairs['prime_control2_IR_AGN_STR'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['prime_control2_idx'] != -99, 'prime_control2_IR_AGN_STR' ] = np.array(all_df.loc[ c2prime_no99, 'IR_AGN_STR' ]) 
    
    # partner galaxy control 1
    true_pairs['partner_control1_cat_ID'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control1_idx'] != -99, 'partner_control1_cat_ID' ] = np.array(all_df.loc[ c1partner_no99, 'ID' ]) 
    true_pairs['partner_control1_drawn_z'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control1_idx'] != -99, 'partner_control1_drawn_z' ] = np.array(all_df.loc[ c1partner_no99, 'drawn_z' ]) 
    true_pairs['partner_control1_drawn_M'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control1_idx'] != -99, 'partner_control1_drawn_M' ] = np.array(all_df.loc[ c1partner_no99, 'drawn_M' ])  
    true_pairs['partner_control1_drawn_LX'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control1_idx'] != -99, 'partner_control1_drawn_LX' ] = np.array(all_df.loc[ c1partner_no99, 'drawn_LX' ])  
    true_pairs['partner_control1_IR_AGN_DON'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control1_idx'] != -99, 'partner_control1_IR_AGN_DON' ] = np.array(all_df.loc[ c1partner_no99, 'IR_AGN_DON' ]) 
    true_pairs['partner_control1_IR_AGN_STR'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control1_idx'] != -99, 'partner_control1_IR_AGN_STR' ] = np.array(all_df.loc[ c1partner_no99, 'IR_AGN_STR' ]) 
    
    # partner galaxy control 2
    true_pairs['partner_control2_cat_ID'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control2_idx'] != -99, 'partner_control2_cat_ID' ] = np.array(all_df.loc[ c2partner_no99, 'ID' ]) 
    true_pairs['partner_control2_drawn_z'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control2_idx'] != -99, 'partner_control2_drawn_z' ] = np.array(all_df.loc[ c2partner_no99, 'drawn_z' ]) 
    true_pairs['partner_control2_drawn_M'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control2_idx'] != -99, 'partner_control2_drawn_M' ] = np.array(all_df.loc[ c2partner_no99, 'drawn_M' ])  
    true_pairs['partner_control2_drawn_LX'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control2_idx'] != -99, 'partner_control2_drawn_LX' ] = np.array(all_df.loc[ c2partner_no99, 'drawn_LX' ])  
    true_pairs['partner_control2_IR_AGN_DON'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control2_idx'] != -99, 'partner_control2_IR_AGN_DON' ] = np.array(all_df.loc[ c2partner_no99, 'IR_AGN_DON' ]) 
    true_pairs['partner_control2_IR_AGN_STR'] = [-99]*len(true_pairs)
    true_pairs.loc[ true_pairs['partner_control2_idx'] != -99, 'partner_control2_IR_AGN_STR' ] = np.array(all_df.loc[ c2partner_no99, 'IR_AGN_STR' ]) 

    true_pairs['prime_cflag1'] = prime_flags[:,0]
    true_pairs['prime_cflag2'] = prime_flags[:,1]
    true_pairs['partner_cflag1'] = partner_flags[:,0]
    true_pairs['partner_cflag2'] = partner_flags[:,1]
    
    
    true_pairs['field'] = [field]*len(true_pairs)

    
    return true_pairs, zspec_count
    
    
    
# -------------------------------------------------------------------------------------------------------------------------- #

def get_control(control_ID, control_mass, control_z, gal_ID, mass, redshift, N_control=2, zfactor=0.2, mfactor=2): 
    
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
    iso = {'ID':control_ID, 'z':control_z, 'mass':control_mass}
    all_iso_df = pd.DataFrame( iso )
    # somehow indices were carried with these values, so if we want to index a list of indices we want:
    all_iso_df = all_iso_df.reset_index(drop=True)
    
    for i, (ID, m, z) in enumerate(zip(gal_ID, mass, redshift)):
        
        control = np.full(2, -99)

        zmin = z - dz
        zmax = z + dz
        mmin = m-np.log10(mfactor)
        mmax = m+np.log10(mfactor)

        # create a dataframe for possible matches
        # cmatch_df = all_iso_df[ (all_iso_df['z'] >= zmin) & (all_iso_df['z'] <= zmax) & (all_iso_df['mass'] >= mmin) &
        #                        (all_iso_df['mass'] <= mmax) ]
        cmatch_df = all_iso_df
        
        # create columns for difference between z/mass control and pair z/m
        cmatch_df['dif'] = (cmatch_df['z'] - z)**2 + (cmatch_df['mass'] - m) **2
                
        # need to sort dataframe based on two columns THEN continue
        cmatch_df.sort_values(by=['dif'], inplace=True, ascending = True)

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


# -------------------------------------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    main()