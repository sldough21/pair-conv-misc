# Sean Dougherty
# 03/11/2022
# a reorganized version of z_cuts.py to maximize efficiency

# import libraries
import pandas as pd
import numpy as np
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
import math as m
from multiprocessing import Pool, freeze_support, RLock

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord

from numpy import random

PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data_CSV/'

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
# determine conservative ang separation correspondong to 150 kpc at z = 0.5
R_kpc = cosmo.arcsec_per_kpc_proper(0.5) # arcsec/kpc at z=0.5
max_R_kpc = (150*u.kpc * R_kpc) # in arcseconds ### this is the bug right here


def main():
    
    print('beginning main()')
    
    # we want to parallelize the data by fields, so:
    all_fields = ['GDS']#,'EGS','COS','GDN','UDS']
    
    
    # Create a multiprocessing Pool
    # pool = Pool()  
    # # process fields iterable with pool -> parallelize code by field
    # all_data = pool.map(process_samples, all_fields)
    # # close pool
    # pool.close()
    # pool.join()
    
    # for now, run things without pooling -> easier to read errors
    process_samples('GDS')
    
# -------------------------------------------------------------------------------------------------------------------------- #

    
def process_samples(field):
    # this is essentially the main function but for each field, to be combined and saved as csv's upon completion
    print('beginning process_samples() for {}'.format(field))

    # load data 
    df_1 = pd.read_csv(PATH+'CANDELS_'+field+'_1018_LX.csv')
    df_2 = pd.read_csv(PATH+'CANDELS_'+field+'_1018_LX_z.csv')
    df_3 = pd.read_csv(PATH+'zcat_'+field+'_v2.0.csv') # need to ask Dale where this data is from !!! # are the zhi/lo accurate tho...
    
    df = df_1.join(df_2).join(df_3)
    print(df['zbest'], df['zbest2']) # these are differet, though the ID's are correct
    
    # check that IDs are consistent then drop IDs
    df = df.drop(['id2'], axis=1)
    
    # make initial galaxy cuts based on PDF range
    df = df[ (df['zlo'] <= 3.0) & (df['zhi'] >= 0.5) & (df['class_star'] < 0.9) & (df['photflag'] == 0) ]
    # check for the spec-z exception and count:
    print('number of gals with zspec outside redshift range:', len( df[ ((df['zspec'] > 3)) | ((df['zspec'] < 0.5) & (df['zspec'] > 0)) ]) )
    
    # reset index
    df = df.reset_index(drop=True)
    
    ##### SMALLER SAMPLE SIZE FOR TEST #####
    df = df.iloc[0:100]
    
    # draw 1000 galaxies for each galaxy and calculate Lx(z) and M(z)
    draw_df_z, draw_df_M, draw_df_LX = draw_z(df, field)
    
    # loop through number of iterations:
    for it in range(0, len(draw_df_z)):
        
        # calculate separation and delta V ----> might not need LX drop for this step... we'll see
        results = determine_pairs(df, draw_df_z.iloc[it], draw_df_M.iloc[it], draw_df_LX.iloc[it], 'phot-z')
    
# -------------------------------------------------------------------------------------------------------------------------- #

    
def draw_z(df, field): # <20 min for one field
    print('Running draw_z for {}'.format(field))
    
    # initialize dictionary
    draw_z = {}
    draw_M = {}
    draw_LX = {}
    
    for i in tqdm(range(0, len(df['ID']))):
        # load PDFs based on string ID
        ID_str = df['ID'][i]
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
        n = 5 # number of draws
        sum1 = np.sum(pdf1['HB4'])
     
        draw1 = random.choice(pdf1['z'], size=n, p=(pdf1['HB4']/sum1))
        
        # this is also where you could calculate Lx(z) and M(z)...
        Mz = [np.array(df['mass'][i])] * n
        LXz = [np.array(df['LX'][i])] * n
        
        # add entry into dictionary
        draw_z['gal_'+str(ID_str)+'_z'] = draw1
        draw_M['gal_'+str(ID_str)+'_M'] = Mz
        draw_LX['gal_'+str(ID_str)+'_LX'] = LXz
    
    # convert dictionary to dataframe with gal ID as columns and redshift selections are rows
    draw_df_z = pd.DataFrame.from_dict(draw_z)
    draw_df_M = pd.DataFrame.from_dict(draw_M)
    draw_df_LX = pd.DataFrame.from_dict(draw_LX)
    
    return draw_df_z, draw_df_M, draw_df_LX


# -------------------------------------------------------------------------------------------------------------------------- #


def determine_pairs(all_df, current_zdraw_df, current_Mdraw_df, current_LXdraw_df, z_type):
    if z_type == 'phot-z':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe, but first check for consistent lengths:
        z_drawn = current_zdraw_df.to_numpy()
        M_drawn = current_Mdraw_df.to_numpy()
        LX_drawn = current_LXdraw_df.to_numpy()
        print('checking length of drawn z list')
        all_df['drawn_z'] = z_drawn
        all_df['drawn_M'] = M_drawn
        all_df['drawn_LX'] = LX_drawn
        
    #elif z_type == 'spec-z':
        
    #elif z_type == 'both':
    
    # make definite redshift cut:
    all_df = all_df[ (all_df['drawn_z'] >= 0.5) & (all_df['drawn_z'] <= 3.0) ]
    
    # match catalogs:
    df_pos = SkyCoord(all_df['RAdeg'],all_df['DEdeg'],unit='deg')
    idxc, idxcatalog, d2d, d3d = df_pos.search_around_sky(df_pos, max_R_kpc)
    # idxc is INDEX of the item being searched around
    # idxcatalog is INDEX of all galaxies within arcsec
    # d2d is the arcsec differece
    
    # place galaxy pairs into a df and get rid of duplicate pairs:
    matches = {'prime_index':idxc, 'partner_index':idxcatalog, 'arc_sep': d2d.arcsecond}
    match_df = pd.DataFrame(matches)
    pair_df = match_df[ (match_df['arc_sep'] != 0.00) ]
    # get rid of inverse row pairs with mass ratio
    pair_df['mass_ratio'] = np.array((all_df.iloc[pair_df['prime_index']])['drawn_M']) - np.array((all_df.iloc[pair_df['partner_index']])['drawn_M'])
    
    pair_df = pair_df[ (pair_df['mass_ratio'] >= 0) ] # WHY ISN'T IT EXACTLY HALF -> when it's 0 you keep the duplicate...
    
    iso_df = match_df[ (match_df['arc_sep'] == 0.00) ]
    # confidently isolated galaxies only match to themselves, so get rid of ID's with other matches
    iso_df = iso_df[ (iso_df['prime_index'].isin(pair_df['prime_index']) == False) ]
    #print(iso_df)
    
    # calculate relative line of sight velocity
    pair_df['dv'] = ( (((np.array((all_df.iloc[pair_df['prime_index']])['drawn_z'])+1)**2 -1)/((np.array((all_df.iloc[pair_df['prime_index']])['drawn_z'])+1)**2 +1)) - (((np.array((all_df.iloc[pair_df['partner_index']])['drawn_z'])+1)**2 -1)/((np.array((all_df.iloc[pair_df['partner_index']])['drawn_z'])+1)**2 +1)) ) * 2.998e5
    
    # calculate projected separation at z
    #R_kpc = cosmo.arcsec_per_kpc_proper(np.array((all_df.iloc[pair_df['prime_index']])['drawn_z'])
    pair_df['kpc_sep'] = (pair_df['arc_sep']) / cosmo.arcsec_per_kpc_proper((all_df.iloc[pair_df['prime_index']])['drawn_z'])
    
    true_pairs = pair_df[ (pair_df['kpc_sep'] <= 80*u.kpc) & (abs(pair_df['dv']) <= 1000) ]
    print(true_pairs)
    
    # select control galaxies from iso_df ---> needs to be fixed ----> need to make sure indices are right here...
    pair_mass = np.concatenate( (np.array(), np.array()), axis=0 )
    pair_z = np.concatenate( (np.array(), np.array()), axis=0 )
    iso_mass = 
    iso_z = 
    controls = get_control(iso_mass, iso_z, pair_mass, pair_z)
    
    
# -------------------------------------------------------------------------------------------------------------------------- #

def get_control(control_mass, control_z, mass, redshift, N_control=2, zfactor=0.2, mfactor=2): 

    dz = zfactor

    # create list to store lists of control indices per prime galaxy
    control_all = []
        
    # create a list for all ID's to make sure there are no duplicates
    control_dup = []
    
    for i, (m, z) in enumerate(zip(mass, redshift)):
        
        control = []
    
        zmin = z - dz
        zmax = z + dz
        mmin = m-np.log10(mfactor)
        mmax = m+np.log10(mfactor)
     
         # create a dataframe for possible matches
        control_match = np.where( (control_z >= zmin) & (control_z <= zmax) & 
                                 (control_m >= mmin) & (control_m <= mmax) )
     
        # randomize df_iso and move through it until we have desired number of control galaxies
        random.shuffle(control_match)
        mcount = 0

        for j in range(0, len(control_match)):
     
            if control_match[j] in control_dup:
                continue
            else:
                control.append(control_match[j])
                control_dup.append(control_match[j])
                mcount+=1
       
            # print('Not enough control galaxies for object {}!'.format(i))
            if mcount == N_control: 
                break
                
        if mcount < N_control:
            print('Not enough control galaxies for object {}!'.format(i))
            control.append(['nan']*(N_control-mcount))
    
        control_all.append([control])

    return control_all


# -------------------------------------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    main()

    
    
# NOTES
# need Dale to clarify a few things:
#   the data from zcat are for the mFDa4 catalog, but the HB4 is best from what I understand, so this is no good
#   will need to choose which PDF's to do the analysis for
#   right now, zlo and zhi are the 68.3% confidence intervals for the mFDa4 method...
#   now what are these photflags?
#   what are the q_specz flags?