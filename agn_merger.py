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

from numpy import random

import matplotlib.pyplot as plt

PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data_CSV/'

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
# determine conservative ang separation correspondong to 150 kpc at z = 0.5
R_kpc = cosmo.arcsec_per_kpc_proper(0.5) # arcsec/kpc at z=0.5
max_R_kpc = (150*u.kpc * R_kpc) # in arcseconds ### this is the bug right here

mass_lo = 8.5


# -------------------------------------------------------------------------------------------------------------------------- #


def main():
    
    print('beginning main()')
    
    # we want to parallelize the data by fields, so:
    all_fields = ['GDS','EGS','COS','GDN','UDS']
    
    
    # Create a multiprocessing Pool
    pool = Pool()  
    # process fields iterable with pool -> parallelize code by field
    all_data = pool.map(process_samples, all_fields)
    # close pool
    pool.close()
    pool.join()
    
    # endf = pd.DataFrame(all_data[0]['0'])
    # endf.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/photoz_TEST.csv')
    
    # combine dfs for each iteration and save them as a csv file:
    # all_data will be 5 dictionaries... test:
    GDS_dict = all_data[0]
    EGS_dict = all_data[1]
    COS_dict = all_data[2]
    GDN_dict = all_data[3]
    UDS_dict = all_data[4]
    for it in GDS_dict:
        combined_df = pd.concat([GDS_dict[it], EGS_dict[it], COS_dict[it], GDN_dict[it], UDS_dict[it]])
        #combined_df.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/photoz_results/photoz_'+str(it)+'.csv')
        combined_df.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/photo-specz_results/photo-specz_'+str(it)+'.csv')
    
    print('files written!')
        
    
    # for now, run things without pooling -> easier to read errors
    # process_samples('GDS')
    
# -------------------------------------------------------------------------------------------------------------------------- #

    
def process_samples(field):
    # this is essentially the main function but for each field, to be combined and saved as csv's upon completion
    print('beginning process_samples() for {}'.format(field))

    # load data 
    df_1 = pd.read_csv(PATH+'CANDELS_'+field+'_1018_LX.csv')
    df_2 = pd.read_csv(PATH+'CANDELS_'+field+'_1018_LX_z.csv')
    df_3 = pd.read_csv(PATH+'zcat_'+field+'_v2.0.csv') # need to ask Dale where this data is from !!! # are the zhi/lo accurate tho...
    
    df = df_1.join(df_2).join(df_3)
    #print(df['zbest'], df['zbest2']) # these are differet, though the ID's are correct
    
    # check that IDs are consistent then drop IDs
    df = df.drop(['id2'], axis=1)
    
    # throw away galaxies with PDF confidence intervals beyond redshift range
    df = df.drop(df[ (df['zlo'] > 3.5) | (df['zhi'] < 0.25) ].index)
    # make additional quality cuts -> make cut on mass log(M) = 1 below limit to get pairs below mass limit
    df = df[ (df['class_star'] < 0.9) & (df['photflag'] == 0) & (df['mass'] > (mass_lo-1)) ]
    
    # check for the spec-z exception and count:
    print('number of gals with zspec outside redshift range:', len( df[ ((df['zspec'] > 3)) | ((df['zspec'] < 0.5) & (df['zspec'] > 0)) ]) )
    
    # reset index
    df = df.reset_index(drop=True)
    
    ##### SMALLER SAMPLE SIZE FOR TEST #####
    #df = df.iloc[0:100]
    
    # draw 1000 galaxies for each galaxy and calculate Lx(z) and M(z)
    draw_df_z, draw_df_M, draw_df_LX = draw_z(df, field)
    
    # create a dictionary to store dfs per each iteration
    field_dict = {}
    
    # loop through number of iterations:
    for it in range(0, len(draw_df_z)):
        print( 'CURRENT ITERATION - '+field, it )
        # calculate separation and delta V ----> might not need LX drop for this step... we'll see
        #results = determine_pairs(df, draw_df_z.iloc[it], draw_df_M.iloc[it], draw_df_LX.iloc[it], 'phot-z', field)
        results = determine_pairs(df, draw_df_z.iloc[it], draw_df_M.iloc[it], draw_df_LX.iloc[it], 'phot+spec_z', field)

        
        # add dataframe to the dictionary
        field_dict[str(it)] = results
    
    return field_dict
    
# -------------------------------------------------------------------------------------------------------------------------- #

    
def draw_z(df, field): # <20 min for one field
    print('Running draw_z for {}'.format(field))
    
    # initialize dictionary
    draw_z = {}
    draw_M = {}
    draw_LX = {}
    
    for i in range(0, len(df['ID'])):
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
        n = 500 # number of draws
        sum1 = np.sum(pdf1['HB4'])
     
        draw1 = random.choice(pdf1['z'], size=n, p=(pdf1['HB4']/sum1))
        
        # this is also where you could calculate Lx(z) and M(z)...
        Mz = [df.loc[i, 'mass']] * n
        LXz = [df.loc[i, 'LX']] * n
        
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


def determine_pairs(all_df, current_zdraw_df, current_Mdraw_df, current_LXdraw_df, z_type, field):
    if z_type == 'phot-z':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe, but first check for consistent lengths:
        z_drawn = current_zdraw_df.to_numpy()
        M_drawn = current_Mdraw_df.to_numpy()
        LX_drawn = current_LXdraw_df.to_numpy()
        #print('checking length of drawn z list')
        all_df['drawn_z'] = z_drawn
        all_df['drawn_M'] = M_drawn
        all_df['drawn_LX'] = LX_drawn
        
    ### BUILD STRUCTURE FOR z_type = 'phot+spec_z' ###
    elif z_type == 'phot+spec_z':
        # if we are choosing just photo-z's, stick with the draws
        # add current z to the all_df dataframe
        z_drawn = current_zdraw_df.to_numpy()
        M_drawn = current_Mdraw_df.to_numpy()
        LX_drawn = current_LXdraw_df.to_numpy()
        #print('checking length of drawn z list')
        all_df['drawn_z'] = z_drawn
        all_df['drawn_M'] = M_drawn
        all_df['drawn_LX'] = LX_drawn
        # if there is a spec q on quality > 1, change drawn_z to spec_z
        ### WILL NEED TO THINK CAREFULLY ON HOW THIS EFFECTS DRAWN M AND LX ###
        all_df.loc[ (all_df['zspec'] > 0.5) & (all_df['q_zspec'] > 1) , 'drawn_z'] = all_df['zspec']
        
    ### CHECK THAT THE SPEC Z CUT WORKED ###
    # print(all_df['zspec'], all_df['q_zspec'], all_df['drawn_z'])
    
    # find out how many zspecs we actually end up using
    # print('number of spec z: ', all_df.loc[ (all_df['zspec'] > 0.5) & (all_df['q_zspec'] > 1) , 'drawn_z'].count())
    
    # make definite redshift cut:
    all_df = all_df[ (all_df['drawn_z'] >= 0.5) & (all_df['drawn_z'] <= 3.0) ]
    # reset this index:
    all_df = all_df.reset_index(drop=True) # this probably means that previous results are hooplaa
    
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
    # get rid of inverse row pairs with mass ratio -------------------------------------> CHANGED THIS, CHECK THAT OK
    pair_df['mass_ratio'] = (np.array(all_df.loc[pair_df['prime_index'], 'drawn_M']) - 
                             np.array(all_df.loc[pair_df['partner_index'],'drawn_M']) )
    
    pair_df = pair_df[ (pair_df['mass_ratio'] >= 0) ] ### BE CAREFUL WITH ISO SELECTION
    
    ######## ways to fix this duplicate issue -> separate out cases with mass_ratio = exactly 0 then...
    # potential solution
    sorted_idx_df = pd.DataFrame(np.sort((pair_df.loc[:,['prime_index','partner_index']]).values, axis=1), 
                                    columns=(pair_df.loc[:,['prime_index','partner_index']]).columns).drop_duplicates()
    pair_df = pair_df.reset_index(drop=True)
    pair_df = pair_df.iloc[sorted_idx_df.index]
    ######## - works, but may consider moving this later in the function to decrease data size in this computation
    
    iso_df = match_df[ (match_df['arc_sep'] == 0.00) ]
    # confidently isolated galaxies only match to themselves, so get rid of ID's with other matches
    iso_df = iso_df[ (iso_df['prime_index'].isin(pair_df['prime_index']) == False) ]
    
    # Do the second bit of mass_ratio cut after iso gal selection so isolated sample doesn't include high mass ratio pairs
    pair_df = pair_df[ (pair_df['mass_ratio'] <= 1) ] ### BE CAREFUL WITH ISO SELECTION
    
    #print(field, 'confident isolated galaxy {}'.format(len(iso_df)))
    
    # calculate relative line of sight velocity
    pair_df['dv'] = ( (((np.array(all_df.loc[pair_df['prime_index'], 'drawn_z'])+1)**2 -1)/ 
                       ((np.array(all_df.loc[pair_df['prime_index'], 'drawn_z'])+1)**2 +1)) - 
                     (((np.array(all_df.loc[pair_df['partner_index'], 'drawn_z'])+1)**2 -1)/ 
                      ((np.array(all_df.loc[pair_df['partner_index'], 'drawn_z'])+1)**2 +1)) ) * 2.998e5
    
    # calculate projected separation at z
    #R_kpc = cosmo.arcsec_per_kpc_proper(np.array((all_df.iloc[pair_df['prime_index']])['drawn_z'])
    pair_df['kpc_sep'] = (pair_df['arc_sep']) / cosmo.arcsec_per_kpc_proper(all_df.loc[pair_df['prime_index'], 'drawn_z'])
        
    #print('before true pairs:', len(pair_df))
    true_pairs = pair_df[ (pair_df['kpc_sep'] <= 100*u.kpc) & (abs(pair_df['dv']) <= 1000) ]
    #print(true_pairs)
    
    ### ADD MASS LIMIT CUT NOW TO TRUE PAIRS, BC WE WILL NEED TO MATCH THE SMALLER GALS IN OUR ISO GROUP
    # n, bins, patches = plt.hist(all_df.loc[ true_pairs['prime_index'], 'drawn_M' ], bins=50, histtype='step')
    # used np array function to resolve duplicate index issues in pandas
    true_pairs = true_pairs.iloc[ np.where( np.array(all_df.loc[ true_pairs['prime_index'], 'drawn_M' ] > mass_lo) == True ) ]
    # plt.hist(all_df.loc[ true_pairs['prime_index'], 'drawn_M' ], bins=bins, histtype='step')
    # plt.hist(all_df.loc[ true_pairs['partner_index'], 'drawn_M' ], bins=bins, histtype='step')
    # plt.show()
    # histograms confirm the mass cut on prime galaxies was successful
    ### -----> revisit this syntax later...
        
    # add galaxies that aren't pairs into the isolated sample:
    ############################################################################################################
    #iso_add = (pair_df[ (pair_df['kpc_sep'] > 100*u.kpc) | (abs(pair_df['dv']) > 10000) ])
    iso_add = (pair_df[ (abs(pair_df['dv']) > 10000) ])

    # just stack prime and partner indices into massive array:
    iso_add_idx = np.concatenate( (np.array(iso_add['prime_index']), np.array(iso_add['partner_index'])), axis=0)
    # return unique indices
    iso_add_uniq = np.unique(iso_add_idx)
    # get rid of cases where those indices appear elsewhere, so create array for true pair indices
    true_pair_idx = np.concatenate( (np.array(true_pairs['prime_index']), np.array(true_pairs['partner_index'])), axis=0)
    # only keep the elements that aren't in true pair:
    mask = np.isin(iso_add_uniq, true_pair_idx, invert=True)
    iso_unq = iso_add_uniq[mask]
    
    ######################### based on the extremely low AGN fractions, I'd bet it's an issue before ###########
    ############################################################################################################
    
    ### perhaps I should just create an array for pair and iso values rather than tracing them back each time?
    
    # select control galaxies from iso_df ---> needs to be fixed ----> need to make sure indices are right here...
    pair_mass = np.concatenate( (np.array(all_df.loc[true_pairs['prime_index'], 'drawn_M']), 
                                          np.array(all_df.loc[true_pairs['partner_index'], 'drawn_M'])), axis=0 )
    pair_z = np.concatenate( (np.array(all_df.loc[true_pairs['prime_index'], 'drawn_z']), 
                                          np.array(all_df.loc[true_pairs['partner_index'], 'drawn_z'])), axis=0 )
    iso_mass = np.concatenate( (np.array(all_df.loc[iso_df['prime_index'], 'drawn_M']), 
                                np.array(all_df.loc[iso_unq, 'drawn_M'])), axis=0 )
    iso_z = np.concatenate( (np.array(all_df.loc[iso_df['prime_index'], 'drawn_z']), 
                             np.array(all_df.loc[iso_unq, 'drawn_z'])), axis=0 )
    
    #sprint( field, 'number of added isolated galaxies to the sample {}'.format(len(iso_z)) )
    
    # should return indices of two control galaxies ---- 
    # OH BUT NEED TO SOMEHOW KEEP THEM WITH PAIR -> split the array directly in half and bring pairs back together
    # print('NUMBER OF PAIRS: {}; NUMBER OF ISOS: {}'.format(len(pair_mass),len(iso_mass)))
    # returning indices for dataframe containing iso_mass and iso_z
    controls = get_control(iso_mass, iso_z, pair_mass, pair_z)
    
    ####################
    # CONTROL BUG HERE #
    iso_idx = np.concatenate( (np.array(iso_df['prime_index']), np.array(iso_unq)), axis=0 )
    # so we need to index iso_idx with indices from controls before getting mass and z from all_df
    ####################
    
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
    idx11_z=[]
    idx11_M=[]
    idx11_LX=[]
    idx12_id=[]
    idx12_z=[]
    idx12_M=[]
    idx12_LX=[]
    idx21_id=[]
    idx21_z=[]
    idx21_M=[]
    idx21_LX=[]
    idx22_id=[]
    idx22_z=[]
    idx22_M=[]
    idx22_LX=[]
    
    for idx11, idx12, idx21, idx22 in zip((prime_controls[:,0]), (prime_controls[:,1]), 
                                         (partner_controls[:,0]), (partner_controls[:,1])):
        if np.isnan(idx11) == True:
            idx11_id.append( np.nan )
            idx11_z.append( np.nan )
            idx11_M.append( np.nan )
            idx11_LX.append( np.nan )
        else:
            idx11_id.append( iso_idx[idx11] )
            idx11_z.append( all_df.loc[iso_idx[idx11], 'drawn_z'] )
            idx11_M.append( all_df.loc[iso_idx[idx11], 'drawn_M'] )
            idx11_LX.append( all_df.loc[iso_idx[idx11], 'drawn_LX'] )
            
        if np.isnan(idx12) == True:
            idx12_id.append( np.nan )
            idx12_z.append( np.nan )
            idx12_M.append( np.nan )
            idx12_LX.append( np.nan )
        else:
            idx12_id.append( iso_idx[idx12] )
            idx12_z.append( all_df.loc[iso_idx[idx12], 'drawn_z'] )
            idx12_M.append( all_df.loc[iso_idx[idx12], 'drawn_M'] )
            idx12_LX.append( all_df.loc[iso_idx[idx12], 'drawn_LX'] )
            
        if np.isnan(idx21) == True:
            idx21_id.append( np.nan )
            idx21_z.append( np.nan )
            idx21_M.append( np.nan )
            idx21_LX.append( np.nan )
        else:
            idx21_id.append( iso_idx[idx21] )
            idx21_z.append( all_df.loc[iso_idx[idx21], 'drawn_z'] )
            idx21_M.append( all_df.loc[iso_idx[idx21], 'drawn_M'] )
            idx21_LX.append( all_df.loc[iso_idx[idx21], 'drawn_LX'] )
            
        if np.isnan(idx22) == True:
            idx22_id.append( np.nan )
            idx22_z.append( np.nan )
            idx22_M.append( np.nan )
            idx22_LX.append( np.nan )
        else:
            idx22_id.append( iso_idx[idx22] )
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
    
    true_pairs['field'] = [field]*len(true_pairs)
    
    return true_pairs
    
    
    
# -------------------------------------------------------------------------------------------------------------------------- #

def get_control(control_mass, control_z, mass, redshift, N_control=2, zfactor=0.1, mfactor=2): 

    dz = zfactor

    # create list to store lists of control indices per prime galaxy
    control_all = []
        
    # create a list for all ID's to make sure there are no duplicates
    control_dup = []
    
    # create a dataframe from the isolated galaxy data
    iso = {'z':control_z, 'mass':control_mass}
    all_iso_df = pd.DataFrame( iso )
    
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
        cmatch_df['zdif'] = np.abs(cmatch_df['z'] - z)
        cmatch_df['massdif'] = np.abs(cmatch_df['mass'] - m)
                
        # need to sort dataframe based on two columns THEN continue
        cmatch_df.sort_values(by=['zdif','massdif'], inplace=True, ascending = [True, True])

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
    main()

    
    
# NOTES
# need Dale to clarify a few things:
#   the data from zcat are for the mFDa4 catalog, but the HB4 is best from what I understand, so this is no good
#   will need to choose which PDF's to do the analysis for
#   right now, zlo and zhi are the 68.3% confidence intervals for the mFDa4 method...
#   now what are these photflags?
#   what are the q_specz flags?