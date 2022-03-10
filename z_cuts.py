# Sean Dougherty
# 3/7/2022
# replication of zcuts

def main():
    print('Initializing code...')
        
    # from multiprocessing import Pool
    from multiprocessing import Pool, freeze_support, RLock

    # load tables for each one
    
    # initialize dictionary
    all_fields = ['GDS','EGS','COS','GDN','UDS']
    
    #fields = [all_fields['GDS'],all_fields['EGS'],all_fields['COS'],all_fields['GDN'],all_fields['UDS']]

    pool = Pool()                         # Create a multiprocessing Pool
    all_data = pool.map(z_cuts, all_fields)  # process data_inputs iterable with pool
    all_pair = []
    all_iso = []
    for i in range(0,len(all_data)):
     all_pair.append(all_data[i][0])
     all_iso.append(all_data[i][1])
    
    print('Combining all fields...')
    all_pair_df = pd.concat(all_pair)
    all_iso_df = pd.concat(all_iso)
    print(all_iso_df)
    
    print('Calculating AGN fractions...')

    #lis = np.array(all_fields_df['dv_draws'])
    #print(len(lis[0]))
    #print(all_fields_df)
    AGNfracs, isofracs = pair_analysis(all_pair_df, all_iso_df)
    print('Done!')

    #distr_plots(AGNfracs)

    AGNfrac_df = pd.DataFrame.from_dict(AGNfracs)
    isofrac_df = pd.DataFrame.from_dict(isofracs)

    print(AGNfrac_df)
    print(isofrac_df)

    # calculate AGN enhancements:
    AGNfrac_df.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/Data_CSV/pair_AGN_frac.csv')
    isofrac_df.to_csv('/nobackup/c1029594/CANDELS_AGN_merger_data/Data_CSV/iso_AGN_frac.csv')




# -------------------------------------------------------------------------------------------------------------------------- #



def draw_z(field, pair_df):

    from numpy import random

    print('Running draw_z for {}...'.format(field))

    all_draw1 = []
    all_draw2 = []
    all_dv = []

    for i in range(0, len(pair_df['ID'])):
     ID_str = pair_df['ID'][i]
     ID_str2 = pair_df['p_ID'][i]
     if len(str(ID_str)) == 1: id_string = '0000'+str(ID_str)
     if len(str(ID_str)) == 2: id_string = '000'+str(ID_str)
     if len(str(ID_str)) == 3: id_string = '00'+str(ID_str)
     if len(str(ID_str)) == 4: id_string = '0'+str(ID_str)
     if len(str(ID_str)) == 5: id_string = str(ID_str)

     if len(str(ID_str2)) == 1: id_string2 = '0000'+str(ID_str2)
     if len(str(ID_str2)) == 2: id_string2 = '000'+str(ID_str2)
     if len(str(ID_str2)) == 3: id_string2 = '00'+str(ID_str2)
     if len(str(ID_str2)) == 4: id_string2 = '0'+str(ID_str2)
     if len(str(ID_str2)) == 5: id_string2 = str(ID_str2)

     if field == "GDS":
      pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSS_ID'+id_string+'.pzd'
      pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSS_ID'+id_string2+'.pzd'
     elif field == "EGS":
      pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/EGS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_EGS_ID'+id_string+'.pzd'
      pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/EGS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_EGS_ID'+id_string2+'.pzd'
     elif field == "GDN":
      pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSN_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSN_ID'+id_string+'.pzd'
      pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/GOODSN_OPTIMIZED03/ALL_OPTIMIZED_PDFS_GOODSN_ID'+id_string2+'.pzd'
     elif field == "COS":
      pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/COSMOS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_COSMOS_ID'+id_string+'.pzd'
      pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/COSMOS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_COSMOS_ID'+id_string2+'.pzd'
     elif field == "UDS":
      pdf_filename1 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/UDS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_UDS_ID'+id_string+'.pzd'
      pdf_filename2 = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data - All Fields/UDS_OPTIMIZED03/ALL_OPTIMIZED_PDFS_UDS_ID'+id_string2+'.pzd' 

     pdf1 = pd.read_csv(pdf_filename1, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                  'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')
     pdf2 = pd.read_csv(pdf_filename2, comment='#', names=['z', 'Finkelstein', 'Fontana', 'Pforr', 'Salvato', 'Wiklind',
                                                  'Wuyts', 'HB4', 'mFDa4'], delimiter=' ')

     # draw the samples
     n = 2000 # number of draws
     sum1 = np.sum(pdf1['HB4'])
     sum2 = np.sum(pdf2['HB4'])
     
     if pair_df['p_zspec'][i] > 0: draw1 = [pair_df['p_zspec'][i]]*n
     elif pair_df['p_zspec'][i] > 0: draw1 = [pair_df['p_zspec'][i]]*n
     else:
      draw1 = random.choice(pdf1['z'], size=n, p=(pdf1['HB4']/sum1))
      draw2 = random.choice(pdf2['z'], size=n, p=(pdf2['HB4']/sum2))
     dv_list = []
     for z1, z2 in zip(draw1, draw2):
      dv = ( (((z2+1)**2 -1)/((z2+1)**2 +1)) - (((z1+1)**2 -1)/((z1+1)**2 +1)) ) * 2.998e5
      dv_list.append(dv)
     all_dv.append(dv_list)
      # think about what else you can calculate here...
      # mass - in Duncan -> use their own SED fitting code... could be available but issues with data consistency if we use.
      # separation - can only calculate projected separation in reality
    
    pair_df['dv_draws'] = all_dv
    return pair_df
      

# -------------------------------------------------------------------------------------------------------------------------- #



def z_cuts(field):

    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    print('Running z_cuts...')
    
    PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/Data_CSV/'

    df_1 = pd.read_csv(PATH+'CANDELS_'+field+'_1018_LX.csv')
    df_2 = pd.read_csv(PATH+'CANDELS_'+field+'_1018_LX_z.csv')
    df = df_1.join(df_2)

    g_df = df[ (df['mass'] >= 9.0) & (df['mass'] <= 30) & (df['zbest'] >= 0.5) & (df['zbest'] <= 3.0) & (df['photflag'] == 0) &
             (df['hmag'] < 24.5) & (df['class_star'] < 0.9) ]
    g_df = g_df.reset_index(drop=True)
	
    # define some constants and initialize some lists
    g_prime = []
    g_partner = []
    dist_part = [] # separation in kpc
    angdist_part = []
    iso = []

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    d_min = 4.0 #kpc
    d_max = 80.0 #kpc
    mratio = np.log10(4)
    mratio3 = np.log10(10)

    if field == 'GDS': pos=1
    elif field == 'EGS': pos=2
    elif field == 'COS': pos=3
    elif field == 'GDN': pos=4
    elif field == 'UDS': pos=5
    
    #pbar = tqdm(total=len(g_df['ID']), position=pos)

    for i in (range(0, len(g_df['ID']))):
    #for i in range(0,100):
     #pbar.update(1)
     R_kpc = cosmo.arcsec_per_kpc_proper(g_df['zbest'][i]) # 1 kpc = arcsec at z
     R_kpc_min = (R_kpc*d_min).value
     R_kpc_max = (R_kpc*d_max).value
     icoord = SkyCoord(g_df['RAdeg'][i], g_df['DEdeg'][i], unit='deg')
     mcoords =  SkyCoord(g_df['RAdeg'], g_df['DEdeg'], unit='deg')
     S_dist = icoord.separation(mcoords)
     S_dist = S_dist.arcsecond
     massrat = g_df['mass'][i] - g_df['mass']
     matches = np.where((S_dist >= R_kpc_min) & (S_dist <= R_kpc_max) & (massrat <= mratio3) & (massrat >= 0))
		
     if len(matches[0]) == 0: iso.append(i)
     if len(matches[0]) > 0:
      for j in range(0, len(matches[0])):
       # g_dist = S_dist[matches[0]].sort() # I guess this was to sort by distance
       g_prime.append(i)
       g_partner.append(matches[0][j])
       dist_part.append((S_dist[matches[0][j]]/R_kpc).value)
       angdist_part.append(S_dist[matches[0][j]])

    #print(' ', field, len(g_prime), len(dist_part))

    # create new dataframe for these pairs
    prime_df = g_df.iloc[g_prime]
    partner_df1 = g_df.iloc[g_partner]
    partner_df = partner_df1.rename(columns={'ID':'p_ID',"RAdeg":"p_RAdeg","DEdeg":"p_DEdeg","mass":"p_mass","hmag":"p_hmag",
        "photflag":"p_photflag","class_star":"p_class_star","flag_xray":"p_flag_xray","zbest":"p_zbest","zspec":"p_zspec",
        "q_zspec":"p_q_zspec"})
    prime_df = prime_df.reset_index(drop=True)
    partner_df = partner_df.reset_index(drop=True)
    pair_df = prime_df.join(partner_df)
    pair_df['dist_kpc'] = dist_part
    pair_df['angdist_arcsec'] = angdist_part
    pair_df['field'] = [field]*len(pair_df['ID'])

    draw_z(field, pair_df)

    # prepare isolated df
    iso_df = g_df.iloc[iso]
    iso_df = iso_df.reset_index(drop=True)
    iso_df['field'] = [field]*len(iso_df['ID']) 

    return pair_df, iso_df

# -------------------------------------------------------------------------------------------------------------------------- #

def pair_analysis(df, iso_df):
    print('in pair analysis now')
    # change constants
    max_merg_dist = 80
    n_bins = 8
    max_dv = 1000

    bin_all_AGNfrac = {}
    bin_all_isofrac = {}

    # create bin sizes
    all_bins = {}
    for i in range(0, n_bins):
      all_bins['bin'+str(i)] = str((max_merg_dist/n_bins)*i)+'-'+str((max_merg_dist/n_bins)+(max_merg_dist/n_bins)*i)
      bin_all_AGNfrac['all_AGNfrac_in_bin'+str(i)] = []
      bin_all_isofrac['all_isofrac_in_bin'+str(i)] = []

    
    lis = np.array(df['dv_draws'])
    for it in tqdm(range(0,len(lis[0]))):
     #print(it)
     
     bin_AGN = {}
     bin_AGNfrac = {}
     bin_isofrac = {}
     # create AGN dictionary for bins:
     for name in all_bins:
      
      info = all_bins[name].split('-')
      low = float(info[0])
      high = float(info[1])

      df['current_dv'] = df.dv_draws.apply(lambda x: x[it])
        
      #print(df['current_dv'])

      bin_df = df[ (df['dist_kpc'] > low) & (df['dist_kpc'] <= high) & (df['current_dv'] < max_dv) ]
      bin_AGN['AGN_in_'+name] = np.concatenate( (np.array(bin_df['flag_xray']), np.array(bin_df['p_flag_xray'])), axis=0)
      bin_AGNfrac['AGNfrac_in_'+name] = sum(bin_AGN['AGN_in_'+name]) / len(bin_AGN['AGN_in_'+name])
      bin_all_AGNfrac['all_AGNfrac_in_'+name].append(bin_AGNfrac['AGNfrac_in_'+name])

      # select control galaxies from bin_df, which holds the dataframe for pairs at the current iteration in a single bin
      control_df = get_control(iso_df, bin_df)
      bin_isofrac['isofrac_in_'+name] = sum(control_df['flag_xray']) / len(control_df['flag_xray'])
      bin_all_isofrac['all_isofrac_in_'+name].append(bin_isofrac['isofrac_in_'+name])

    return bin_all_AGNfrac, bin_all_isofrac


# -------------------------------------------------------------------------------------------------------------------------- #

def get_control(df_iso, df_agn, N_control=2, zfactor=0.2, mfactor=2):
	
    dz = zfactor
	
    # create list to store control selections
    control = []
    control_f = []
    
    m_all = np.concatenate( (np.array(df_agn['mass']), np.array(df_agn['p_mass'])), axis=0 )
    z_all = np.concatenate( (np.array(df_agn['zbest']), np.array(df_agn['p_zbest'])), axis=0 )
    f_all = np.concatenate( (np.array(df_agn['field']), np.array(df_agn['field'])), axis=0 )
    
    for i in range(0, len(m_all)):
     m = m_all[i] # if possible, this is where I would need to keep track of mass(z)
     z = z_all[i] # so this is where I would need to keep track of the drawn z's... just continue as is for now...
     f = f_all[i]
     zmin = z - dz
     zmax = z + dz
     
     # create a dataframe for possible matches
     iso_match = df_iso[ (df_iso['zbest'] >= zmin) & (df_iso['zbest'] <= zmax) & 
     	(df_iso['mass'] >= m-np.log10(mfactor)) & (df_iso['mass'] <= m+np.log10(mfactor)) &
     	(df_iso['field'] == f) ]

     iso_match = iso_match.reset_index(drop=True) 
     
     # randomize df_iso and move through it until we have desired number of control galaxies
     iso_match = iso_match.sample(frac=1).reset_index(drop=True)
     mcount = 0

     for j in range(0, len(iso_match)):
     
      if iso_match['ID'][j] in control and iso_match['field'][j] in control_f:
       continue
      else:
       control.append(iso_match['ID'][j])
       control_f.append(iso_match['field'][j])
       mcount+=1
       
     #if mcount < N_control:
     # print('Not enough control galaxies for object {}!'.format(i))
     if mcount == N_control: 
      break
      
    # return a dataframe of selected control galaxies:
    df_control = df_iso.loc[ (df_iso['ID'].isin(control)) & (df_iso['field'].isin(control_f)) ]
    return df_control


# -------------------------------------------------------------------------------------------------------------------------- #


def distr_plots(all_bin_dict):
    #fig, ax = plt.figure()
    for name in all_bin_dict:
     plt.hist(all_bin_dict[name], density=True, histtype='step', label=name)
    plt.legend()
    plt.show()


# -------------------------------------------------------------------------------------------------------------------------- #



if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from time import sleep
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import math as m

    main()
