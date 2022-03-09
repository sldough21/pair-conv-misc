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
    all_field_df = pool.map(z_cuts, all_fields)  # process data_inputs iterable with pool

    print('Combining all fields...')
    all_fields_df = pd.concat(all_field_df)
    print(all_fields_df)

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
      

def z_cuts(field):

    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from time import sleep
    from tqdm import tqdm

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
    
    pbar = tqdm(total=len(g_df['ID']), position=pos)

    for i in (range(0, len(g_df['ID']))):
    #for i in range(0,100):
     pbar.update(1)
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
       dist_part.append(S_dist[matches[0][j]]/R_kpc)
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
    

    return(pair_df)



if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    main()
