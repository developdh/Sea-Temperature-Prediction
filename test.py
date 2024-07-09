# velocity_data_19711990 = xr.open_dataset('data/vo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_197101-199012.nc')
# print(velocity_data_19711990)


from matplotlib import pyplot as plt # import libraries
import pandas as pd # import libraries
import netCDF4 # import libraries
fp='data/vo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_197101-199012.nc'
nc = netCDF4.Dataset(fp) # reading the nc file and creating Dataset
""" in this dataset each component will be 
in the form nt,nz,ny,nx i.e. all the variables will be flipped. """
plt.imshow(nc['Temp'][1,:,0,:]) 
""" imshow is a 2D plot function
according to what I have said before this will plot the second
iteration of the vertical slize with y = 0, one of the vertical
boundaries of your model. """
plt.show() # this shows the plot