import numpy as np
import astropy.constants as cons

class Image:
    
    '''Class for loading and manipulating images from .npz files.'''
    h_ev = 4.135667662e-15
    eV_2_erg = 1.60218e-12
    h_erg = 6.626e-27
    kB=cons.k_B.cgs.value
    c = 2.99792458e10
    gg_msun = 1.32712440018e26


    def __init__(self,filename):
        '''Initialize the Image object by loading the .npz file and reading metadata.'''
        self.filename = filename
        if self.filename[-4:]=='.npz':
            try:
                self.file = np.load(self.filename)
            except:
                raise RuntimeError('File not found or otherwise unable to be read by numpy.')
        else:
            raise NotImplementedError('File format not supported yet. Sorry!')
        
        self.frequencies = self.file['frequency'][:]
        self.mass_msun = self.file['mass_msun'][0]
        self.width_rg = self.file['width'][0]
        self.distance = self.file['distance'][0]
        try:
            self.file['Q_nu']
            self.polarization = True
        except KeyError:
            self.polarization = False
        
        try:
            self.file['sigma_I']
            self.error = True
        except KeyError:
            self.error = False
        

        self.adaptive_num_levels = self.file['adaptive_num_levels'][0]
        
        if self.adaptive_num_levels > 0:
            self.adaptive_num_blocks = {}
            self.block_locs = {}
            self.adaptive_num_blocks[0] = self.file['adaptive_num_blocks'][0]
            
            for level in range(1, self.adaptive_num_levels + 1):
                self.adaptive_num_blocks[level] = self.file['adaptive_num_blocks'][level]
                self.block_locs[level] = self.file['adaptive_block_locs_{0}'.format(level)][:]
        self.image = None
        
        
    def load_image(self):
        '''Load the stokes parameters from the .npz file and store them in a dictionary.'''
        if self.image is not None:
            return self.image
        self.image = {}
        if self.adaptive_num_levels > 0:
            for level in range(1, self.adaptive_num_levels + 1):
                if self.polarization:
                    key_i = 'adaptive_I_nu_{0}'.format(level)
                    key_q = 'adaptive_Q_nu_{0}'.format(level)
                    key_u = 'adaptive_U_nu_{0}'.format(level)
                    key_v = 'adaptive_V_nu_{0}'.format(level)
                    i_nu = self.file[key_i][:]
                    q_nu = self.file[key_q][:]
                    u_nu = self.file[key_u][:]
                    v_nu = self.file[key_v][:]
                    self.image[level] = np.vstack((i_nu[None,:,:,:], q_nu[None,:,:,:], u_nu[None,:,:,:], v_nu[None,:,:,:]))
                else:
                    key_i = 'adaptive_I_nu_{0}'.format(level)
                    i_nu = self.file[key_i][:]
                    q_nu = np.zeros(i_nu.shape)*np.nan
                    u_nu = np.zeros(i_nu.shape)*np.nan
                    v_nu = np.zeros(i_nu.shape)*np.nan
                    self.image[level] = np.vstack((i_nu[None,:,:,:], q_nu[None,:,:,:], u_nu[None,:,:,:], v_nu[None,:,:,:]))
        
        if self.polarization:
                i_nu = self.file['I_nu'][:]
                q_nu = self.file['Q_nu'][:]
                u_nu = self.file['U_nu'][:]
                v_nu = self.file['V_nu'][:]
                if self.error:
                    sigma_I = self.file['sigma_I'][:]
                else:
                    sigma_I = np.zeros(i_nu.shape)*np.nan
                self.image[0] = np.vstack((i_nu[None,:,:], q_nu[None,:,:], u_nu[None,:,:], v_nu[None,:,:],sigma_I[None,:,:]))
        else:
                i_nu = self.file['I_nu'][:]
                q_nu = np.zeros(i_nu.shape)*np.nan
                u_nu = np.zeros(i_nu.shape)*np.nan
                v_nu = np.zeros(i_nu.shape)*np.nan
                if self.error:
                    sigma_I = self.file['sigma_I'][:]
                else:
                    sigma_I = np.zeros(i_nu.shape)*np.nan
                self.image[0] = np.vstack((i_nu[None,:,:], q_nu[None,:,:], u_nu[None,:,:], v_nu[None,:,:],sigma_I[None,:,:]))
                #print("in this loop",self.image[0])
        
                
    
    def get_I(self, level=0):
        '''Get the intensity image for a given level. If the image has not been loaded yet, it will be loaded first.'''
        self.load_image()
        I = {}
        I[0] = self.image[0][0,:,:,:]
        for level in range(1,level+1):
            I[level] = self.image[level][0,:,:,:]
        return I

    def get_Q(self, level=0):
        '''Get the Q Stokes parameter for a given level. If the image has not been loaded yet, it will be loaded first.'''
        image = self.load_image()
        if self.polarization:
            I = {}
            I[0] = self.image[0][1,:,:,:]
            for level in range(1,level+1):
                I[level] = self.image[level][1,:,:,:]
            return I
        else:
            raise RuntimeError('No polarization data in file.')
    
    def get_U(self, level=0):
        '''Get the U Stokes parameter for a given level. If the image has not been loaded yet, it will be loaded first.'''
        image = self.load_image()
        if self.polarization:
            I = {}
            I[0] = self.image[0][2,:,:,:]
            for level in range(1,level+1):
                I[level] = self.image[level][2,:,:,:]
            return I
        else:
            raise RuntimeError('No polarization data in file.')
    
    def get_V(self, level=0):
        '''Get the V Stokes parameter for a given level. If the image has not been loaded yet, it will be loaded first.'''
        image = self.load_image()
        if self.polarization:
            I = {}
            I[0] = self.image[0][3,:,:,:]
            for level in range(1,level+1):
                I[level] = self.image[level][3,:,:,:]
            return I
        else:
            raise RuntimeError('No polarization data in file.')
    
    def get_sigmaI(self,level=0):
        image = self.load_image()
        if self.error:
            I = {}
            I[0] = self.image[0][4,:,:,:]
            for level in range(1,level+1):
                I[level] = self.image[level][4,:,:,:]
            return I
        else:
            I = {}
            I[0] = np.zeros(np.array(self.get_I(level)[0]).shape)
            return I
    
    def get_Alternate_Image(self, image_name, level=0):
        '''Get an alternate image (eg. optical depth)'''
        try:
            alt_image = {}
            # Always try to load the root image first
            try:
                alt_image[0] = self.file[image_name][:]
            except KeyError:
                alt_image[0] = self.file[f'{image_name}_0'][:]

            if self.adaptive_num_levels > 0:
                for l_idx in range(1, self.adaptive_num_levels + 1):
                    key = f'adaptive_{image_name}_{l_idx}'
                    alt_image[l_idx] = self.file[key][:]
            
            return alt_image[level]
        except Exception as e:
            raise RuntimeError(f"{image_name} level {level} not found. Error: {e}")
    
    def get_flux(self):
        '''Calculate the spatially-averaged flux from the image.
        
        Inputs: 
        - distance: distance to the source in gravitational radii
        Outputs:
        - flux in erg/s/cm^2/Hz
        '''
        if self.adaptive_num_levels > 0:
            raise UserWarning('The ability to get adaptive-resolution images flux is still being developed!')
            # Preserve original state
            original_image = self.image
            original_block_locs = self.block_locs

            # Load image without permanently mutating self
            self.load_image()

            try:
                # Work on local copies instead of self.*
                image_levels = [None] * len(self.image)
                block_locs = [None] * len(self.block_locs)

                # Copy loaded adaptive levels
                for i in range(1, len(self.image)):
                    image_levels[i] = self.image[i]

                for i in range(1, len(self.block_locs)):
                    block_locs[i] = self.block_locs[i]
                root_image = self.get_I()[0]
                # Disassemble root image into blocks
                block_res = image_levels[1].shape[-1]
                num_blocks_root_linear = root_image.shape[-1] // block_res

                image_levels[0] = np.reshape(
                    root_image,
                    (-1,
                    num_blocks_root_linear,
                    block_res,
                    num_blocks_root_linear,
                    block_res)
                )

                image_levels[0] = np.swapaxes(image_levels[0], 2, 3)

                image_levels[0] = np.reshape(
                    image_levels[0],
                    (-1,
                    self.adaptive_num_blocks[0],
                    block_res,
                    block_res)
                )

                # Describe disassembled image locations
                block_locs[0] = np.empty(
                    (self.adaptive_num_blocks[0], 2),
                    dtype=int
                )

                for block in range(self.adaptive_num_blocks[0]):
                    block_locs[0][block, 1] = block % num_blocks_root_linear
                    block_locs[0][block, 0] = block // num_blocks_root_linear

                # Prepare flags indicating blocks to be counted
                flags = {}
                for l in range(self.adaptive_num_levels + 1):
                    flags[l] = [True] * self.adaptive_num_blocks[l]

                # Determine which blocks are masked by those at higher levels
                for l in range(1, self.adaptive_num_levels + 1):
                    for block_fine in range(self.adaptive_num_blocks[l]):
                        x_loc_fine = block_locs[l][block_fine, 1]
                        y_loc_fine = block_locs[l][block_fine, 0]

                        x_loc_coarse = x_loc_fine // 2
                        y_loc_coarse = y_loc_fine // 2

                        for block_coarse in range(self.adaptive_num_blocks[l - 1]):
                            x_loc_coarse_test = block_locs[l - 1][block_coarse, 1]
                            y_loc_coarse_test = block_locs[l - 1][block_coarse, 0]

                            if (
                                x_loc_coarse == x_loc_coarse_test and
                                y_loc_coarse == y_loc_coarse_test
                            ):
                                flags[l - 1][block_coarse] = False

                # Prepare fluxes
                if self.polarization:
                    flux = np.zeros(4)
                else:
                    flux = np.zeros(1)

                nan_found = False

                # Calculate flux from non-masked blocks
                block_width_root = self.width_rg / num_blocks_root_linear

                for l in range(self.adaptive_num_levels + 1):
                    block_width = block_width_root / (2 ** l)

                    for block in range(self.adaptive_num_blocks[l]):
                        if flags[l][block]:

                            if np.any(np.isnan(image_levels[l][:, block, :, :])):
                                nan_found = True

                            flux += (
                                np.nanmean(
                                    image_levels[l][:, block, :, :],
                                    axis=(1, 2)
                                )
                                * block_width ** 2
                            )

                if nan_found:
                    print('Warning: ignoring NaN')
                    print('')

            finally:
                # Restore original object state
                self.image = original_image
                self.block_locs = original_block_locs

        else:
            image = self.get_I()[0]
            error = self.get_sigmaI()[0]
            image_width = 2 * np.arctan(0.5 * self.width_rg / self.distance)

            flux = np.array([])
            flux_err = np.array([])

            for freq in range(len(self.frequencies)):
                tempImage = np.copy(image[freq, :, :])
                tempImageErr = np.copy(error[freq, :, :])
                
                flux = np.append(flux,np.nanmean(tempImage[np.isfinite(tempImage)]) * image_width ** 2)
                #print(np.sqrt(np.nansum(tempImageErr[np.isfinite(tempImageErr)]**2)) * (image_width ** 2)/np.shape(tempImageErr[np.isfinite(tempImageErr)])[0],np.nanmean(tempImageErr[np.isfinite(tempImageErr)]**2)* (image_width ** 2))
                flux_err = np.append(flux_err,np.sqrt(np.nansum(tempImageErr[np.isfinite(tempImageErr)]**2)) * (image_width ** 2)/np.shape(tempImageErr[np.isfinite(tempImageErr)])[0])

        return flux,flux_err

    def get_luminosity(self):
        '''Calculate the luminosity from the flux and distance.

        Inputs: 
        - distance: distance to the source in gravitational radii
        Outputs:
        - luminosity in erg/s/Hz
        '''
        flux,flux_err = self.get_flux()
        rg = self.gg_msun * self.mass_msun / (self.c ** 2)
        luminosity = 2*((rg*self.distance)**2)*flux
        return luminosity,2*((rg*self.distance)**2)*flux_err


        
        

        
        
