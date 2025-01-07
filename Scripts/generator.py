#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Libraries
import os
import numpy as np
import tensorflow as tf
from scipy.signal import savgol_filter
from scipy.special import wofz

# In[6]:
# Normalize each baseline to have values between 0 and 1, we can then scale them to tune the signal
def normalize(x, eps = 0.00000000001):
    # use nan to num to avoid division by 0 if baseline is flat at 0
    return np.nan_to_num((x - np.min(x))/(np.max(x) - np.min(x) + eps))
# Generate simple, smooth baselines (provide reference to original authors)
def make_baselines(length,
                   sin_max_periodicity = 8,
                   pol_max_degree = 10,
                   eps = 0.00000001):
    
    sin_scaler = np.random.random_sample()
    pol_scaler = np.random.random_sample()
    gaussian_scaler = np.random.random_sample()
    div_scaler = np.random.random_sample()
    
    
    
    # As Ion advised, optimize the parameters for a vector x whose elements are interpolated in [0, 1]
    # Initialize the x-axis, x will be [0, 1] with "length" elements
    x = np.linspace(0, 1, length)
    
    # Linear_baseline
    (first_point, last_point) = np.random.random_sample(2)
    lin_baseline = np.linspace(first_point, last_point, length)
    
    """
    Reasons to keep this as is:
     1. Guarantees that the array elements are within the interval [0, 1]
     2. No need to normalize it because of 1.
     3. Using a formula for the line (b + mx) means that we need to normalize it to guarantee the [0, 1] intervall
     4. Normalizing it with min max will lead to exclusively 3 different baselines, with slope 1, -1 or 0
     5. We would need to define a different normalization method for the function, which brings it to [0, 1],
        all this would lead to more code which only seeks to give the exact same result as the two lines above
    """
### Sinusoidal baseline formula ### 
    offset = np.random.uniform(0, np.pi*2) # Changes the y-intercept of the sine wave, periodic at 2pi
    
    # Decides the frequency of the wave, 4pi means it will have 4 local optima
    stretch = np.random.uniform(low = 0.1, high= np.pi*sin_max_periodicity, size=(1,)) 
    
    sin_baseline = np.sin(x * stretch + offset)
    sin_baseline = normalize(sin_baseline) * sin_scaler # Normalize and scale the wave intensities
    
    # Avoid guaranteed minimum at 0 with y-offset, subtracting 1 from the maximum value ensures maximum of the final wave don't exceed 1
    y_offset = np.random.uniform(0, 1 - np.max(sin_baseline), size = (1,))
    sin_baseline += y_offset
    
### Polynomial baseline ###
    
    pol_degree = np.random.randint(2, pol_max_degree + 1)
    pol_baseline = np.sum([np.random.uniform(-1, 1, size = (1,)) * np.power( x, i) for i in range(pol_degree + 1)], axis = 0)
    
    # Normalize and scale the polynomial
    pol_baseline = normalize(pol_baseline) * pol_scaler
    
    y_offset = np.random.uniform(0, 1 - np.max(pol_baseline), size = (1,))
    pol_baseline += y_offset

### Gaussian baseline ###
    # Gaussian baseline from the Gaussian function: a * e^(-(x - b)^2 * 1/(2c^2))
    # Parameters are optimized for the interval [0, 1]
    a = np.random.uniform(low=-1, high= 1, size=(1,)) # This scales the amplitude of the curve, 0 would result in a flat baseline
    b = np.random.uniform(low=0, high= 1, size=(1,)) # This provides an offset of the curve
    # Limit b to offset the center of the curve to the ends of the spectrum
    if np.random.random_sample() > 0.5:
        b = np.random.uniform(low=0.8, high= 1, size=(1,))
    else:
        b = np.random.uniform(low=0, high= 0.2, size=(1,))
        
    c = np.random.uniform(low=0.3, high= 1, size=(1,)) # This affects the sharpness of the curve, if close to 0, it starts resembling 
                                                       #a band, and so we set a minimum of at least 0.3

    # Given from original article adn included here, but appears to require specific parameters to get unique shape
    gaussian_baseline = a * np.exp(-np.power(x - b, 2) / (2 * c**2) )
    
    # Normalize the baseline and scale it with the scaler
    gaussian_baseline = normalize(gaussian_baseline) * gaussian_scaler
    
    # Add a y-offset to remove guaranteed minimum of 0
    y_offset = np.random.random_sample() * (1 - np.max(gaussian_baseline))
    gaussian_baseline += y_offset

    
### Lorentz-baseline ###
    # Parameters for div-baselines, optimized for the interval [0, 1]
    # Special case for a: when a is close to 0, the lorentzian peak reaches infinite height
    # Randomly set a to negative, this avoids the edge case: a = 0 but allows a to be negative
    a = np.random.uniform(low=0.1, high= 0.5, size=(1,))
    if np.random.random_sample() > 0.5:
        a *= -1
        
    # Limit b to offset the center of the curve to the ends of the spectrum
    if np.random.random_sample() > 0.5:
        b = np.random.uniform(low=0.8, high= 1, size=(1,))
    else:
        b = np.random.uniform(low=0, high= 0.2, size=(1,))
    
    # add small number to avoid division by 0, fromula: https://mathworld.wolfram.com/LorentzianFunction.html
    lorentzian_baseline = (1/np.pi) * a/((a**2 + np.power(x  - b, 2)) + eps)
    lorentzian_baseline = normalize(lorentzian_baseline) * div_scaler
    
    y_offset = np.random.random_sample() * (1 - np.max(lorentzian_baseline))
    lorentzian_baseline = lorentzian_baseline + y_offset

    baselines = [lin_baseline, sin_baseline, pol_baseline, gaussian_baseline, lorentzian_baseline]
    return baselines


def rbf(x, epsilon = 1):
    
    # Definition of the gaussial RBF function
    return np.exp( -np.power((x * epsilon), 2))

def lorentz(x, epsilon):
    
    curve = (1/np.pi) * epsilon/(np.power(epsilon, 2) + np.power(x, 2)) 
    curve = curve - np.min(curve) # Make the ends of th curve touch 0
    curve = curve/np.max(curve) # Set maximum to 1                             
    return curve

def voigt(x, sigma, gamma):
    z = np.vectorize(complex)(x, gamma)/(sigma * np.sqrt(2))
    z_p = wofz(z).real
    
    ret = z_p /(sigma * np.sqrt(2 * np.pi))
    
    return normalize(ret)
    
# Return a normal distribution with random standard deviation
def get_random_peak(length = 16, extremity = 0.5):
    
    # Define the extremities of the domain for the distribution and how many unique, succeeding samples to take
    line = np.linspace(-extremity, extremity, length)
    rbf_epsilon = np.random.randint(5, 10)
    lorentz_epsilon = np.random.uniform(0.1, 0.3) # Interval chosen visually, for sharp curves in interval [-0.5, 0.5]
    voigt_sigma = np.random.uniform(0.01, 0.3)
    voigt_gamma = np.random.uniform(0, 0.1)
    # Get the distribution, it is now a peak in the generator with provided length
    
    rand = np.random.random_sample()

    if rand < 0.33:
        p = rbf(line, rbf_epsilon)
    elif rand < 0.66:
        p = lorentz(line, lorentz_epsilon)
    else:
        p = voigt(line, voigt_sigma, voigt_gamma)

    # Scale the peak randomly to add diversity in terms of band intensities
    return p * np.random.random_sample()

# Generate a spectrum with only comsic rays
def generate_cosmic_rays(length = 1000, max_num = 30):
    
    """
    Generates a line with 0's and spikes at random indices with random intensities
    Expects: "length": (int) The length of the lien
             "max_num": The maximal allowed number of spikes
    Returns: Cosmic_rays (np.array) A vector with comsic rays
    """
    spectrum = np.zeros(length)
    
    # Number of random rays in the spectrum, can be 0
    num_spikes = np.random.randint(max_num)
    
    # Locations of rays
    spike_locs = np.random.randint(length, size = num_spikes)
    
    # Generate intensities, minimum = 0.5, maximum = 1
    spike_intensities = np.random.uniform(low = 0.0, high = 1.0, size=(num_spikes,))
    
    # Set rays onto spectrum
    spectrum[spike_locs] =  spike_intensities
    
    return spectrum

# Generate a line of noise, by far the easiest method
def generate_noise(length = 1000, scaling_factor = 0.1):
    """
    Generate list of noise with given length
    
    Expects: "length": (int) The length of the noise vector
    Returns: Noise (np.array) The noise for the spectrum
    """
    spectrum = np.random.normal(0, np.random.random_sample() * scaling_factor, length)
    spectrum[0] = 0
    spectrum[-1] = 0
    return spectrum


def generate_peaks(length = 1000, max_num_peaks = 10, min_peak_width = 16, max_peak_width = 100, eps = 0.0000001):
    
    """
        This method uses the generate_random_peak method to generate a synthetic spectrum consisting of several randomly placed, gaussian curves which can overlap randomly
    
    
    Expects: "length": (int) The length of the spectrum
             "max_num_peaks": (int) The maximum number of peaks
             "min_peak_width": (int) The minimum width of a generated peak
             "max_peak_width": (int) The maximum width of a generated peak
             "eps": (int) Small real value used to move denominator in division to avoid division by 0
             
    Returns: A spectrum which consists of exclusively of peaks
    """
    
    # Step locations show where the baseline diverges, spikes should not be put on those! 
    # Spikes should only appear between these points, otherwise it is not reasonable to expect that
    # the model will be able to learn the baseline and peaks!
    spectrum = np.zeros(length)
    
    # Get random number of peaks in spectrum, cannot be 0!
    # check edge_case!
    if max_num_peaks < 1:
        num_peaks = 1
    else:
        num_peaks = np.random.randint(1, max_num_peaks)

    # Generate the peaks on the spectrum
    for i in range(num_peaks):
        
        # Get random width of band
        peak_width = np.random.randint(min_peak_width, max_peak_width)
        mid_point = int(peak_width/2)
        
        loc = np.random.randint(mid_point, length - mid_point)

            # Generate pseudo voligt-like peak, will have maximum at approximately 1
        peak = get_random_peak(length = peak_width)
            
            # We add the peak signal to the spectrum here, this way several peaks can combine to a "peak area"
        possible_peak_length = len(spectrum[loc - mid_point : loc - mid_point + peak_width])
        spectrum[loc - mid_point : loc - mid_point + peak_width] += peak[:possible_peak_length]

    # As a precaution, set the extremities to 0, signal should not be there!
    spectrum[0] = 0
    spectrum[-1]= 0
    
    # Before returning, we normalize the whole spectrum to range [0, 1]. The additive nature means that the peaks can
    # exceed this range, we can now reliably scale the peaks after generating outside this method
    spectrum = spectrum/np.max(spectrum + eps)
    
    
    # Return the peaks
    return spectrum

def generate_spectrum(length, min_width = 16, max_width = 256, num_peaks = 10):
    
    # Baselines
    bls = make_baselines(length) 
    # make random combination of baselines
    summed_bl = np.zeros(length)
    bl_mask = np.random.randint(2, size = (len(bls),))
    
    for m, b in zip(bl_mask, bls):
            
        summed_bl += m * b
    
    # Cosmic rays, ray occurence frequency
    rays = generate_cosmic_rays(length)
    
    # Noise
    noise = generate_noise(length)
    
    # Peaks
    
    if length < 500:
        min_width = int(length/32)
        max_width = int(length/2)
        
    else:
        min_width = min_width 
        max_width = max_width 
    peaks = generate_peaks(length = length,
                           max_num_peaks = num_peaks, 
                           min_peak_width = min_width,
                           max_peak_width = max_width)

    
    
    return summed_bl, rays, noise, peaks
        
# The training generator
def train_generator(length = 2000, batch_size = 64, min_width = 5, max_width = 256, max_num_peaks = 10, increase_noise = False):
    """
    Provide the length and batch size to generate a batch for the CNN model
    """
    
    data = []
    target = []
    
    t_bl = []
    t_ray = []
    t_noise = []
    t_peak = []
    
    while True:
            
        bl, ray, noise, peak = generate_spectrum(length, min_width, max_width, max_num_peaks)
        
        scalers = np.random.randint(low = 0.01, high = 10.0, size = (3,))
        scalers = scalers/np.sum(scalers)
        bl = bl * scalers[0]
        ray = ray * scalers[1]
        peak = peak* scalers[2]
        # The spectrum is the sum of the 4 components (Noise is added later)
        s = bl + ray + peak

        
        # Ignore noise signal at random, do not divide noise by maximum
        if not increase_noise:
            
            maxim = np.max(s)
            bl = bl/maxim
            ray = ray/maxim
            peak = peak/maxim
            
            if np.random.random_sample() < 0.05: noise = np.zeros(length)
            noise *= scalers[2] # Use same scale as the peaks component

            s = s + noise
            
        else:

            noise *= np.random.randint(low = 3.0, high = 10.0)# Increase the scale of the noise i.e. decrease SNR
            noise *= scalers[2] # Use same scale as the peaks component
            # Re-normalize the values since the noise may escape the maximum of 1

            s = s + noise
            minimum = np.min(s) 
            s = s - minimum # Drag spectrum back to minimum of 0
            bl = bl - minimum # Apply the shift also to the baseline component
        
            maxim = np.max(s)
            s = s/maxim
            bl = bl/maxim
            ray = ray/maxim
            peak = peak/maxim
            noise = noise/maxim
        
        data.append(s)
        t_bl.append(bl)
        t_ray.append(ray)
        t_noise.append(noise)
        t_peak.append(peak)
        
        if len(data) == batch_size:

            data = np.nan_to_num(np.array(data))
            t_bl = np.nan_to_num(np.array(t_bl))
            t_ray = np.nan_to_num(np.array(t_ray))
            t_noise = np.nan_to_num(np.array(t_noise))
            t_peak = np.nan_to_num(np.array(t_peak))
            
            data = tf.keras.backend.expand_dims(data)
            
            target = [t_bl, t_ray, t_noise, t_peak]
            
            yield data, target
            
            data = []
            t_bl = []
            t_ray = []
            t_noise = []
            t_peak = []
            target = []