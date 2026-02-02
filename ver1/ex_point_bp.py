import numpy as np
from scipy.constants import speed_of_light, pi
import backprojection as bp
from matplotlib import pyplot as plt
from scipy.constants import c, pi
from numpy import sqrt, linspace, zeros_like, exp, sin, cos, ones
from scipy.interpolate import interp1d
from scipy.fftpack import ifft, fftshift

def reconstruct(signal, sensor_x, sensor_y, sensor_z, range_center, x_image, y_image, z_image, frequency, fft_length):
    """
    Reconstruct the two-dimensional image using the filtered backprojection method.
    :param signal: The signal in K-space.
    :param sensor_x: The sensor x-coordinate (m).
    :param sensor_y: The sensor y-coordinate (m).
    :param sensor_z: The sensor z-coordinate (m).
    :param range_center: The range to the center of the image (m).
    :param x_image: The x-coordinates of the image (m).
    :param y_image: The y-coordinates of the image (m).
    :param z_image: The z-coordinates of the image (m).
    :param frequency: The frequency array (Hz).
    :param fft_length: The number of points in the FFT.
    :return: The reconstructed image.
    """
    # Get the frequency step size
    frequency_step = frequency[1] - frequency[0]

    # Calculate the maximum scene size and resolution
    range_extent = c / (2.0 * frequency_step)

    # Calculate the range window for the pulses
    range_window = linspace(-0.5 * range_extent, 0.5 * range_extent, fft_length)

    # Initialize the image
    bp_image = zeros_like(x_image, dtype=complex)

    # Loop over all pulses in the data
    term = 1j * 4.0 * pi * frequency[0] / c

    # To work with stripmap
    if not isinstance(range_center, list):
        range_center *= ones(len(sensor_x))

    index = 0
    for xs, ys, zs in zip(sensor_x, sensor_y, sensor_z):

        # Calculate the range profile
        range_profile = fftshift(ifft(signal[:, index], fft_length))

        # Create the interpolation for this pulse
        f = interp1d(range_window, range_profile, kind='linear', bounds_error=False, fill_value=0.0)

        # Calculate the range to each pixel
        range_image = sqrt((xs - x_image) ** 2 + (ys - y_image) ** 2 + (zs - z_image) ** 2) - range_center[index]

        # Interpolate the range profile onto the image grid and multiply by the range phase
        # For large scenes, should check the range window and index
        bp_image += f(range_image) * exp(term * range_image)

        index += 1

    return bp_image

# **Set the range to the image center (m)**
range_center = 1000

# **Set the point target locations (m) and RCS (m<sup>2</sup>)**
x_target = [3, 0, -3]
y_target = [-3, 0, 3]
rcs = [10, 10, 20]

# **Set the image span (m)**
x_span = 20
y_span = 20

# **Set the number of bins in the image**
nx = 500 # Number of bins in x-direction
ny = 500 # Number of bins in y-direction

# **Set the start frequency and bandwidth (Hz)**
start_frequency = 5e9 
bandwidth = 300e6 

# **Set the azimuth span of the synthetic apreture (deg)**
az_start = -3
az_end = 3

# **Set the window type for the image**
window_type = 'Hanning'

# **Set the dynamic range for the image plot (dB)**
dynamic_range = 50

# **Set up the azimuth space**
r = np.sqrt(x_span**2 + y_span**2)
da = speed_of_light / (2.0 * r * start_frequency)
na = round((az_end - az_start) / da)
az = np.linspace(az_start, az_end, int(na))

# **Set up the frequency space**
df = speed_of_light / (2.0 * r)
nf = np.floor(bandwidth / df)
frequency = np.linspace(start_frequency, start_frequency + bandwidth, int(nf))

# **Set the length of the FFT**
fft_length = int(8 * 2**np.ceil(np.log2(nf)))

# **Set up the aperture positions**
sensor_x = range_center * np.cos(np.radians(az))
sensor_y = range_center * np.sin(np.radians(az))
sensor_z = np.zeros_like(sensor_x)

# **Set up the image space**
xi = np.linspace(-0.5 * x_span, 0.5 * x_span, nx)
yi = np.linspace(-0.5 * y_span, 0.5 * y_span, ny)
[x_image, y_image] =np. meshgrid(xi, yi)
z_image = np.zeros_like(x_image)

# **Calculate the signal in wavenumber space**
# Initialize the signal
signal = np.zeros((int(nf), int(na)), dtype=complex)

# Short hand and helps with computational load
ca = np.cos(np.radians(az))
sa = np.sin(np.radians(az))

# Calculate the wavenumber (rad/m)
kc = 2 * pi * frequency / speed_of_light

# Loop over all azimuth angles
i = 0
for a, c, s in zip(az, ca, sa):    
    r_los = [c, s]    
    
    for xt, yt, rt in zip(x_target, y_target, rcs):    
        r_target = np.dot(r_los, [xt, yt])        
        signal[:, i] += rt * np.exp(1j * 2.0 * kc * r_target)        
    i += 1
    
# **Get the window coefficients**
if window_type == 'Hanning':
    coefficients = np.outer(np.hanning(nf), np.hanning(na))
elif window_type == 'Hamming':
    coefficients = np.outer(np.hamming(nf), np.hamming(na))
else:
    coefficients = np.ones_like(signal)
    
# **Apply the selected window**
signal = signal * coefficients

# **Reconstruct the image**
bp_image = reconstruct(signal, sensor_x, sensor_y, sensor_z, range_center, x_image, y_image, z_image, frequency, fft_length)