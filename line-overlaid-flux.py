from astro_da.xmmrgs_lines import *
from sys import argv
import os

def main():
	flux_filename = str(argv[1])
	
	#os.environ['ASTRODAT'] = os.environ['HOME']+'/AstroDA/data-files'
	os.environ['ASTRODAT'] = os.environ['HOME']+'/astro-data'
	print('You have chosen: '+flux_filename)
	print('Using ion list: '+os.environ['ASTRODAT']+'/ion.list')
	print('Using NIST ground levels: '+os.environ['ASTRODAT']+'/nist_5-40_gnd.csv')
	
	mission_name = 'XMM-Newton'
	instrument_name = 'RGS'
	observation_ID = '0111150101'
	source_name = 'RX J0925.7-4758'
	
	[angstrom, flux, error] = get_flux(flux_filename)
	preview_flux(angstrom, flux)
	[low_wl, high_wl] = get_wave_limits()
	all_lines = get_lines(low_wl, high_wl)		# Comment this to remove filter
	# all_lines = get_lines(low_wl, high_wl, False)	# Uncomment this to remove filter
	plot_spectrum(angstrom, flux, error, all_lines, low_wl, high_wl)

if __name__ == "__main__":
	main()
