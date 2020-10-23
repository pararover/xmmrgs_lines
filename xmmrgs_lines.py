class InvalidInputError(Exception):
	# ABOUT:	A class for catching custom exceptions.
	def __init__(self, data):
		self.data = data
	def __str__(self):
		return repr(self.data)

def write_roman(num):
	# ABOUT:	A function that returns the roman numeral equivalent
	#			for a given integer
	# INPUT:	An integer.
	# OUTPUT:	A string containing the equivalent roman numerals.
	# SYNTAX:	<string> = write_roman(<integer>)
	from collections import OrderedDict
	roman = OrderedDict()
	roman[1000] = "M"
	roman[900] = "CM"
	roman[500] = "D"
	roman[400] = "CD"
	roman[100] = "C"
	roman[90] = "XC"
	roman[50] = "L"
	roman[40] = "XL"
	roman[10] = "X"
	roman[9] = "IX"
	roman[5] = "V"
	roman[4] = "IV"
	roman[1] = "I"
	def roman_num(num):
		for r in roman.keys():
			x, y = divmod(num, r)
			yield roman[r] * x
			num -= (r * x)
			if num <= 0:
				break
	return "".join([a for a in roman_num(num)])

def write_decimal(rom_num):
	# ABOUT:	A function that returns the integer equivalent
	#			for given roman numerals.
	# INPUT:	A string containing the roman numerals.
	# OUTPUT:	An integer containing the equivalent value.
	# SYNTAX:	<integer> = write_decimal(<string>)
	from collections import OrderedDict
	num = OrderedDict()
	num["M"] = 1000
	num["D"] = 500
	num["C"] = 100
	num["L"] = 50
	num["X"] = 10
	num["V"] = 5
	num["I"] = 1

	result = 0
	i = 0
	while i < len(rom_num):
		sym1 = num[rom_num[i]]
		if i+1 < len(rom_num):
			sym2 = num[rom_num[i+1]]
			if sym1 >= sym2:
				result += sym1
				i += 1
			else:
				result += (sym2 - sym1)
				i += 2
		else:
			result += sym1
			i += 1
	return result

def is_element(name):
	# ABOUT:	A function that checks if a given input string
	#			corresponds to any known element.
	#			Returns True if element exists, and False if it does not.
	# INPUT:	A string containing the symbol of the proposed element.
	# OUTPUT:	A boolean value.
	# SYNTAX:	<boolean> = is_element(<string>)
	from mendeleev import element
	from sqlalchemy.orm.exc import NoResultFound
	try:
		elem = element(name.capitalize())
	except NoResultFound as e:
		return False
	else:
		return True

def is_roman(rom_num):
	# ABOUT:	A function that checks if a given input string
	#			corresponds to any valid roman numerals.
	#			Returns True if roman numerals exist, and False if it does not.
	# INPUT:	A string containing the proposed roman numerals.
	# OUTPUT:	A boolean value.
	# SYNTAX:	<boolean> = is_roman(<string>)
	roman_letters = ['I','V','X','L','C','D','M']
	flag = True
	for char in rom_num:
		if not char.upper() in roman_letters:
			flag = False
			break
	return flag

def get_ion(input_str):
	# ABOUT:	A function that returns the ion stage in standard format,
	#			i.e. as {Symbol} {Stage (roman numerals)}, for a given string.
	#			For example, C III, N II, O VI, Fe XVII
	#			The function returns a list of strings, with first element
	#			containing the symbol and the second element containing ion stage.
	# INPUT:	A string containing proposed ion stage.
	# OUTPUT:	A list of two formatted strings -- element symbol and ion stage.
	# SYNTAX:	<list 'string'> = get_ion(<string>)
	from mendeleev import element
	from sqlalchemy.orm.exc import NoResultFound
	try:
		raw_parts = input_str.split(' ')
		parts = []
		for part in raw_parts:
			if not part == "":
				parts.append(part)
		if not len(parts) == 2:
			raise InvalidInputError("Input must be in the format <Z> <Ion Stage>.")
	except InvalidInputError as e:
		print("Invalid input:", e.data)
		return "NULL"
	else:
		try:
			parts[0] = parts[0].capitalize()
			parts[1] = parts[1].upper()
			if not is_element(parts[0]):
				raise InvalidInputError("No such element as "+parts[0]+" exists.")
			if not is_roman(parts[1]):
				raise InvalidInputError("Ion stage must be in roman numerals.")
		except InvalidInputError as e:
			print("Invalid input:", e.data)
			return ["NULL", "NULL"]
		else:
			return parts

def check_wave_presence(wl, low_wl, high_wl):
	if wl >= low_wl and wl <= high_wl:
		return True
	else:
		return False

def read_ions(ion_filename):
	ion_file = open(ion_filename, 'r')
	ions = ion_file.readlines()
	ion_file.close()
	for i in range(len(ions)):
		ions[i] = ions[i].strip()
	return ions

def get_level(conf, term, J):
	level = conf.ljust(10, ' ')+'| '+term.ljust(5,' ')+'| '+J
	return level

def get_nums(ion):
	from mendeleev import element
	ion_parts = get_ion(ion)
	return [element(ion_parts[0]).atomic_number, write_decimal(ion_parts[1])]

def retrieve_ground_index(ion, df):
	nums = get_nums(ion)
	rec = df[(df['at_num']==nums[0]) & (df['sp_num']==nums[1])]
	gnd_idx = rec.loc[:,'ground_conf'].index[0]
	return gnd_idx

def get_ground_level(ion, df):
	idx = retrieve_ground_index(ion, df)
	ground_level = get_level(df['ground_conf'][idx], df['ground_term'][idx], df['ground_J'][idx])
	return ground_level

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	from matplotlib.colors import LinearSegmentedColormap
	from numpy import linspace
	new_cmap = LinearSegmentedColormap.from_list(
		'trunc({n}, {a:.2f}, {b:2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(linspace(minval, maxval, n)))
	return new_cmap

def get_atom_from_ion(ion):
	atom = ion[0:2]
	return atom.replace(' ','')

def get_colour(ion):
	from matplotlib.pyplot import get_cmap
	from numpy import linspace
	cmap_dict = {'H':'bone_r', \
				'He':'pink_r', \
				'C':'Purples', \
				'N':'Blues', \
				'O':'Greens', \
				'Ne':'Oranges', \
				'Mg':'Reds', \
				'Fe':'Greys'}
	[at_num, sp_num] = get_nums(ion)
	low_cm_val = 0.5
	high_cm_val = 1.0
	cm_vals = linspace(low_cm_val, high_cm_val, at_num)
	cm = get_cmap(cmap_dict[get_atom_from_ion(ion)])
	cm = truncate_colormap(cm, 0.5, 0.9)

	return cm(cm_vals[sp_num - 1])

def show_lines(all_lines, ax, a=1.0):
	y_min, y_max = ax.get_ylim()
	for record in all_lines:
		if record['Included']:
			ax.plot([record['Wavelength'], record['Wavelength']], [y_min, y_max], color=record['Color'], lw=1, alpha=a)

def show_labeled_lines(all_lines, ax, show_wavelength, a=1.0, lwidth=1.0):
	y_min, y_max = ax.get_ylim()
	for record in all_lines:
		if record['Included']:
			ax.plot([record['Wavelength'], record['Wavelength']], [y_min, y_max], color=record['Color'], lw=lwidth, alpha=a)
			if show_wavelength == False:
				ax.text(record['Wavelength'], 0.95*y_max, record['Ion'], color=record['Color'], rotation=90)
			else:
				ax.text(record['Wavelength'], 0.85*y_max, record['Ion']+': '+'%6.3f'%record['Wavelength']+' $\AA$', color=record['Color'], rotation=90)
	ax.set_xlabel('Wavelength ($\AA$)')
	#ax.axes.yaxis.set_visible(False)
			# ax.text(record['Wavelength'], 0.95*y_max, record['Ion']+': '+str(record['Wavelength'])+r'$\AA$', color=record['Color'], rotation=90)

def show_filtered_lines(low_wl, high_wl, ion_filename, gnd_level_datafile):
	import matplotlib.pyplot as plt
	import pandas as pd
	from astroquery.nist import Nist
	import astropy.units as u

	nist_gnd = pd.read_csv(gnd_level_datafile)

	ground_levels = []
	for i in range(nist_gnd.shape[0]):
		ground_levels.append({'Ion':nist_gnd.loc[i,'sp_name'], 'Ground level':get_level(nist_gnd.loc[i,'ground_conf'], nist_gnd.loc[i,'ground_term'], nist_gnd.loc[i,'ground_J'])})

	ground_transitions_only = True

	ions = read_ions(ion_filename)
	ion_table = []

	for ion in ions:
		try:
			table = Nist.query(low_wl*u.AA, high_wl*u.AA, linename=ion)
		except Exception as e:
			print(ion+':')
			print(e)
		else:
			ion_table.append({'Ion':ion, 'Data':table})

	all_lines = []

	for record in ion_table:
		ion = record['Ion']
		ground_level = get_ground_level(ion, nist_gnd)
		ion_table = record['Data']
		NREC = len(ion_table)
		col = get_colour(ion)
		for i in range(NREC):
			if ground_transitions_only:
				level = ion_table[i]['Lower level']
				level_info = level.replace(' ', '').split('|')
				level = get_level(level_info[0], level_info[1], level_info[2])
				if level == ground_level:
					if isinstance(ion_table[i]['Ritz'], float):
						line = {'Wavelength':ion_table[i]['Ritz'], 'Ion':ion, 'Included':True, 'Color':col}
					else:
						line = {'Wavelength':float(ion_table[i]['Ritz'].replace('+','')), 'Ion':ion, 'Included':True, 'Color':col}
				else:
					if isinstance(ion_table[i]['Ritz'], float):
						line = {'Wavelength':ion_table[i]['Ritz'], 'Ion':ion, 'Included':False, 'Color':col}
					else:
						line = {'Wavelength':float(ion_table[i]['Ritz'].replace('+','')), 'Ion':ion, 'Included':False, 'Color':col}
			else:
				if isinstance(ion_table[i]['Ritz'], float):
					line = {'Wavelength':ion_table[i]['Ritz'], 'Ion':ion, 'Included':True, 'Color':col}
				else:
					line = {'Wavelength':float(ion_table[i]['Ritz'].replace('+','')), 'Ion':ion, 'Included':True, 'Color':col}
			all_lines.append(line)

	fig = plt.figure(figsize=(20, 3))
	ax = fig.add_subplot(1,1,1)
	ax.set_ylim(0, 1)

	show_lines(all_lines, ax)
	plt.show()

def get_flux(flux_filename):
	from astropy.io import fits
	from numpy import isnan
	hdul = fits.open(flux_filename)
	data = hdul[1].data
	columns = hdul[1].columns
	
	angstrom = []
	flux = []
	error = []
	for record in data:
		a = record[0]
		f = record[1]
		e = record[2]
		angstrom.append(float(a))
		if isnan(f):
			flux.append(0.0)
		else:
			flux.append(float(f))
		if isnan(e):
			error.append(0.0)
		else:
			error.append(float(e))
	hdul.close()
	return [angstrom, flux, error]

def preview_flux(angstrom, flux):
	from matplotlib.pyplot import step, show, xlabel, ylabel, title, text
	# step(angstrom, flux, color = 'r', lw = 1.5)
	step(angstrom, flux, color = 'k', lw = 1.2)
	xlabel(r'Wavelength ($\AA$)')
	ylabel('Flux')
	title('Preview of spectrum')
	text((max(angstrom)-min(angstrom))*0.7, (max(flux)-min(flux))*0.5, r'Note down limits on $\lambda$', fontsize = 14, bbox=dict(facecolor='red', alpha=0.2))
	print('Displaying a preview of the spectrum.\nNote down the lower and upper limits on wavelength (in angstroms)\n\tbefore closing the preview plot.')
	show()
	
def get_wave_limits():
	low_wl = float(input('Enter lower wavelength (in angstroms): '))
	high_wl = float(input('Enter upper wavelength (in angstroms): '))
	return [low_wl, high_wl]

def get_lines(low_wl, high_wl, v_radial=0, ground_transitions_only=True):
	import pandas as pd
	import astropy.units as u
	from astroquery.nist import Nist
	from os import environ
	from constant.electrodynamics import c
	
	v_c = c/1E3
	
	ion_filename = environ['ASTRODAT']+'/ion.list'
	gnd_level_datafile = environ['ASTRODAT']+'/nist_5-40_gnd.csv'
	
	nist_gnd = pd.read_csv(gnd_level_datafile)
	ground_levels = []
	for i in range(nist_gnd.shape[0]):
		ground_levels.append({'Ion':nist_gnd.loc[i,'sp_name'], 'Ground level':get_level(nist_gnd.loc[i,'ground_conf'], nist_gnd.loc[i,'ground_term'], nist_gnd.loc[i,'ground_J'])})
		
	ions = read_ions(ion_filename)
	ion_table = []
	for ion in ions:
		try:
			table = Nist.query(low_wl*u.AA, high_wl*u.AA, linename=ion)
		except Exception as e:
			print(ion+': '+str(e))
		else:
			ion_table.append({'Ion':ion, 'Data':table})

	all_lines = []
	for record in ion_table:
		ion = record['Ion']
		ground_level = get_ground_level(ion, nist_gnd)
		ion_table = record['Data']
		NREC = len(ion_table)
		col = get_colour(ion)
		for i in range(NREC):
			if ground_transitions_only:
				level = ion_table[i]['Lower level']
				level_info = level.replace(' ', '').split('|')
				level = get_level(level_info[0], level_info[1], level_info[2])
				if level == ground_level:
					if isinstance(ion_table[i]['Ritz'], float):
						line = {'Wavelength':ion_table[i]['Ritz']*(1-(v_radial/v_c)), 'Ion':ion, 'Included':True, 'Color':col}
					else:
						line = {'Wavelength':float(ion_table[i]['Ritz'].replace('+',''))*(1-(v_radial/v_c)), 'Ion':ion, 'Included':True, 'Color':col}
				else:
					if isinstance(ion_table[i]['Ritz'], float):
						line = {'Wavelength':ion_table[i]['Ritz']*(1-(v_radial/v_c)), 'Ion':ion, 'Included':False, 'Color':col}
					else:
						line = {'Wavelength':float(ion_table[i]['Ritz'].replace('+',''))*(1-(v_radial/v_c)), 'Ion':ion, 'Included':False, 'Color':col}
			else:
				if isinstance(ion_table[i]['Ritz'], float):
					line = {'Wavelength':ion_table[i]['Ritz']*(1-(v_radial/v_c)), 'Ion':ion, 'Included':True, 'Color':col}
				else:
					line = {'Wavelength':float(ion_table[i]['Ritz'].replace('+',''))*(1-(v_radial/v_c)), 'Ion':ion, 'Included':True, 'Color':col}
			all_lines.append(line)
	return all_lines

def find_flux_limits(angstrom, flux, low_wl, high_wl):
	subset = []
	for i in range(len(angstrom)):
		if check_wave_presence(angstrom[i], low_wl, high_wl):
			subset.append(flux[i])
	return [min(subset), max(subset)]

def data_ROI(angstrom, flux, error, low_wl, high_wl):
	a = []
	f = []
	e = []
	for i in range(len(angstrom)):
		if angstrom[i] >= low_wl and angstrom[i] <= high_wl:
			a.append(angstrom[i])
			f.append(flux[i])
			e.append(error[i])
	return [a, f, e]

def normalize(data):
	min_val = min(data)
	max_val = max(data)
	for i in range(len(data)):
		data[i] = (data[i]-min_val)/(max_val-min_val)
	return data

def ref_normalize(data, reference):
	min_val = min(reference)
	max_val = max(reference)
	for i in range(len(data)):
		data[i] = (data[i])/(max_val-min_val)
		# data[i] = abs(data[i]/reference[i])
	return data

def wavelength_partition(low_wl, high_wl, npart=4):
	parts = []
	delta_wl = (high_wl - low_wl)/npart
	wl = low_wl
	for i in range(npart):
		part_low = wl
		part_high = wl + delta_wl
		parts.append([part_low, part_high])
		wl = part_high
	return parts

def wavelength_partition_2(low_wl, high_wl, markers):
	try:
		markers.sort()
		if low_wl > markers[0] or high_wl < markers[len(markers)-1]:
			raise Exception('The wavelength markers should lie within the wavelength range chosen')
		npart = len(markers)-1
		parts = []
		wl = markers[0]
		for i in range(npart):
			part_low = wl
			part_high = markers[i+1]
			parts.append([part_low, part_high])
			wl = markers[i+1]
		return parts
	except Exception as e:
		print('Error: '+str(e))

def error_bars(angstrom, flux, error, ax):
	for i in range(len(angstrom)):
		ax.plot([angstrom[i], angstrom[i]], [flux[i]-error[i]*0.5, flux[i]+error[i]*0.5], lw=1.2, color='k', alpha=0.8)

def show_line_overlaid_spec(angstrom, flux, all_lines, low_wl, high_wl, min_flux, max_flux, ax, spec_width=1.0, spec_col='#2ed006', lwidth=1.0):
	ax.set_xlim(low_wl, high_wl)
	ax.set_ylim(min_flux, max_flux)
	ax.step(angstrom, flux, color=spec_col, lw=spec_width)
	# show_lines(all_lines, ax, 0.6)
	show_labeled_lines(all_lines, ax, False, 0.8, lwidth)
	ax.set_xlabel('Wavelength ($\AA$)')
	ax.set_ylabel('Normalized flux (s$^{-1}$ cm$^{-2}\AA^{-1}$)')

def show_line_detail(angstrom, flux, error, all_lines, low_wl, high_wl, min_flux, max_flux, ax, spec_width=1.0, spec_col='#2ed006', lwidth=1.0):
	ax.set_xlim(low_wl, high_wl)
	ax.set_ylim(min_flux-min(error), max_flux+max(error))
	#ax.plot(angstrom, flux, 'o-', color=spec_col, lw=spec_width)
	ax.step(angstrom, flux, color=spec_col, lw=spec_width)
	#error_bars(angstrom, flux, error, ax)
	# show_lines(all_lines, ax, 0.6)
	show_labeled_lines(all_lines, ax, 0.8, lwidth)
	ax.set_xlabel('Wavelength ($\AA$)')
	ax.set_ylabel('Normalized flux (s$^{-1}$ cm$^{-2}\AA^{-1}$)')
	

def plot_spectrum(angstrom, flux, error, all_lines, low_wl, high_wl, show_wavelength=False):
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(20, 12))
	ax_spec = fig.add_subplot(3,1,1)
	ax_lines = fig.add_subplot(3,1,2)
	ax_overlay = fig.add_subplot(3,1,3)
	
	ax_spec.step(angstrom, flux, color='#2ed006', lw=1.0)
	[min_flux, max_flux] = find_flux_limits(angstrom, flux, low_wl, high_wl)
	ax_spec.set_xlim(low_wl, high_wl)
	ax_spec.set_ylim(min_flux, max_flux)
	ax_spec.set_xlabel('Wavelength ($\AA$)')
	ax_spec.set_ylabel('Normalized flux (s$^{-1}$ cm$^{-2}\AA^{-1}$)')
	
	ax_lines.set_ylim(0, 1)
	ax_lines.set_xlim(low_wl, high_wl)
	ax_lines.axes.yaxis.set_visible(False)
	# show_lines(all_lines, ax_lines)
	show_labeled_lines(all_lines, ax_lines, show_wavelength)
	
	show_line_overlaid_spec(angstrom, flux, all_lines, low_wl, high_wl, min_flux, max_flux, ax_overlay)
	fig.tight_layout(pad = 2.5)
	plt.show()

def plot_partitions(angstrom, flux, all_lines, low_wl, high_wl, markers):
	import matplotlib.pyplot as plt
	parts = wavelength_partition_2(low_wl, high_wl, markers)
	nparts = len(parts)
	fig = plt.figure(figsize=(20,14))
	axes = []
	for i in range(nparts):
		axes.append(fig.add_subplot(nparts,1,i+1))
	
	for k in range(len(axes)):
		ang = []
		flx = []
		for i in range(len(angstrom)):
			if check_wave_presence(angstrom[i], parts[k][0], parts[k][1]):
				ang.append(angstrom[i])
				flx.append(flux[i])
		[min_flux, max_flux] = find_flux_limits(ang, flx, parts[k][0], parts[k][1])
		show_line_overlaid_spec(ang, flx, all_lines, parts[k][0], parts[k][1], min_flux, max_flux, axes[k], 1.5, 'k', 1.5)
	
	plt.show()

def bandpass_spec(angstrom, flux, error, wl_i, wl_f):
	ang = []
	flx = []
	err = []
	for i in range(len(angstrom)):
		if check_wave_presence(angstrom[i], wl_i, wl_f):
			ang.append(angstrom[i])
			flx.append(flux[i])
			err.append(error[i])
	return [ang, flx, err]

def bandpass_lines(all_lines, wl_i, wl_f):
	band_lines = []
	for line in all_lines:
		if check_wave_presence(line['Wavelength'], wl_i, wl_f):
			band_lines.append(line)
	return band_lines

def obtain_ROI(low_wl, high_wl):
	ROI_acceptable = False
	while not ROI_acceptable:
		wl_i = float(input('Enter lower wavelength (in angstrom) for region of interest: '))
		wl_f = float(input('Enter upper wavelength (in angstrom) for region of interest: '))
		try:
			if wl_i >= wl_f:
				raise ValueError('You have entered lower wavelength to be greater than upper wavelength.')
			if not check_wave_presence(wl_i, low_wl, high_wl) and not check_wave_presence(wl_f, low_wl, high_wl):
				raise ValueError('You have entered wavelengths that are outside acceptable range of ['+str('%4.1f'%low_wl)+', '+str('%4.1f'%high_wl)+'] angstrom.')
			if not check_wave_presence(wl_i, low_wl, high_wl):
				raise ValueError('You have entered lower wavelength that is outside acceptable range of ['+str('%4.1f'%low_wl)+', '+str('%4.1f'%high_wl)+'] angstrom.')
			if not check_wave_presence(wl_f, low_wl, high_wl):
				raise ValueError('You have entered upper wavelength that is outside acceptable range of ['+str('%4.1f'%low_wl)+', '+str('%4.1f'%high_wl)+'] angstrom.')
		except ValueError as e:
			print('Error: '+str(e))
			print('\nEnter again...')
			continue
		else:
			ROI_acceptable = True
	return [wl_i, wl_f]

def examine_ROI(angstrom, flux, error, all_lines, low_wl, high_wl, wl_i, wl_f):
	import matplotlib.pyplot as plt
	spec = bandpass_spec(angstrom, flux, error, wl_i, wl_f)
	lines = bandpass_lines(all_lines, wl_i, wl_f)
	
	fig = plt.figure(figsize=(20,4))
	ax = fig.add_subplot(1,1,1)
	
	# [min_flux, max_flux] = find_flux_limits(spec[0], spec[1], wl_i, wl_f)
	show_line_detail(spec[0], spec[1], spec[2], lines, wl_i, wl_f, min(spec[1]), max(spec[1]), ax, 1.5)
	fig.tight_layout(pad = 1.0)
	plt.show()

def waveshift_to_velocity(lmda_0, lmda):
	c = 299792458/1E3					# speed of light (in km/s)
	v = (c/lmda_0)*(lmda-lmda_0)
	return v

def velocity_to_waveshift(lmda_0, v):
	c = 299792458/1E3					# speed of light (in km/s)
	lmda = (1+(v/c))*lmda_0
	return lmda

def get_lyman_data():
	return [  {'ion':'C V', 'transition':'alpha', 'angstrom':40.2678}, \
				{'ion':'C V', 'transition':'beta', 'angstrom':34.9728}, \
				{'ion':'C V', 'transition':'gamma', 'angstrom':33.4262}, \
				{'ion':'C VI', 'transition':'alpha', 'angstrom':33.7396}, \
				{'ion':'C VI', 'transition':'beta', 'angstrom':28.4663}, \
				{'ion':'C VI', 'transition':'gamma', 'angstrom':26.9901}, \
				{'ion':'N VI', 'transition':'alpha', 'angstrom':28.787}, \
				{'ion':'N VI', 'transition':'beta', 'angstrom':24.898}, \
				{'ion':'N VI', 'transition':'gamma', 'angstrom':23.771}, \
				{'ion':'N VII', 'transition':'alpha', 'angstrom':24.7846}, \
				{'ion':'N VII', 'transition':'beta', 'angstrom':20.9106}, \
				{'ion':'N VII', 'transition':'gamma', 'angstrom':19.8261}, \
				{'ion':'O VII', 'transition':'alpha', 'angstrom':21.602}, \
				{'ion':'O VII', 'transition':'beta', 'angstrom':18.627}, \
				{'ion':'O VII', 'transition':'gamma', 'angstrom':17.768}, \
				{'ion':'O VIII', 'transition':'alpha', 'angstrom':18.9725}, \
				{'ion':'O VIII', 'transition':'beta', 'angstrom':16.0067}, \
				{'ion':'O VIII', 'transition':'gamma', 'angstrom':15.1765}, \
				{'ion':'Fe XVII', 'transition':'alpha', 'angstrom':17.0510}, \
				{'ion':'Fe XVII', 'transition':'beta', 'angstrom':16.7760}, \
				{'ion':'Fe XVII', 'transition':'gamma', 'angstrom':15.2620}, \
				{'ion':'Fe XVIII', 'transition':'alpha', 'angstrom':16.0050}, \
				{'ion':'Fe XVIII', 'transition':'beta', 'angstrom':14.3730}, \
				{'ion':'Fe XVIII', 'transition':'gamma', 'angstrom':14.2080}]

def get_lyman_line(ion, transition='alpha'):
	lyman_data = get_lyman_data()
	for record in lyman_data:
		if ion==record['ion'] and transition==record['transition']:
			return record['angstrom']

def get_lyman_wave_limits(lmda_0, v_lim=5000.0):
	lmda_min = velocity_to_waveshift(lmda_0, -v_lim)
	lmda_max = velocity_to_waveshift(lmda_0, v_lim)
	return [lmda_min, lmda_max]

def map_spec_to_vel(lmda_0, angstrom, flux, error, v_lim=5000.0):
	wave_limit = get_lyman_wave_limits(lmda_0, v_lim)
	a = []
	f = []
	e = []
	v = []
	for i in range(len(angstrom)):
		if angstrom[i] >= wave_limit[0] and angstrom[i] <= wave_limit[1]:
			v.append(waveshift_to_velocity(lmda_0, angstrom[i]))
			a.append(angstrom[i])
			f.append(flux[i])
			e.append(error[i])
	return [v, a, f, e]

def plot_doppler_lines(element, angstrom, flux, error, v_lim=5000.0, mode='abs'):
	import matplotlib.pyplot as plt
	# flux = normalize(flux)
	ion_species = [{'element':'C', 'sp_1':'V', 'sp_2':'VI'}, \
					{'element':'N', 'sp_1':'VI', 'sp_2':'VII'}, \
					{'element':'O', 'sp_1':'VII', 'sp_2':'VIII'}, \
					{'element':'Fe', 'sp_1':'XVII', 'sp_2':'XVIII'}]
	transitions = ['alpha', 'beta', 'gamma']
	trans_symbol = {'alpha':r'$\alpha$', 'beta':r'$\beta$', 'gamma':r'$\gamma$'}
	trans_color = {'alpha':'r', 'beta':'g', 'gamma':'b'}
	plot_series = []
	for member in ion_species:
		if member['element'] == element:
			plot_series.append(member['element']+' '+member['sp_1'])
			plot_series.append(member['element']+' '+member['sp_2'])
	
	num_rows = len(transitions)
	num_cols = len(plot_series)
	num_plots = num_rows*num_cols
	axis = []
	
	fig = plt.figure(figsize=(12,12))
	k = 1
	for i in range(num_rows):
		for j in range(num_cols):
			axis.append({'ion':plot_series[j], 'transition':transitions[i], 'axis':fig.add_subplot(num_rows,num_cols,k)})
			k += 1
	
	print('\nDisplaying '+element+' lines...')
	for ax in axis:
		lmda_0 = get_lyman_line(ax['ion'], ax['transition'])
		[v, a, f, e] = map_spec_to_vel(lmda_0, angstrom, flux, error, v_lim)
		ax['axis'].axvline(linewidth=1.5, color='gray', dashes=(5, 2))
		if f:
			if mode == 'abs':
				ax['axis'].axvline(v[f.index(min(f))], 0, linewidth=1.5, color='magenta', dashes=(5, 2))
				ax['axis'].text(v[f.index(min(f))]+50, 1.0*max(f), '%6.1f'%v[f.index(min(f))]+' km/s', color='magenta', rotation=90)
				ax['axis'].text(v[f.index(min(f))]-300, 1.0*max(f), '%6.4f'%(a[f.index(min(f))]-lmda_0)+' $\AA$', color='magenta', rotation=90)
			elif mode == 'ems':
				ax['axis'].axvline(v[f.index(max(f))], 0, linewidth=1.5, color='forestgreen', dashes=(5, 2))
				ax['axis'].text(v[f.index(max(f))]+50, 0.6*max(f), '%6.1f'%v[f.index(min(f))]+' km/s', color='forestgreen', rotation=90)
				ax['axis'].text(v[f.index(max(f))]-300, 0.5*max(f), '%6.4f'%(a[f.index(min(f))]-lmda_0)+' $\AA$', color='forestgreen', rotation=90)
			elif mode == 'Pcyg':
				ax['axis'].axvline(v[f.index(min(f))], 0, linewidth=1.5, color='magenta', dashes=(5, 2))
				ax['axis'].text(v[f.index(min(f))]+50, 1.0*max(f), '%6.1f'%v[f.index(min(f))]+' km/s', color='magenta', rotation=90)
				ax['axis'].text(v[f.index(min(f))]-300, 1.0*max(f), '%6.4f'%(a[f.index(min(f))]-lmda_0)+' $\AA$', color='magenta', rotation=90)
				ax['axis'].axvline(v[f.index(max(f))], 0, linewidth=1.5, color='forestgreen', dashes=(5, 2))
				ax['axis'].text(v[f.index(max(f))]+50, 0.6*max(f), '%6.1f'%v[f.index(min(f))]+' km/s', color='forestgreen', rotation=90)
				ax['axis'].text(v[f.index(max(f))]-300, 0.5	*max(f), '%6.4f'%(a[f.index(min(f))]-lmda_0)+' $\AA$', color='forestgreen', rotation=90)
						
		ax['axis'].plot(v, f, 'o-', color=trans_color[ax['transition']], lw=1.5)
		for i in range(len(v)):
			ax['axis'].plot([v[i], v[i]], [f[i]-e[i]*0.5, f[i]+e[i]*0.5], lw=1.2, color='k', alpha=0.8)
		if get_atom_from_ion(ax['ion'])=='Fe':
			ax['axis'].set_title(ax['ion']+' at '+'%7.4f'%lmda_0+r' $\AA$')
		else:
			ax['axis'].set_title(ax['ion']+' Ly'+trans_symbol[ax['transition']]+' at '+'%7.4f'%lmda_0+r' $\AA$')
		ax['axis'].set_xlim(-v_lim, v_lim)
		ax['axis'].set_xlabel('Radial velocity (km/s)')
		ax['axis'].set_ylabel('Normalized flux (s$^{-1}$ cm$^{-2}\AA^{-1}$)')
	
	fig.tight_layout(pad = 2.5)
	plt.show()


