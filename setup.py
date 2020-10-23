from os import getcwd,makedirs,path,chdir,system
import site
import os
from shutil import copy2
from sys import argv

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            makedirs(directory)
    except OSError:
        print('Error :: Creating directory. ' + directory)

site_package_folder = site.getusersitepackages()
#site_package_folder = os.environ['HOME']+'/Desktop/site-packages'
data_folder = os.environ['HOME']+'/astro-data'
#data_folder = os.environ['HOME']+'/Desktop/astro-data'

source_folder = getcwd()

print('\nRemoving tar file...')
system('rm '+source_folder+'/xmmrgs_lines.tar')

req_packages = ['numpy', 'matplotlib', 'pandas', 'mendeleev', 'sqlalchemy', 'astropy', 'astroquery']

print('Upgrading existing version of pip...')
system('python3 -m pip install --upgrade pip')

for package in req_packages:
	install_command = 'python3 -m pip install '+package
	print('\nInstalling package '+package)
	system(install_command)

print('\nCopying user site packages...')
target_folder = site_package_folder+'/constant'
create_folder(target_folder)
copy2(source_folder+'/electrodynamics.py', target_folder+'/electrodynamics.py')
target_folder = site_package_folder+'/astro_da'
create_folder(target_folder)
copy2(source_folder+'/xmmrgs_lines.py', target_folder+'/xmmrgs_lines.py')

print('\nCopying data files...')
target_folder = data_folder
create_folder(target_folder)
copy2(source_folder+'/ion.list', target_folder+'/ion.list')
copy2(source_folder+'/nist_5-40_gnd.csv', target_folder+'/nist_5-40_gnd.csv')

print('\nRemoving redundant files...')
system('rm '+source_folder+'/electrodynamics.py')
system('rm '+source_folder+'/xmmrgs_lines.py')
system('rm '+source_folder+'/ion.list')
system('rm '+source_folder+'/nist_5-40_gnd.csv')

print('\nYou may now test the Python code in the following files:\n\tline-overlaid-flux.py\n\tflux-inspect.py\n\tline-shift.py \nUse the flux files provided in the folder /fluxes. The flux file is to be used as an argument while running the Python code.')

os.remove(argv[0])
