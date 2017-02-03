from __future__ import print_function

import argparse
import os
import shutil
import sys
import threading
from multiprocessing import Pool
import subprocess

def percentage(perc):
    print('Percentage complete: ','[', int(perc*80)*'=', ' '* int(80-perc*80), ']')

def findTemplateName(model_dir):

    with open(os.path.join(model_dir, 'template.hoc'), 'r') as temp_file:

        for line in temp_file:
            #Retrun template name
            if line.__contains__('begintemplate'):
                return line.strip().split()[1]

    return None

"""
    Function called by thread to run a simulation for single model
"""
def simulate(args, model_dir):

    mechanisms_relative_path = os.path.relpath(os.path.join(model_dir, "mechanisms"))
    subprocess.call("nrnivmodl {0} ".format(mechanisms_relative_path))

    if not args.clean:
        from simulation import Simulator
        from modelfit import GIFFit

        model_name = model_dir.split('/')[-1]
        print(40 * '#', 'FITTING ' + model_name, 40 * '#')

        template_name = findTemplateName(model_dir)
        if template_name:
            args.cell_template_name = template_name

        try:
            simulator = Simulator(model_path=model_dir, cell_template_name=args.cell_template_name)
            if args.optimize:
                print('Optimizing...')
                simulator.main(optimize=True)
            if not args.fit:
                print('Simulating...')
                simulator.main(optimize=False)
            print('Fitting...')
            GIFFit(simulator=simulator, plot=args.plot).run()

        except Exception as e:
            print(e)

    else:
        model_name = model_dir.split('/')[-1]
        print(40 * '#', 'CLEANING ' + model_name, 40 * '#')
        if os.path.isdir(os.path.join(model_dir, 'simulation')):
            shutil.rmtree(os.path.join(model_dir, 'simulation'))



def main():

    print("Starting simulations...")

    # Arguement parsing
    parser = argparse.ArgumentParser(description="Parse parameters for running fitting of NEURON models to NEST models. ")

    parser.add_argument("-md", "--model-dirs", type=str, help="Space divided list of paths to model directories",
                        default=[])
    parser.add_argument("-mr", "--models-root", type=str, help="Root directory containing the model directories",
                        default=False)
    parser.add_argument("-p", "--plot", help="Plot during fitting (requires presence during fitting)",
                        default=False, action='store_true')
    parser.add_argument("-j", "--json", help="Json config file with all of the parameters needed for the simulations and fitting",
                        default=None)
    parser.add_argument("-t", "--test",
                        help="Use if you want to test out the program with hardcoded values",
                        default=False, action='store_true')
    parser.add_argument("-f", "--fit",
                        help="Just run fitting",
                        default=False, action='store_true')
    parser.add_argument("-c", "--clean",
                        help="Clean all of the simulation data of the specified models",
                        default=False, action='store_true')
    parser.add_argument("-o", "--optimize",
                        help="Do optimize step before doing the simulation",
                        default=False, action='store_true')
    parser.add_argument("-th", "--threads",
                        help="The number of threads that should be used. One simulation per thread is to be made",
                        default=1)

    parser.add_argument("-ctn", "--cell-template-name",
                        help="Cell template name", type=str,
                        default='cADpyr232_L5_TTPC1_0fb1ca4724')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.test:
        print("Test mode, reading default model from environment variables...")
        args.model_dirs = [os.environ['NEURON_DEFAULT_MODEL']]
    else:
        try:
            if args.models_root:
                args.model_dirs = []
                for ind, f in enumerate(os.listdir(args.models_root)):
                    if f.__contains__('zip') or f.__contains__('.'):
                        continue
                    else:
                        args.model_dirs.append(os.path.join(args.models_root, f))
            else:
                args.model_dirs = args.model_dirs.split()
        except Exception as e:
            print("There are no models specified, please specify model directories")
            return

    pool = Pool(processes=args.threads)
    print("Running {} threads...".format(args.threads))
    print(args.model_dirs)
    for model_dir in args.model_dirs:
        pool.apply_async(func=simulate, kwds={"args" : args, "model_dir" : model_dir})

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()

