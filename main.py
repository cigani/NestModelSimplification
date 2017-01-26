from __future__ import print_function
import sys
import argparse
import os
import shutil

def percentage(perc):
    print('Percentage complete: ','[', int(perc*80)*'=', ' '* int(80-perc*80), ']')

def main():

    print("Starting simulations...")

    # Arguement parsing
    parser = argparse.ArgumentParser(description="Parse parameters for running fitting of NEURON models to NEST models. ")

    parser.add_argument("-md", "--model-dirs", type=str, help="Space divided list of paths to model directories",
                        default=[])
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


    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.test:
        print("Test mode...")
        args.model_dirs = ['/Users/vlasteli/Documents/Models/L5_TTPC1_cADpyr232_1']
    else:
        try:
            args.model_dirs = args.accumulate(args.model_dirs)
        except Exception as e:
            print("There are no models specified, please specify model directories")
            return

    counter = 0
    if not args.clean:
        from Simulator import Simulator
        from modelfit import GIFFit
        #For each model directory, do fitting
        for model_dir in args.model_dirs:
            model_name = model_dir.split('/')[-1]
            percentage(float(counter) / len(args.model_dirs))
            print(40*'#', 'FITTING ' + model_name, 40*'#')
            try:
                simulator = Simulator(model_path=model_dir)
                if not args.fit:
                    simulator.main(optimize=False)
                GIFFit(simulator=simulator, plot=args.plot).run()

            except Exception as e:
                print(e)

    else:
        #Clean flag specified, remove all simulation data
        for model_dir in args.model_dirs:
            model_name = model_dir.split('/')[-1]
            print(40 * '#', 'CLEANING ' + model_name, 40 * '#')
            if os.path.isdir(os.path.join(model_dir, 'simulation')):
                shutil.rmtree(os.path.join(model_dir, 'simulation'))


if __name__ == '__main__':
    main()

