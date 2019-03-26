import sys
import pickle
import msbm
import varinf
import init_msbm_vi as im

# ################ INITIALIZATION FUNCTIONS ###############


def experiment_consensus(data_file_url, out_file_url):

    data, par = load_data(data_file_url)

    # override hyper-parameters from data
    # par['Q'] = 3 
    # par['M'] = 2 

    mom, prior = im.init_moments(par)

    #Debugging purpose
    mom0, prior0 = im.init_moments_truth(par, data)

    elbos = msbm.compute_elbos(mom0, data, prior, par)
    par['elbos0'] = elbos0

    par['MAX_ITER'] = 100

    # START VARIATIONAL INFERENCE
    results_mom = varinf.infer(mom, data, prior, par, 'cavi')

    print('\nSaving results to {:s} ... '.format(out_file_url), end='')
    pickle.dump({'mom': results_mom}, open(out_file_url, 'wb'))
    print('Saved.')


def main():

    if len(sys.argv) < 3:

        path_data = 'data'
        fname = 'msbm2'
        data_file_url = path_data + '/' + fname + '.pickle'
        out_file_url = path_data + '/' + 'results_' + fname + '.pickle'

    else:
        data_file_url = sys.argv[1]
        out_file_url = sys.argv[2]

    test_varinf(data_file_url, out_file_url)


if __name__ == '__main__':
    main()
