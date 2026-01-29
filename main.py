import os
import numpy as np
import pickle
import gzip
import argparse
from DSTS.dsts import dsts
import time


# Load the train and validation datasets
def load_data(dataset):
    train_path=f'TSdata/{dataset}/{dataset}.pkl'
    try:
        try:
            # Load the data from gzip-compressed pickle files
            with gzip.open(train_path, 'rb') as f:
                train_data = pickle.load(f)
        except:
            with open(train_path, 'rb') as f:
                train_data = pickle.load(f)

        print("Training data shape:", train_data.shape, flush=True)

        # If data is 3D, return as numpy arrays for further processing
        if len(train_data.shape) == 3:
            pass
        else:
            # If 2D, convert to 3D
            train_data = train_data[:,:,np.newaxis]

        return train_data

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)


def main(args):
    start_time = time.time()
    train_data = load_data(args.dataset)
 
    # Generate data independently
    if args.indep:
        generated_data_list = []
        for i in range(train_data.shape[2]):
            print("Generating channel: ", i+1)
            data = train_data[:,:,i][:,:,np.newaxis]
            mixup_model = dsts(args.sort, args.centering, args.pca_mixup)
            mixup_model.fit(data)

            generated_data = mixup_model.generate(aug=args.aug)
            generated_data_list.append(generated_data)

        generated_data = np.concatenate(generated_data_list, axis=2)
    
    # Do not generate data independently
    else:
        mixup_model = dsts(args.sort, args.centering, args.pca_mixup)
        mixup_model.fit(train_data)

        generated_data = mixup_model.generate(aug=args.aug)

        
    # Save data
    if args.pca_mixup:
        output_directory = f'TSexperiments/dstspca/{args.dataset}/pca_mixup/{args.centering}'
    else:
        output_directory = f'TSexperiments/dstspca/{args.dataset}/pca_nn/{args.centering}'
        
    os.makedirs(output_directory, mode=0o775, exist_ok=True)
    print("output directory:", output_directory)

    with open(os.path.join(output_directory, f'synth_{args.dataset}_{args.rep}.pkl'), 'wb') as tf:
        pickle.dump(generated_data, tf)

    print("Synthetic data generated.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='energy')
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--pca_mixup', action='store_true')
    parser.add_argument('--centering', type=str, required=True)
    parser.add_argument('--indep', action='store_true')
    parser.add_argument('--aug', default=1)
    parser.add_argument('--rep', default=1)
    args = parser.parse_args()

    main(args)
