import argparse



def plot_activations(args):

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to model checkpoint .pth file")

    args = parser.parse_args()
    plot_activations(args)

if __name__ == "__main__":
    main()