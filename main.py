import argparse
from utils.CrossValM2 import CrossValidationM2

parser = argparse.ArgumentParser()

parser.add_argument('--filename', "-f", default="iris", help="Dataset name")
parser.add_argument('--debug', action='store_true', help="Set to true for debugging")
parser.add_argument('--prune', '-p', action='store_true', help="Set to true for pruning")
parser.add_argument('--minsup', type=float, default=0.05, help='Minimum support level')
parser.add_argument('--minconf', type=float, default=0.5, help='Minimum confidence level')
parser.add_argument('--multiple', action='store_true', help="Multiple minsup")


def main(args):
    if args.debug:
        print("Non-flag arguments:", args)
    data_path = f"./datasets/{args.filename}.data"
    scheme_path = f"./datasets/{args.filename}.names"
    print("data_path:", data_path)
    print("scheme_path:", scheme_path)
    minsup = args.minsup
    minconf = args.minconf

    validation = CrossValidationM2(data_path, scheme_path, minsup, minconf)

    print(f"Prune: {args.prune}, Multiple Minsup: {args.multiple}")
    validation.cross_validation(multiple=args.multiple,
                                prune=args.prune) # multiple minsups

if __name__ =="__main__":
    args = parser.parse_args()
    main(args)
