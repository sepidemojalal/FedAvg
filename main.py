

import argparse

from src.client import start_client
# if __name__ == "__main__":
#     start_client()


from src.server import start_federated_learning



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run federated learning')
    parser.add_argument('--client', action='store_true', help='If true, run client')
    parser.add_argument('--federated_learning', action='store_true', help='If true, run federated learning')
    args = parser.parse_args()


    if args.federated_learning:
        start_federated_learning()
    if args.client:
        start_client()
    if not (args.client or args.federated_learning):
        print('Nothing specified to run')