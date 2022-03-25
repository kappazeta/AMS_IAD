import os
import re
import csv
import shutil
import argparse


def load_list(path):
    valid_tiles = []

    with open(path, 'r', newline='') as fi:
        reader = csv.reader(fi, delimiter=',', quotechar='"')
        header_row = True
        for row in reader:
            if header_row:
                header_row = False
            else:
                valid_tiles.append((row[0], int(row[1]), int(row[2])))

    return valid_tiles


def download(path_in, path_out):
    path_head, filename = os.path.split(path_in)
    print(filename)

    m = re.match(r".*/(\d+_\d+_\d+T\d+_T[A-Z\d]+.KM)/(tile_\d+_\d+)$", path_head)
    if m:
        dirname = m.group(1)
        subtile = m.group(2)
        full_path_out = os.path.join(path_out, dirname, subtile)
        os.makedirs(full_path_out, exist_ok=True)

        cmd = "s3cmd -c ~/.s3cfg_wasabi get --recursive \"s3://pria-ams/{}/{}/{}\" \"{}/\"".format(dirname, subtile, filename, full_path_out)
        os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description='AMS_IAD redownloader from S3')
    parser.add_argument('list_path', type=str, help='Path to the list with S2 products')
    parser.add_argument('km_path', type=str, help='Path to the output directory')
    args = parser.parse_args()

    i = 0
    tiles = load_list(args.list_path)
    for tile in tiles:
        print("{} / {}".format(i, len(tiles)))
        download(tile[0], args.km_path)
        i += 1

if __name__ == "__main__":
    main()

