import os
import re
import csv
import shutil
import argparse
import numpy as np
from numpy.random import default_rng
from netCDF4 import Dataset


rng = default_rng()


class AMS_IAD_Picker(object):
    def __init__(self, path):
        self.path_dir = path
        self.path_valid_sorted = "valid_sorted.csv"
        self.path_out = "split_output"
        self.valid_tiles = []

    def check(self, fpath):
        root = Dataset(fpath, "r")

        b_data = root["/B02"][:]
        num_valid = len(b_data[b_data > 0.1])

        b_data = root["/Label"][:]
        num_labels = len(b_data[b_data > 0])

        root.close()

        return num_valid, num_labels

    def check_tiffs(self):
        # Look for TIFF files in subdirectories.
        tiffnames = []
        for f in os.listdir(self.path_dir):
            if f.endswith(".tif") and not f.endswith("_mask.tif"):
                m = re.match(r"(?:\d+)_(\d+)_(\d+T\d+)_([A-Z\d]+)\.tif", f)
                if (m):
                    tiffnames.append("{}_{}".format(m.group(3), m.group(2)))
        return tiffnames

    def save_list(self, path, data):
        with open(path, 'w', newline='') as fo:
            writer = csv.writer(fo, delimiter=',', quotechar='"')
            writer.writerow(["path", "num_valid", "num_labels"])
            for row in data:
                writer.writerow(row)

    def load_list(self, path):
        self.valid_tiles = []

        with open(path, 'r', newline='') as fi:
            reader = csv.reader(fi, delimiter=',', quotechar='"')
            header_row = True
            for row in reader:
                if header_row:
                    header_row = False
                else:
                    self.valid_tiles.append((row[0], int(row[1]), int(row[2])))

        return self.valid_tiles

    def process(self, th_valid=0, th_labels=0, skip_tiles=[]):
        fpaths = []

        print("Processing {}".format(self.path_dir))

        # Look for NetCDF files in subdirectories.
        for root, dirs, files in os.walk(self.path_dir):
            for f in files:
                if f.endswith(".nc"):
                    # Skip subtiles from a list of specific tiles.
                    skip_subtile = False
                    for stile in skip_tiles:
                        if stile in f:
                            skip_subtile = True
                    if skip_subtile:
                        continue
                    print(f)

                    # We should not try to open any NetCDF files which are still being processed.
                    being_processed = False
                    tiffnames = self.check_tiffs()
                    for tiffname in tiffnames:
                        if f.startswith(tiffname):
                            being_processed = True
                            break

                    if being_processed:
                        continue

                    nc_path = os.path.join(root, f)
                    fpaths.append(nc_path)

                    num_valid, num_labels = self.check(nc_path)
                    if num_valid > th_valid and num_labels > th_labels:
                        print("Valid {} {} {}".format(nc_path, num_valid, num_labels))
                        self.valid_tiles.append((nc_path, num_valid, num_labels))

                        if len(self.valid_tiles) == 10:
                            self.save_list("test.csv", self.valid_tiles)

        print("Sorting")
        self.valid_tiles.sort(key=lambda x: x[1], reverse=True)
        for nc_path, num_valid, num_labels in self.valid_tiles:
            print(nc_path, num_valid, num_labels)

        self.save_list(self.path_valid_sorted, self.valid_tiles)

    def load(self):
        self.load_list(self.path_valid_sorted)

    def cp_sets(self, path_dir, train_paths, val_paths, test_paths):
        if os.path.exists(path_dir) and os.path.isdir(path_dir):
            shutil.rmtree(path_dir)

        path_dir_train = os.path.join(path_dir, "train")
        path_dir_val = os.path.join(path_dir, "val")
        path_dir_test = os.path.join(path_dir, "test")
        os.makedirs(path_dir_train)
        os.makedirs(path_dir_val)
        os.makedirs(path_dir_test)

        for path in train_paths:
            shutil.copy2(path, path_dir_train)
        for path in val_paths:
            shutil.copy2(path, path_dir_val)
        for path in test_paths:
            shutil.copy2(path, path_dir_test)

    def split(self, n_total, p_train, p_val, p_test):
        n = min(n_total, len(self.valid_tiles))
        n_val = int(n * p_val)
        n_test = int(n * p_test)

        ids_test = rng.integers(low=0, high=n, size=n_test)
        test_tiles = [self.valid_tiles[i] for i in ids_test]
        the_rest = [self.valid_tiles[i] for i in range(0, n) if i not in ids_test]
        n_rest = len(the_rest)

        ids_val = rng.integers(low=0, high=n_rest, size=n_val)
        val_tiles = [the_rest[i] for i in ids_val]
        train_tiles = [the_rest[i] for i in range(0, n_rest) if i not in ids_val]

        print("{} valid tiles selected, {} in train, {} in val, and {} in test".format(len(self.valid_tiles), len(train_tiles), len(val_tiles), len(test_tiles)))

        # Check that there are no duplicates.
        train_paths = set([x[0] for x in train_tiles])
        val_paths = set([x[0] for x in val_tiles])
        test_paths = set([x[0] for x in test_tiles])

        print("{} intersections between test and train set".format(len(test_paths.intersection(train_paths))))
        print("{} intersections between test and val set".format(len(test_paths.intersection(val_paths))))
        print("{} intersections between train and val set".format(len(train_paths.intersection(val_paths))))

        self.cp_sets(self.path_out, train_paths, val_paths, test_paths)


def main():
    parser = argparse.ArgumentParser(description='AMS_IAD picker for train and test sets')
    parser.add_argument('km_path', type=str, help='Path to the directory with the subtiled product (.KM directory)')
    parser.add_argument('-p', '--process', action='store_true', dest='do_process', help='Process all the subdirectories', default=False)
    parser.add_argument('-n', '--num', type=int, dest='num_tiles', help='Number of subtiles to split', default=10000)
    parser.add_argument('--valid', type=float, dest='th_valid', help='Percentage of pixels which need to be cloudless', default=0.2)
    args = parser.parse_args()

    picker = AMS_IAD_Picker(args.km_path)
    if args.do_process:
        picker.process(th_valid=args.th_valid*256*256, th_labels=0.01*256*256, skip_tiles=["T34V"])
    else:
        picker.load()
    picker.split(int(args.num_tiles), 0.8, 0.1, 0.1)


if __name__ == "__main__":
    main()

