import argparse

from utils.utils import parse_config


def parse_opt():
    parser = argparse.ArgumentParser(description = 'Run P2')
    parser.add_argument('--cfg', type = str, default = '', help = 'config.json path')
    parser.add_argument('--predict', dest='predict', action='store_true')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--mosaick', dest='mosaick', action='store_true')
    args = parser.parse_args()
    return args


def main(opt):
    config = parse_config(opt.cfg)

    if opt.train:
        from utils.train_utils import train
        train(config, resume_training=opt.resume)
    if opt.evaluate:
        from utils.eval_utils import evaluate
        evaluate(config)
    if opt.predict:
        from utils.predict_utils import predict
        predict(config)
    if opt.mosaick:
        from utils.raster_mosaic import mosaick
        mosaick(config)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
