from argparse import ArgumentParser

parser = ArgumentParser(description = 'ConvLSTM Live Prediction')
parser.add_argument('--input_size', default=2, type=int, help='input size size for ConvLSTM')
parser.add_argument('--hidden_size', default=6, type=int, help='hidden size for ConvLSTM')
parser.add_argument('--numLayers', default=1, type=int, help='layers for ConvLSTM')
parser.add_argument('--learning_rate', default=1e-3, type=float,  help='learning rate of model')
parser.add_argument('--epochs', default=50, type=int, help='train epochs')

parser.add_argument('--start_frame', default=0, type=int, help = 'started frameId')
parser.add_argument('--gblur_size_width', default=5, type=int, help = 'gblur_size_width')
parser.add_argument('--gblur_size_high', default=5, type=int, help = 'gblur_size_high')
parser.add_argument('--windows', '-w', default=4, type=int, help='prediction window size')
parser.add_argument('--threshold', default=0.2, type=float, help='control predict_tile choose')
parser.add_argument('--showImage', default=False, type=bool, help='show test image or not')
parser.add_argument('--salFlag', default="sample_attention", type=str, help='use our own dataset or not')

# path setting
parser.add_argument('--save_path', default = 'D:/VR_project/ViewPrediction/online/',
                    type=str, metavar='PATH', help='path to save online model')
parser.add_argument('--log_path', default = 'D:/VR_project/ViewPrediction/online/log/9x16/',
                    type=str, metavar='PATH', help='log base path')

def get_args():
    args = parser.parse_args()
    return args