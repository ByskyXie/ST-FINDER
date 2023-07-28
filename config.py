from adapter import MatrixDataset, get_traffic_matrix_geant, get_traffic_matrix_abilene


""" Train config (Global) """
seed = 777

input_matrix_num = 8
predict_matrix_num = 1
batch_size = 8
in_channels = 1
epochs = 1000
sampling_rate1 = 0.2  # sampling rate of matrix
sampling_rate2 = 0.5  # sampling rate of time interval

betas = (0.9, 0.999)
multi_head_channels = 8
using_early_stop = True
using_tube_samping = False


""" Model config """
class ModelConfigAbilene:
    name='Abilene'
    blocks_num = 4
    matrix_row = 12
    matrix_column = 12
    embed_dim = 12 * 12
    freq_of_minute = 5
    lr = 0.0004
    max_grad = 100

    all_batch_num = 2000
    scaler = 1e4
    path = 'C:/Python project/DataFillingCPC/Measured'  # dataset path
    fn_get_traffic_matrix = get_traffic_matrix_abilene

    # Encoder
    encoder_block_kernels = [(3, 3, 3), (1, 3, 3)]
    encoder_block_paddings = [(1, 0, 0), (0, 1, 1)]
    downsample_kernel = (3, 3, 3)
    downsample_padding = (1, 0, 0)
    # Decoder
    decoder_block_kernels = [(3, 3, 3), (3, 3, 3), (1, 3, 3)]
    decoder_block_paddings = [(1, 0, 0), (1, 1, 1), (0, 1, 1)]

    def show(self):
        pass


class ModelConfigGEANT:
    name='GEANT'
    blocks_num = 4
    matrix_row = 23
    matrix_column = 23
    embed_dim = 23 * 23
    freq_of_minute = 15
    lr = 0.0004
    max_grad = 100

    all_batch_num = 2000
    scaler = 1
    path = 'C:/Python project/DataFillingCPC/GEANT'  # dataset path
    fn_get_traffic_matrix = get_traffic_matrix_geant

    # Encoder
    encoder_block_kernels = [(3, 5, 5), (1, 3, 3)]
    encoder_block_paddings = [(1, 0, 0), (0, 1, 1)]
    downsample_kernel = (3, 5, 5)
    downsample_padding = (1, 0, 0)
    # Decoder
    decoder_block_kernels = [(3, 5, 5), (3, 3, 3), (1, 3, 3)]
    decoder_block_paddings = [(1, 0, 0), (1, 1, 1), (0, 1, 1)]

    def show(self):
        pass



