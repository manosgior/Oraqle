import hls4ml
import torch
import numpy as np
import networks.CNN as cnn


model = cnn.CNN(m_param=8)
model.load_state_dict(torch.load(
    'checkpoints/CNN_small_m8_ds40_1000ns_singlehead.pth', map_location=torch.device('cpu')))
input = np.load('inputs/CNN_small_m8_ds40_input.npy')
input_shape = input.shape[1:]

config = hls4ml.utils.config_from_pytorch_model(
    model,
    input_shape,
    granularity='name',
    default_precision='ap_fixed<16,6>'
)

clock_period = 2  # 2 ns = 500 MHz
config['Model']['ClockPeriod'] = clock_period
print(config['LayerName'].keys())
# set per conv layer to its output length (or just high)
config['LayerName']['conv1']['ParallelizationFactor'] = 7   # conv1 has 7 output steps
config['LayerName']['conv2']['ParallelizationFactor'] = 4
config['LayerName']['res_block_conv1']['ParallelizationFactor'] = 4   # residual convs
config['LayerName']['res_block_conv2']['ParallelizationFactor'] = 4
# config['Model']['ReuseFactor'] = 4

hls_project_dir = 'hls-project-cnn'

hls_model = hls4ml.converters.convert_from_pytorch_model(
    model=model,
    hls_config=config,
    backend='Vitis',
    part='xczu49dr-ffvf1760-2-e',
    clock_period=clock_period,
    output_dir=hls_project_dir,
    project_name='cnn_hls4ml'
)

hls_model.compile()
hls_model.build()

hls4ml.report.read_vivado_report(hls_project_dir)
