import hls4ml
import torch
import numpy as np
import networks.KLiNQ_StudentModel as klinq


input = np.load('inputs/KLiNQ_student_q0_input.npy')
input_shape = input.shape[1:]
model = klinq.KLiNQStudentModel(input_shape[0])
model.load_state_dict(torch.load(
    'checkpoints/KLiNQ_student_paper_q0_1000ns.pth', map_location=torch.device('cpu')))

config = hls4ml.utils.config_from_pytorch_model(
    model,
    input_shape,
    granularity='name',
    default_precision='ap_fixed<16,6>'
)

clock_period = 2  # 2 ns = 500 MHz
config['Model']['ClockPeriod'] = clock_period
hls_project_dir = 'hls-project-klinq'

hls_model = hls4ml.converters.convert_from_pytorch_model(
    model=model,
    hls_config=config,
    backend='Vitis',
    part='xczu49dr-ffvf1760-2-e',
    clock_period=clock_period,
    output_dir=hls_project_dir,
    project_name='klinq_hls4ml'
)

hls_model.compile()
hls_model.build()

hls4ml.report.read_vivado_report(hls_project_dir)
