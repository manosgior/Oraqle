EXAMPLE INPUTS FOR hls4ml / FPGA PORTING
=========================================
512 real shots, 1000 ns readout window (500 raw samples @ 500 MHz), drawn at
random (diverse states) from the test set. For each model:

  <model>_input.npy   -> the FINAL preprocessed array the model's forward() accepts
  <model>_output.npy  -> the float (PyTorch) reference output on that input
  <model>_labels.npy  -> ground-truth qubit bit(s)

The input/output pairs are self-consistent: feeding *_input.npy to the float
checkpoint reproduces *_output.npy. Use them to check hls4ml bit-accuracy.

IMPORTANT: hls4ml converts ONLY the neural network. None of the preprocessing
below (demodulation, matched filters, FIR, integration) is part of the exported
graph -- it must be implemented separately in HLS/RTL (or the readout firmware).

Checkpoints live in ../checkpoints/ ; model classes in ../networks/.

-----------------------------------------------------------------------------
1) QubiCML  (networks/Qubic.py : Arxiv240618807FNN)   checkpoint: QubiCML_q0_1000ns.pth
   input  : (512, 2)   float32, range ~[0,1]      verify acc (q0): 0.959
   output : (512, 1)   P(qubit=|1>), sigmoid
   labels : (512,)     qubit-0 bit
   PREPROCESSING (per qubit, NOT in hls4ml):
     - complex demodulate the raw trace at the qubit IF (lo_i=cos, lo_q=-sin),
       freq f_q in {64.729,25.366,-24.79,-70.269,-127.282} MHz
     - time-average -> one (I,Q) point per qubit
     - min-max normalise to [0,1] using TRAIN-set per-feature min/max
   NOTE: one model per qubit (5 instances). min/max are train-set extremes
         (not stored in the checkpoint -- recompute from the calibration set).

2) KLiNQ student (networks/KLiNQ_StudentModel.py)     checkpoint: KLiNQ_student_paper_q0_1000ns.pth
   input  : (512, 31)  float32 (z-normed)          verify acc (q0): 0.965
   output : (512, 1)   sigmoid(logit) = P(|1>)
   labels : (512,)     qubit-0 bit
   PREPROCESSING (per qubit, NOT in hls4ml):
     - bin raw I and Q into 15 time-bins each -> 30 averaged features
     - 1 matched-filter scalar = raw trace . Wiener envelope (per-qubit,
       envelope = mean(gnd-exc)/var(gnd-exc) over the calibration set)
     - z-normalise the 30 avg features and the MF scalar -> 31-vector
   NOTE: one model per qubit. Needs a matched-filter envelope per qubit.

3) MCMit-CNN (networks/CNN.py : CNN)                  checkpoint: CNN_1000ns.pth
   input  : (512, 10, 25) float32 (per-channel z-normed)  verify per-qubit:
                                                  [0.92,0.69,0.93,0.74,0.96]
   output : (512, 5)   sigmoid(logits), one P(|1>) per qubit
   labels : (512, 5)   5 qubit bits
   PREPROCESSING (shared, NOT in hls4ml):
     - per-qubit complex demod + windowed-sinc FIR low-pass + downsample(20)
       -> 10 channels (5 qubits x I/Q), 25 time steps for 1000 ns
     - per-channel z-norm using TRAIN mean/std
   NOTE: this checkpoint was trained with the ds20 (downsample-20) pipeline,
         so its input is (10,25). The "her-preprocessing" variant in the paper
         uses downsample-40 (10,13) -- different FIR/demod; ask if you need it.

4) HERQULES (networks/HERQULES.py head = Net_rmf)     checkpoint: HERQULES_1000ns.pth
   input  : (512, 10)  float32  [MF_q0..q4 | RMF_q0..q4]   verify per-qubit:
                                                  [0.96,0.75,0.91,0.96,0.97]
   output : (512, 32)  logits over the 32 five-qubit states (argmax -> 5 bits)
   labels : (512, 5)   5 qubit bits
   PREPROCESSING (the heavy part, NOT in hls4ml):
     - per-qubit Butterworth demodulation
     - 5 matched-filter (MF) scores  + 5 relaxation-MF (RMF) scores per shot
       (each = demod trace . calibrated envelope) -> 10-vector
   NOTE: the .pth is only the tiny 10->10->20->32 MLP (1002 params); it is
         useless without the MF/RMF front-end. Full pipeline:
         trainers/HERQULES_original.py + trainers/train_HERQULES.py.
