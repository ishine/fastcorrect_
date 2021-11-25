# Fastcorrect Implementation Code from Authors

You can check [this page](https://openreview.net/forum?id=N3oi7URBakV)

***

We include the FastCorrect model and autoregressive model.
The models are fine-tuned on the AISHELL-1 dataset.

We only include the test script and the recognition result of AISHELL-1 ASR model.
We will release 1) the ASR model with its Espnet training config, 2) pretrained FastCorrect model on 400M crawled data, 3) full training script of FastCorrect, and 4) fine-tuned FastCorrect model on AISHELL-1 once our paper is accepted.


To obtain the results of FastCorrect and autoregressive models of AISHELL-1 as Table 1 in paper, there are several steps:

1) Download the FastCorrect and autoregressive models
    Download FastCorrect model from "https://drive.google.com/file/d/1Y0M8tAxJxPmkgxLt1NpFsiMPJhusHTA4/view?usp=sharing" and put it into "checkpoints/aishell_nat".
    Download autoregressive model from "https://drive.google.com/file/d/1YZ-TsYocTn7Gvnx8z3GoNZ1WTsHBe8pP/view?usp=sharing" and put it into "checkpoints/aishell_at"

2) Prepare environment
    `bash install_sctk.sh` to install sctk (please remember the bin folder of sctk, e.g., "./sctk/bin")
    Other environment preparations are included in the test shell script.

3) Inference
    `bash test_aishell_nat.sh` to obtain the result of FastCorrect model (resulting in "checkpoints/aishell_nat/results_aishell_b0").
    `bash test_aishell_at.sh` to obtain the result of autoregressive model (resulting in "checkpoints/aishell_at/results_aishell_b0").
    The architecture of result folder is:
    -- results_aishell_b0
      -- dev (result of dev set) 
          -- data.json (inference result json, "rec_text" is the inference result, "text" is the ground-truth text)
          -- dev_time.txt (in the format of sentence number, total time, average time)
      -- test (result of test set) 
          -- data.json
          -- test_time.txt

4) Calculate WER
    Assuming that the path of the bin folder of sctk is saved in `SCTK_BIN_DIR`
    
    Get WER without correction:
        `bash cal_wer_aishell.sh eval_data/dev $SCTK_BIN_DIR`  (Dev set, 4.46)
        `bash cal_wer_aishell.sh eval_data/test $SCTK_BIN_DIR`  (Test set, 4.83)

    Get WER of FastCorrect:
        `bash cal_wer_aishell.sh checkpoints/aishell_nat/results_aishell_b0/dev $SCTK_BIN_DIR`  (Dev set, 3.89)
        `bash cal_wer_aishell.sh checkpoints/aishell_nat/results_aishell_b0/test $SCTK_BIN_DIR`  (Test set, 4.16)

    Get WER of autoregressive:
        `bash cal_wer_aishell.sh checkpoints/aishell_at/results_aishell_b0/dev $SCTK_BIN_DIR`  (Dev set, 3.80)
        `bash cal_wer_aishell.sh checkpoints/aishell_at/results_aishell_b0/test $SCTK_BIN_DIR`  (Test set, 4.08)

Some explanations about folders and files:
    eval_data: containing the inference result of ASR model, which is the input of correction model.
    fastcorrect: containing the code of FastCorrect.
    sentence.bpe.model: sentencepiece model for tokenization
    espnet_wer_calculation: containing codes needed for WER calculation (copying from the Espnet codebase)
