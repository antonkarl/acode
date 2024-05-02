# acode

## Pipeline

Run `> pythyon3 pipeline.py`

- The folder defined as `original_data_dir` is expected to contain directories `PATIENTSDATA` and `CONTROLSTDATA`, which respectively contain folders `{P,C}-AudioFiles` <!-- and `{P,C}-Transcripts` -->

- Comment in/out calls to functions `run_diarisation()`, `run_asr()`, and `run_featurisation()`, to select which parts of the pipeline to run - ASR is extremely slow and is also not used in feature computation anyway.

- Due to data protection, scripts use local (on disk) copies of several speech processing models, which you need to download. If you don't download them, but you change the local file paths to public model names like most documentation recommends, the scripts will run and possibly transmit protected clinical data to external companies' servers in violation of policy. If you're not sure what you're doing, at least comment out Whisper ASR!


## Setup and requirements

- General python requirements: Pytorch (preferably with cuda), faster-whisper, transformers, pydub, pyannote.audio
- Can use virtual environment and install requirements with pip.

`python3 -m venv acodenv`
`source acodenv/bin/activate`

#### Diarisation

Requires both **LDC BPCSAD**, which itself depends on HTK, and **pyannote VAD + diarisation**.

Here are LDC's instructions: http://linguistic-data-consortium.github.io/ldc-bpcsad/install.html

This installation also worked:

`git clone https://github.com/Linguistic-Data-Consortium/ldc-bpcsad.git`

- From http://speech.ee.ntu.edu.tw/homework/, download "HTK 3.4.1 source code (zip)" (not one of the compiled binaries), and unzip its contents to `ldc-bpcsad/tools/htk/`
    -  do not just unzip as `ldc-bpcsad/tools/HTK-3.4.1/htk/`, this will not work for LDC.

`cd ldc-bpcsad`
`chmod u+x tools/htk/configure`
`pip install wheel`
`sudo apt-get install gcc-multilib make patch libsndfile1`
   *  specific to OS; don't do that unless you try the next step and failed with relevant error message
   
`sudo tools/install_htk.sh --njobs 1 --stage 2 PLACEHOLDER`
   *  don't need to replace the placeholder with any other word, it just needs text to fill the number of arguments. that argument will never be used for anything.

`pip install . `

* Now the LDC executable is at `acodenv/bin/ldc-bpcsad`. Location can end up elsewhere depending on OS and virtual environment setup.

- **TODO:** About pyannote models for offline use


#### ASR

- TODO: about models for offline use.

#### Acode features

- Install REAPER, TODO change this to Praat.



## TODO

- Use manually corrected file exported from ELAN as input to feature calculation
- Use Praat instead of REAPER for F0 tracking
- Documentation. For now see comments in code.
- Command line arguments for pipeline.py
- Add speech/articulation rate feature, phonemes or syllables per second
- Add initial pause features, e.g. tracking Participant's pauses of at least 1 second directly following Interviewer's questions
- Add other voice/spectral features


