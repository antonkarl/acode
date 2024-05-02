import os, subprocess, glob
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment, Timeline, Annotation, notebook
import torchaudio
import numpy as np
from util import *


# ----------------------------
# 
# --- segmentation + diarisation ---
#
# ----------------------------

# Outline:
# - Run pyannote diarisation
# - Separately, run LDC segmentation
# - Project pyannote's speaker labels onto LDC's segments
# - Reason: LDC finds better segment boundaries, but can't label them.

# Notes: LDC only finds non-overlapping segments.
#   If two people are talking at the same time,
#   this will need to be manually corrected.
# This implementation is for recordings with 2 speakers.
#   Otherwise edit get_pya_vad() and transfer_2spk_labels()



# initialise pyannote
def setup_pya_vad(vad_yaml_path):
    vad = Pipeline.from_pretrained(vad_yaml_path)
    
    # min_duration_on is min speech duration
    # should be 250ms to match cho et al 2021,
    #   but this parameter can't actually be set in pyannote 3.1
    # min_duration_off is min pause duration, should be 150ms
    HYPER_PARAMETERS = { 'segmentation' :
    {#"min_duration_on": 0.25,
    "min_duration_off": 0.15}
    }
    
    # pyannote 3.1 has default minimum pause duration 0.0 not 0.15, 
    #  but even pyannote with 0.0 finds less short pauses than LDC does with 0.15.
    # therefore, leaving hyperparameter instantiation commented out to run with default.
    #vad.instantiate(HYPER_PARAMETERS)
    #pth = vad.parameters(instantiated=True)
    #print(pth)
    return vad
    

# run pyannote on a file
#  or load its output from file if it already existed
def get_pya_vad(wav,save_path,labels_dict,vad_pipeline):
    if os.path.exists(save_path):
        with open(save_path,'r') as handle:
            pya_data = handle.read().splitlines()
        pya_vad = Annotation()
        pya_data = [l.split('\t') for l in pya_data]
        for l,s,e in pya_data:
            pya_vad[Segment(float(s),float(e))]=l

    else:
        wf, sr = torchaudio.load(wav) # supposedly faster
        with ProgressHook() as hook:
            pya_vad = vad_pipeline({"waveform": wf, "sample_rate": sr}, num_speakers=2, hook=hook)
            
        main_speaker = pya_vad.chart()[0][0]
        other_speaker = pya_vad.chart()[1][0]
        with open(save_path,'w') as handle:
            for segment,track,label in pya_vad.itertracks(yield_label=True):
                if label == main_speaker:
                    output_label = labels_dict['main']
                elif label == other_speaker:
                    output_label = labels_dict['interviewer']
                else:
                    print("WARNING! edit the program before using it on files with more than 2 speakers")
                    output_label = label
                handle.write(f'{output_label}\t{segment.start}\t{segment.end}\n')
    return pya_vad
    
    
    
    
# run LDC executable sad_cmd for wav file_path
# return path to sad_cmd output file
#   if it already existed, don't re-run
# minimum duration for speech segments: 250ms
# minimum duration for silent pauses: 150ms
#  following cho et al 2021, Lexical and Acoustic 
#  Characteristics of Young and Older Healthy Adults
def ldc_sad(wav_path, sad_cmd, lab_path):
    
    if not(os.path.exists(lab_path)):
        sad_proc = subprocess.call([sad_cmd, "--channel", "1", 
        "--output-dir", os.path.dirname(lab_path), "--speech", "0.250", 
        "--nonspeech", "0.150", wav_path])
    
    return read_ldc(lab_path)
    
    
    
# read ldcbpsad output file to unlabelled pyannote Timeline
def read_ldc(ldc_path):
    timeline = Timeline()
    with open(ldc_path, 'r') as handle:
        ldc = handle.read().splitlines()
    ldc = [l.split('\t') for l in ldc]
    ldc = [l for l in ldc if 'non-speech' not in l]
    for s,e,_ in ldc:
        timeline.add(Segment(float(s),float(e)))
    return timeline
    
    
# heuristic transfer speakers from broad pyannote diarisation
#   onto sensitive unlabelled LDC segmentation
# speaker labelling assumes the patient/control speaks most
#  if the interviewer speaks most, the automatic speaker labels will be wrong (switched)
def transfer_2spk_labels(ldc,pya,ldict):
    main_speaker = pya.chart()[0][0]
    other_speaker = pya.chart()[1][0]
    
    transferred = Annotation()
    for lseg in ldc:
        
        # find all labelled PYA segments that overlap with the LDC segment
        candidates = [(pseg, label) for pseg,track,label in pya.itertracks(yield_label=True) if lseg & pseg]
        if not candidates:
            print(f'NOTICE: PyAnnote did not label speech for segment {lseg}')
            transferred[lseg] = ldict['unknown']
        else:
            candidates = [spk for seg,spk in candidates]
            if main_speaker in candidates:
                transferred[lseg] = ldict['main']
            else:
                transferred[lseg] = ldict['interviewer']
    return transferred





# save pyannote Annotation 
# in format that can be imported to Elan
# https://www.mpi.nl/corpus/html/elan/ch01s04s02.html#Sec_Importing_CSV_Tab-delimited_Text_Files
def pya2eln(annot,save_path):
    # elan requires Annotation column to import data
    # so if you want to use speaker labels as tier names
    # an extra empty column must be added as segment content annotation
    eln = [f'{label}\t{segment.start}\t{segment.end}\t' for segment,track,label in annot.itertracks(yield_label=True)]
    eln = '\n'.join(eln)
    with open(save_path,'w') as handle:
        handle.write(eln)



# perform full diarisation pipeline for one file
def diarise_one(file_paths,vad,sad_cmd,labels_dict):
    wav = file_paths['wav']
    lab = file_paths['lab']
    pya = file_paths['pya']
    ldc = file_paths['ldc']
    
    print(f'Diarising sample {fn(wav)}')
    
    tmp_wav = wav16mono(wav)
    
    ldc_vad = ldc_sad(tmp_wav, sad_cmd, lab)
    pya_vad = get_pya_vad(tmp_wav,pya,labels_dict,vad)
    relabeled = transfer_2spk_labels(ldc_vad,pya_vad,labels_dict)
    pya2eln(relabeled,ldc)



