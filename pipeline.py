import diarise as dia
import featurise as fea
import asrs as asr
from util import *


# organise paths to original recordings + transcripts, 
# and output of diarisations and ASR,
# assuming they (will) exist in a certain directory structure
def find_files(recording_dir = '../datadrive/', output_dir="./output/"):
    control_prefix = f'{recording_dir}CONTROLSDATA/C-'
    patient_prefix = f'{recording_dir}PATIENTSDATA/P-'
    
    control_wavs = glob.glob(control_prefix+'AudioFiles/*.wav')
    patient_wavs = glob.glob(patient_prefix+'AudioFiles/*.wav')
    
    for fdir in [output_dir, f'{output_dir}diarisation',f'{output_dir}asr']:
        if not os.path.exists(fdir):
            os.mkdir(fdir)
    
    files_dict = {'control' : {fn(f) : 
            {'wav' : f, 
             'xcp': f'{control_prefix}Transcripts/{fn(f)}-trans.txt',
             'pya': f'{output_dir}diarisation/{fn(f)}-PyaVad.txt',
             'lab': f'{output_dir}diarisation/{fn(f)}.lab',
             'ldc': f'{output_dir}diarisation/{fn(f)}-LdcPya.txt'} 
            for f in control_wavs },
        'patient' : {fn(f) : 
            {'wav' : f, 
             'xcp': f'{patient_prefix}Transcripts/{fn(f)}-trans.txt',
             'pya': f'{output_dir}diarisation/{fn(f)}-PyaVad.txt',
             'lab': f'{output_dir}diarisation/{fn(f)}.lab',
             'ldc': f'{output_dir}diarisation/{fn(f)}-LdcPya.txt'}
            for f in patient_wavs } }
            
    return files_dict 
    
    

def run_diarisation(data_files, sad_cmd):
    # sad_cmd is path to executable from installing LDC-bpcsad
    #   where to find this depends on your OS and virtual environment.
    # http://linguistic-data-consortium.github.io/ldc-bpcsad
    # LDC sad also requires installing HTK 
    # http://speech.ee.ntu.edu.tw/homework/ 
    #  v3.4.1, download the source code (not the compiled binary) 
    #   and follow LDC's instructions for installing their HTK patch

    # decide how to label speakers in output
    labels_dict = {"main": "V", "interviewer": "S", "unknown": "XX"}
    
    vad = dia.setup_pya_vad("localmodels/pyannote-speakerdia-3.1/diarization31config.yaml")
    # to use pyannote vad locally without access token,
    # download the pytorch model.bin and config.yaml 
    #   from https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/tree/main
    #   also the model.bin and config.yaml from https://huggingface.co/pyannote/segmentation-3.0/tree/main
    # and download config.yaml and handler.py
    #   from https://huggingface.co/pyannote/speaker-diarization-3.1/tree/main
    # then edit the embedding and segmentation in the pyvad config file defined above
    #   to point at your local wespeaker embedding & segmentation3.0 model.bin files
    # ex. <embedding: localmodels/pyannote-speakerdia-3.1/wespeaker-voxceleb-resnet34-LMpytorch_model.bin>
    #     <segmentation: localmodels/pyannote-speakerdia-3.1/segmentation30pytorch_model.bin>
    
    for speaker,file_paths in data_files['control'].items():
        dia.diarise_one(file_paths,vad,sad_cmd,labels_dict)
        
    for speaker,file_paths in data_files['patient'].items():
        dia.diarise_one(file_paths,vad,sad_cmd,labels_dict)
    


def run_featurisation(data_files, f0_extractor, save_path):

    #f0_extractor is currently (temporarily)
    # the path to executable from installing reaper
    # https://github.com/google/REAPER/
    # to be replaced with Praat formant tracking hopefully
    
    # key to lookup what file to use as pause segmentation
    segmentation = 'ldc'
    
    final_features = []
    
    for speaker,file_paths in data_files['control'].items():
        final_features.append([speaker, 'Control', fea.featurise_one(file_paths['wav'],file_paths[segmentation],f0_extractor) ])
        
    for speaker,file_paths in data_files['patient'].items():
        final_features.append([speaker, 'Patient', fea.featurise_one(file_paths['wav'],file_paths[segmentation],f0_extractor) ])
        
    with open(save_path,'w') as handle:
        handle.write('\t'.join(['sample_id', 'group', 'avg_speech_duration', 'avg_pause_duration', 'pauses_per_minute','percent_speaking','pitch_range_global','pitch_range_local'])+'\n')
        for spk, group, features in final_features:
            handle.write(f'{spk}\t{group}\t'+'\t'.join([str(round(f,4)) for f in features])+'\n')



def run_asr(data_files):

    # comment out to select ASR and diarisation options
    # whisper takes the longest time to run.
    asr_list = ['w2v2','whisper']
    dia_list = ['pya','ldc']

    for participant_group in ['control', 'patient']:
        for speaker,file_paths in data_files[participant_group].items():
            print(f'ASR for {speaker}')
            for a in asr_list:
                for d in dia_list:
                    print(f'  ... {a} - {d}')
                    asr.asr_one(file_paths,a,d)



# refer to comments in each pipeline function
def run():
    original_data_dir = '/home/cati/proj/acode/datadrive/'
    sad_executable = '/home/cati/proj/acode/processing/acode/acodenv/bin/ldc-bpcsad'
    f0_extractor = "./localmodels/REAPER/build/reaper"
    save_dir = './output/'
    
    data_files = find_files(original_data_dir, save_dir)
    
    run_diarisation(data_files, sad_executable)
    run_featurisation(data_files,f0_extractor, f'{save_dir}FEATURES.tsv')
    #run_asr(data_files)
    
if __name__ == "__main__":
    run()
    
   
