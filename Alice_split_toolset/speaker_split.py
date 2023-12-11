from pyannote.audio import Pipeline
import torch
import os
import datetime
import argparse


def speaker_identify(wav_file, num_speakers=2):
    speaker_ts_lst = []
    diarization = pipeline(wav_file, min_speakers=num_speakers, max_speakers=num_speakers+2)
    last_speaker = ""
    start = stop = 0.0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
#         if speaker == last_speaker:
#             stop = turn.end
#         else:
#             if last_speaker != "" and stop > start + 2:
#                 speaker_ts_lst.append([last_speaker, start, stop])
#             last_speaker = speaker
#             start = turn.start
#             stop = turn.end
#     speaker_ts_lst.append([last_speaker, start, stop])
        start = turn.start
        stop = turn.end
        if stop > start + 2:
            speaker_ts_lst.append([speaker, start, stop])
    return speaker_ts_lst

def split_wav(speaker_ts_lst, wav_file, output_path):
    i = 0
    for speaker, start, stop in speaker_ts_lst:
        start_time = datetime.timedelta(seconds=start)
        stop_time = datetime.timedelta(seconds=stop)
        if not os.path.exists(f"{output_path}/{speaker}"):
            os.mkdir(f"{output_path}/{speaker}")
        os.system(f"ffmpeg -i {wav_file}  -ss {start_time} -to {stop_time} -c copy {output_path}/{speaker}/p{i}.wav")
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num_speakers', type=int, help='说话人数')
    args = parser.parse_args()
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cuda"))
    path = "upload"
    speaker_ts_lst = []
    wav_files = os.listdir(path)
    for wav_file in wav_files: 
        if wav_file.endswith(".wav") or wav_file.endswith(".WAV") or wav_file.endswith(".mp3") or wav_file.endswith(".MP3"):
            # apply pretrained pipeline
            wav_file = os.path.join(path, wav_file)
            speaker_ts_lst = speaker_identify(wav_file, args.num_speakers)
            print("speaker_ts_lst=", speaker_ts_lst)
            split_wav(speaker_ts_lst, wav_file, path)

            break  # 为了避免多个wav文件中speaker01指代不同角色